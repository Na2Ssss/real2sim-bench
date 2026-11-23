"""Unified Cross-Scene Real-to-Sim Transfer Framework.

Distills all verified engineering lessons from Demo #1 (AUTOLab, 100 eps),
Demo #2 (AUTOLab articulated drawer, 74 eps), and Demo #3 (GuptaLab motorized desk, 74 eps)
into reusable, scene-agnostic primitives so migrating to a brand-new DROID lab/task
requires only a concise `SceneTransferConfig` instead of bespoke scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import mujoco
import numpy as np

from r2s_bench.eval.task_success import (
    RolloutMonitor,
    TaskEvalResult,
    evaluate_block_in_drawer,
    evaluate_joint_state,
    evaluate_object_in_container,
    evaluate_object_on_target,
    evaluate_object_out_of_container,
    evaluate_particles_in_target,
)
from r2s_bench.schema import CameraCalib

SupportMode = Literal["fixed_table", "motorized_desk", "articulated_workbench"]
AssetKind = Literal["rigid_hollow", "rigid_solid", "articulated_slide", "articulated_hinge"]
PredicateKind = Literal[
    "object_in_container",
    "object_out_of_container",
    "joint_state",
    "object_on_target",
    "particles_in_target",
    "block_in_drawer",
]


@dataclass
class AssetSpec:
    """Declarative specification for one manipulable or articulated scene asset."""

    name: str
    sam_prompt: str
    kind: AssetKind = "rigid_solid"
    target_height_m: float = 0.08
    target_diameter_m: Optional[float] = None
    wall_thickness_m: float = 0.003
    taper_bot_scale: Optional[float] = None
    coacd_max_hulls: int = 24
    mass_kg: float = 0.08
    friction: Tuple[float, float, float] = (1.0, 0.01, 0.001)


@dataclass
class OutcomeSpec:
    """Single clean outcome predicate for a task (no brittle process gates)."""

    predicate: PredicateKind
    params: Dict[str, object] = field(default_factory=dict)


@dataclass
class SceneTransferConfig:
    """Complete declarative config to onboard any new DROID scene onto A100 MuJoCo 4.0.1."""

    scene_id: str
    lab: str
    date: str
    task_description: str
    support_mode: SupportMode
    assets: List[AssetSpec]
    outcome: OutcomeSpec
    table_sam_prompt: str = "tabletop"
    dual_view_weights: Tuple[float, float] = (0.75, 0.25)
    cam_shift_guard_m: float = 0.04
    default_table_z: float = -0.018
    mujoco_gl: str = "egl"
    mp_start_method: str = "spawn"  # Mandatory on Linux to prevent fork+EGL deadlocks

    def to_json(self, path: Optional[Path] = None) -> str:
        text = json.dumps(asdict(self), indent=2, ensure_ascii=False)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return text

    @classmethod
    def from_json(cls, path: Path) -> "SceneTransferConfig":
        raw = json.loads(path.read_text())
        raw["assets"] = [AssetSpec(**a) for a in raw["assets"]]
        raw["outcome"] = OutcomeSpec(**raw["outcome"])
        raw["dual_view_weights"] = tuple(raw["dual_view_weights"])
        return cls(**raw)


def select_camera_extrinsics(
    T_raw: np.ndarray,
    T_ref: Optional[np.ndarray],
    max_shift_m: float = 0.04,
) -> Tuple[np.ndarray, bool]:
    """Moving-tripod guard (learned from Demo #3 GuptaLab where Cam 2 shifted 21.7 cm mid-session).

    Uses hand-eye refined `T_ref` only when the episode's raw extrinsic translation agrees with
    `T_ref` within `max_shift_m`; otherwise falls back safely to the episode's own `T_raw`.
    Returns `(T_selected, used_refined)`.
    """
    T_raw = np.asarray(T_raw, dtype=np.float64)
    if T_ref is None:
        return T_raw, False
    T_ref = np.asarray(T_ref, dtype=np.float64)
    shift = float(np.linalg.norm(T_raw[:3, 3] - T_ref[:3, 3]))
    if shift <= max_shift_m:
        return T_ref, True
    return T_raw, False


def triangulate_two_views(
    c1: CameraCalib,
    uv1: Tuple[float, float],
    c2: CameraCalib,
    uv2: Tuple[float, float],
) -> Tuple[np.ndarray, float]:
    """Pure Frame-0 dual-view 3D ray triangulation (zero trajectory leakage).

    Essential for height-adjustable motorized desks (Demo #3 GuptaLab) where `table_z`
    varies by >10 cm across episodes. Returns `(point_3d_base, ray_miss_dist_m)`.
    """
    rays = []
    for cal, (u, v) in ((c1, uv1), (c2, uv2)):
        T = np.asarray(cal.cam_to_base, dtype=np.float64)
        o = T[:3, 3]
        d_cam = np.array([(u - cal.cx) / cal.fx, (v - cal.cy) / cal.fy, 1.0], dtype=np.float64)
        d = T[:3, :3] @ d_cam
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        rays.append((o, d))

    (o1, d1), (o2, d2) = rays
    w0 = o1 - o2
    b = float(np.dot(d1, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = max(1.0 - b * b, 1e-6)
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    p1 = o1 + max(s, 0.05) * d1
    p2 = o2 + max(t, 0.05) * d2
    mid = 0.5 * (p1 + p2)
    miss = float(np.linalg.norm(p1 - p2))
    return mid, miss


def evaluate_scene_outcome(
    config: SceneTransferConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    monitor: Optional[RolloutMonitor] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> TaskEvalResult:
    """Dispatch single outcome predicate defined in `config.outcome`."""
    params = dict(config.outcome.params)
    if overrides:
        params.update(overrides)
    pred = config.outcome.predicate

    if pred == "object_in_container":
        return evaluate_object_in_container(model=model, data=data, monitor=monitor, **params)
    if pred == "object_out_of_container":
        return evaluate_object_out_of_container(model=model, data=data, monitor=monitor, **params)
    if pred == "joint_state":
        return evaluate_joint_state(model=model, data=data, monitor=monitor, **params)
    if pred == "object_on_target":
        return evaluate_object_on_target(model=model, data=data, monitor=monitor, **params)
    if pred == "particles_in_target":
        return evaluate_particles_in_target(model=model, data=data, monitor=monitor, **params)
    if pred == "block_in_drawer":
        return evaluate_block_in_drawer(model=model, data=data, monitor=monitor, **params)
    raise ValueError(f"Unsupported outcome predicate: {pred}")


# ---------------------------------------------------------------------------
# Ready-to-Run Scene Configs (Existing 3 Demos + 4 New Calibrated DROID Labs)
# ---------------------------------------------------------------------------

PRESET_SCENES: Dict[str, SceneTransferConfig] = {
    "demo1_cup_in_bowl": SceneTransferConfig(
        scene_id="demo1_cup_in_bowl",
        lab="AUTOLab",
        date="2023-07-27",
        task_description="Put the orange cup in the black bowl",
        support_mode="fixed_table",
        default_table_z=-0.018,
        assets=[
            AssetSpec("orange_cup", "orange cup", "rigid_hollow", target_height_m=0.105, taper_bot_scale=0.68, coacd_max_hulls=24),
            AssetSpec("black_bowl", "black bowl", "rigid_hollow", target_height_m=0.068, coacd_max_hulls=20),
        ],
        outcome=OutcomeSpec("object_in_container", {"object_body": "orange_cup", "container_body": "black_bowl", "max_xy_dist_m": 0.085}),
    ),
    "demo2_block_in_drawer": SceneTransferConfig(
        scene_id="demo2_block_in_drawer",
        lab="AUTOLab",
        date="2023-10-21",
        task_description="Put the blue lego brick in the open drawer and close it",
        support_mode="articulated_workbench",
        assets=[
            AssetSpec("blue_block", "blue lego block", "rigid_solid", target_height_m=0.036, mass_kg=0.045, friction=(2.2, 0.03, 0.003)),
            AssetSpec("drawer_box", "open drawer", "articulated_slide", target_height_m=0.10),
        ],
        outcome=OutcomeSpec("block_in_drawer", {"block_body": "blue_block", "drawer_body": "drawer_box", "drawer_joint": "drawer_slide"}),
    ),
    "demo3_pouring_beans": SceneTransferConfig(
        scene_id="demo3_pouring_beans",
        lab="GuptaLab",
        date="2023-04-30",
        task_description="Pour the contents of the red cup into the bowl",
        support_mode="motorized_desk",
        assets=[
            AssetSpec("red_cup", "red cup", "rigid_hollow", target_height_m=0.112, target_diameter_m=0.074, coacd_max_hulls=24),
            AssetSpec("target_bowl", "bowl", "rigid_hollow", target_height_m=0.065, coacd_max_hulls=20),
        ],
        outcome=OutcomeSpec("particles_in_target", {"target_body": "target_bowl", "particle_prefix": "bean_", "min_particles_in": 1}),
    ),
    # --- Brand-new DROID Lab Migration Candidates (all >=180 calibrated multi-view episodes) ---
    "rpl_can_in_bowl": SceneTransferConfig(
        scene_id="rpl_can_in_bowl",
        lab="RPL",
        date="2023-11-30",
        task_description="Put the can in the bowl",
        support_mode="fixed_table",
        assets=[
            AssetSpec("target_can", "can", "rigid_solid", target_height_m=0.115, target_diameter_m=0.066, coacd_max_hulls=16),
            AssetSpec("target_bowl", "bowl", "rigid_hollow", target_height_m=0.070, coacd_max_hulls=20),
        ],
        outcome=OutcomeSpec("object_in_container", {"object_body": "target_can", "container_body": "target_bowl", "max_xy_dist_m": 0.09}),
    ),
    "tri_marker_from_cup": SceneTransferConfig(
        scene_id="tri_marker_from_cup",
        lab="TRI",
        date="2023-09-21",
        task_description="Remove the marker from the cup",
        support_mode="fixed_table",
        assets=[
            AssetSpec("marker", "marker pen", "rigid_solid", target_height_m=0.135, target_diameter_m=0.016, mass_kg=0.025),
            AssetSpec("holder_cup", "cup", "rigid_hollow", target_height_m=0.100, coacd_max_hulls=20),
        ],
        outcome=OutcomeSpec("object_out_of_container", {"object_body": "marker", "container_body": "holder_cup", "min_xy_dist_m": 0.095}),
    ),
    "rail_open_drawer": SceneTransferConfig(
        scene_id="rail_open_drawer",
        lab="RAIL",
        date="2023-07-14",
        task_description="Open the drawer",
        support_mode="articulated_workbench",
        assets=[
            AssetSpec("drawer_box", "drawer", "articulated_slide", target_height_m=0.12),
        ],
        outcome=OutcomeSpec("joint_state", {"joint_name": "drawer_slide", "q_init": 0.0, "q_target": -0.20, "min_progress_ratio": 0.75}),
    ),
    "iprl_lid_on_pot": SceneTransferConfig(
        scene_id="iprl_lid_on_pot",
        lab="IPRL",
        date="2024-01-01",
        task_description="Put the lid on the pot",
        support_mode="fixed_table",
        assets=[
            AssetSpec("pot_lid", "pot lid", "rigid_solid", target_height_m=0.045, coacd_max_hulls=20),
            AssetSpec("cooking_pot", "pot", "rigid_hollow", target_height_m=0.110, coacd_max_hulls=24),
        ],
        outcome=OutcomeSpec("object_on_target", {"object_body": "pot_lid", "target_body": "cooking_pot", "max_xy_dist_m": 0.085}),
    ),
}
