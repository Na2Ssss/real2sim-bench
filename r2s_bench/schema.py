"""Strict data contracts for Real2Sim-Bench v0.2.

Separates:
- Public inputs visible to the Real2Sim Agent (Track A: init frames; Track B: + SysID demos)
- Hidden verifier assets (Held-out multi-pose teleop replays, ground-truth calibration & masks)
- Standardized Agent submission bundle & Grader evaluation report
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np


@dataclass
class CameraCalib:
    """OpenCV-convention pinhole camera calibration relative to robot base_link."""

    name: str
    serial: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    cam_to_base: List[List[float]]  # 4x4 row-major homogeneous matrix
    quality_iou: Optional[float] = None

    @property
    def T_base_cam(self) -> np.ndarray:
        return np.asarray(self.cam_to_base, dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled(self, dst_width: int, dst_height: int) -> "CameraCalib":
        sx = dst_width / float(self.width)
        sy = dst_height / float(self.height)
        return CameraCalib(
            name=self.name,
            serial=self.serial,
            width=dst_width,
            height=dst_height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            cam_to_base=self.cam_to_base,
            quality_iou=self.quality_iou,
        )


@dataclass
class EpisodeSpec:
    """Metadata for a single real-robot teleoperation trajectory."""

    episode_id: str
    source_dataset: str  # e.g., "droid", "rh20t", "furniturebench", "franka_gello"
    robot_type: str  # e.g., "franka_robotiq_2f85", "franka_panda", "g1d"
    fps: float
    num_steps: int
    language_instruction: str
    split: str  # "sysid_public" or "heldout_eval"
    cameras: List[CameraCalib]
    video_paths: Dict[str, str]  # cam_name -> relative path to mp4
    trajectory_path: str  # relative path to normalized npz (qpos, gripper, cartesian, timestamps)


@dataclass
class CaseBundleSpec:
    """Complete specification for a Real2Sim-Bench evaluation case."""

    case_id: str
    tier: str  # "tier1_rigid", "tier2_articulated", "tier3_precision", "tier4_deformable"
    track: str  # "track_a_static" or "track_b_sysid_multipose"
    source_dataset: str
    robot_type: str
    scene_cluster_id: str
    public_episodes: List[EpisodeSpec]
    heldout_episodes: List[EpisodeSpec]
    provide_extrinsics: bool = True  # False => Hard/Uncalibrated Track

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: Path) -> "CaseBundleSpec":
        raw = json.loads(path.read_text())

        def _parse_ep(d: dict) -> EpisodeSpec:
            d = dict(d)
            d["cameras"] = [CameraCalib(**c) for c in d["cameras"]]
            return EpisodeSpec(**d)

        raw["public_episodes"] = [_parse_ep(e) for e in raw["public_episodes"]]
        raw["heldout_episodes"] = [_parse_ep(e) for e in raw["heldout_episodes"]]
        return cls(**raw)


@dataclass
class AgentSubmission:
    """Standard output contract submitted by a Real2Sim Agent (e.g., GPT-6 / Claude / Codex)."""

    case_id: str
    backend: str  # "mujoco" or "rss_newton"
    scene_file: str  # path to MJCF (.xml) or RoboSimStudio executable python scene (.py)
    object_bodies: List[str]  # names of dynamic/articulated interactive bodies created by Agent
    cameras_override: Optional[List[CameraCalib]] = None  # required only in Uncalibrated Track
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeEvalMetrics:
    episode_id: str
    anti_cheat_passed: bool
    anti_cheat_violations: List[str]
    resting_max_drift_m: float
    resting_max_penetration_m: float
    l1_robot_mask_iou: float
    l1_fg_psnr_db: float
    l2_obj_track_mean_iou: Optional[float] = None
    l2_obj_track_final_iou: Optional[float] = None
    l2_centroid_mean_err_px: Optional[float] = None
    l2_contact_onset_err_sec: Optional[float] = None
    l3_replay_score: Optional[float] = None


@dataclass
class EvalReport:
    case_id: str
    submission_valid: bool
    aggregate_score: float  # 0.0 to 1.0 composite score
    per_episode: List[EpisodeEvalMetrics]
    failure_category: str  # "pass", "schema_error", "cheat_detected", "calib_fail", "physics_unstable", "dynamics_fail"
    notes: str = ""

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
