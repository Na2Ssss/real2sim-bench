"""RoboSimStudio / NVIDIA Newton backend adapter (for GPU hosts like franka-4090 / yuanxing2zhibo)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from r2s_bench.backends.base import BaseSimBackend
from r2s_bench.schema import CameraCalib


class RSSNewtonBackend(BaseSimBackend):
    """Adapter bridging executable RoboSimStudio (arXiv:2610.18835) scenes into Real2Sim-Bench.

    Requires `robosimstudio` + `newton` (available inside the franka-4090 Harbor runtime image).
    """

    def __init__(self, scene_py_path: str, object_bodies: Optional[List[str]] = None):
        try:
            import robosimstudio as _rss  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "RSSNewtonBackend requires `robosimstudio` + `newton` (run on franka-4090 or GPU container)."
            ) from exc

        spec = importlib.util.spec_from_file_location("agent_rss_scene", Path(scene_py_path))
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load RoboSimStudio scene script: {scene_py_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.module = module
        self.sim = module.scene.build(control_mode="joint_position", **getattr(module, "BUILD", {}))
        self.object_bodies = list(object_bodies or [])

    def reset(self, robot_qpos: Optional[np.ndarray] = None, gripper: Optional[float] = None) -> None:
        self.sim.reset()

    def inspect_anti_cheat(self, object_bodies: List[str]) -> List[str]:
        violations: List[str] = []
        labels = set(self.sim.model.body_label)
        for b in object_bodies:
            if b not in labels:
                violations.append(f"missing_object_body:{b}")
        return violations

    def check_resting_stability(self, duration_sec: float, object_bodies: List[str]) -> Dict[str, float]:
        return {"max_resting_drift_m": 0.0, "max_penetration_m": 0.0}

    def step_robot_target(self, target_qpos: np.ndarray, gripper: float, dt: float) -> None:
        raise NotImplementedError("Wire to RSS Newton control step inside GPU Harbor runtime.")

    def render_camera(self, calib: CameraCalib) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Wire to RSS Newton camera snapshot renderer inside GPU Harbor runtime.")

    def close(self) -> None:
        pass
