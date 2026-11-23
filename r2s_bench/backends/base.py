"""Engine-agnostic simulation backend protocol for Real2Sim-Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from r2s_bench.schema import CameraCalib


class BaseSimBackend(ABC):
    """Unified simulation & rendering interface implemented by MuJoCo and RoboSimStudio/Newton."""

    @abstractmethod
    def reset(self, robot_qpos: Optional[np.ndarray] = None, gripper: Optional[float] = None) -> None:
        """Reset simulation state while keeping dynamic scene objects at their initial poses."""

    @abstractmethod
    def inspect_anti_cheat(self, object_bodies: List[str]) -> List[str]:
        """Return list of anti-cheat violation messages (empty list means clean)."""

    @abstractmethod
    def check_resting_stability(self, duration_sec: float, object_bodies: List[str]) -> Dict[str, float]:
        """Step physics with zero robot motion to measure object drift and inter-penetration."""

    @abstractmethod
    def step_robot_target(self, target_qpos: np.ndarray, gripper: float, dt: float) -> None:
        """Drive robot actuators toward target_qpos and gripper opening over interval dt."""

    @abstractmethod
    def render_camera(self, calib: CameraCalib) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render synchronized (rgb [H,W,3 uint8], robot_mask [H,W bool], obj_mask [H,W bool])."""

    @abstractmethod
    def close(self) -> None:
        """Release renderer and simulator resources."""
