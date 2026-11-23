"""MuJoCo backend with exact principal-point pinhole rendering and anti-cheat checks."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from r2s_bench.backends.base import BaseSimBackend
from r2s_bench.schema import CameraCalib


def opencv_cam_to_mujoco_pose(T_base_cam_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV cam2base (+X right, +Y down, +Z forward) to MuJoCo (+X right, +Y up, -Z forward)."""
    R_cv_to_gl = np.diag([1.0, -1.0, -1.0])
    pos = T_base_cam_cv[:3, 3].copy()
    rot_gl = T_base_cam_cv[:3, :3] @ R_cv_to_gl
    q_xyzw = R.from_matrix(rot_gl).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return pos, q_wxyz


class MuJoCoBackend(BaseSimBackend):
    """Production MuJoCo backend supporting calibrated projection and closed-book grading."""

    def __init__(
        self,
        xml_path: Optional[str] = None,
        xml_string: Optional[str] = None,
        object_bodies: Optional[List[str]] = None,
        robot_body_prefixes: Tuple[str, ...] = ("link", "hand", "left_finger", "right_finger", "panda", "base"),
        camera_name: str = "eval_cam",
    ):
        if xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        elif xml_string is not None:
            self.model = mujoco.MjModel.from_xml_string(xml_string)
        else:
            raise ValueError("Provide either xml_path or xml_string.")

        self.data = mujoco.MjData(self.model)
        self.object_bodies = list(object_bodies or [])
        self.camera_name = camera_name
        self._renderers: Dict[Tuple[int, int], mujoco.Renderer] = {}

        self.obj_geom_ids: Set[int] = self._collect_body_geoms(self.object_bodies)
        self.robot_geom_ids: Set[int] = self._collect_prefix_geoms(robot_body_prefixes) - self.obj_geom_ids

    def _collect_body_geoms(self, body_names: List[str]) -> Set[int]:
        target_bids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_names
        }
        target_bids.discard(-1)
        return {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in target_bids}

    def _collect_prefix_geoms(self, prefixes: Tuple[str, ...]) -> Set[int]:
        ids: Set[int] = set()
        for g in range(self.model.ngeom):
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]) or ""
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if any(bname.startswith(p) or gname.startswith(p) for p in prefixes):
                ids.add(g)
        return ids

    def _get_renderer(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def reset(self, robot_qpos: Optional[np.ndarray] = None, gripper: Optional[float] = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if robot_qpos is not None:
            n = min(len(robot_qpos), self.model.nq)
            self.data.qpos[:n] = robot_qpos[:n]
            if self.model.nu >= n:
                self.data.ctrl[:n] = robot_qpos[:n]
        if gripper is not None and self.model.nq >= 9:
            finger = float(np.clip(0.04 * (1.0 - gripper), 0.0, 0.04))
            self.data.qpos[7] = finger
            self.data.qpos[8] = finger
            if self.model.nu >= 8:
                self.data.ctrl[7] = 255.0 * (1.0 - float(np.clip(gripper, 0.0, 1.0)))
        mujoco.mj_forward(self.model, self.data)

    def inspect_anti_cheat(self, object_bodies: List[str]) -> List[str]:
        """Verify that interactive objects are genuinely dynamic and not secretly actuated."""
        violations: List[str] = []
        actuated_jids = {
            int(self.model.actuator_trnid[a, 0])
            for a in range(self.model.nu)
            if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
        }

        for bname in object_bodies:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                violations.append(f"missing_object_body:{bname}")
                continue
            jnt_num = int(self.model.body_jntnum[bid])
            if jnt_num == 0:
                violations.append(f"static_non_dynamic_object:{bname}")
                continue
            jnt_adr = int(self.model.body_jntadr[bid])
            for j in range(jnt_adr, jnt_adr + jnt_num):
                if j in actuated_jids:
                    violations.append(f"secretly_actuated_object:{bname}")
            if float(self.model.body_mass[bid]) <= 1e-5:
                violations.append(f"zero_or_invalid_mass:{bname}")
        return violations

    def check_resting_stability(self, duration_sec: float, object_bodies: List[str]) -> Dict[str, float]:
        bids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in object_bodies
        ]
        bids = [b for b in bids if b >= 0]
        init_pos = {b: self.data.xpos[b].copy() for b in bids}

        n_steps = max(1, int(round(duration_sec / self.model.opt.timestep)))
        max_pen = 0.0
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
            for c in range(self.data.ncon):
                dist = float(self.data.contact[c].dist)
                if dist < 0.0:
                    max_pen = max(max_pen, -dist)

        max_drift = max((float(np.linalg.norm(self.data.xpos[b] - init_pos[b])) for b in bids), default=0.0)
        return {"max_resting_drift_m": max_drift, "max_penetration_m": max_pen}

    def step_robot_target(self, target_qpos: np.ndarray, gripper: float, dt: float) -> None:
        n = min(len(target_qpos), self.model.nu)
        self.data.ctrl[:n] = target_qpos[:n]
        if self.model.nu >= 8:
            # Menagerie Panda tendon actuator uses [0..255] where 255=open, 0=closed
            self.data.ctrl[7] = 255.0 * (1.0 - float(np.clip(gripper, 0.0, 1.0)))
        n_sub = max(1, int(round(dt / self.model.opt.timestep)))
        for _ in range(n_sub):
            mujoco.mj_step(self.model, self.data)

    def render_camera(self, calib: CameraCalib) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        renderer = self._get_renderer(calib.width, calib.height)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id < 0:
            cam_id = 0

        pos, quat = opencv_cam_to_mujoco_pose(calib.T_base_cam)
        self.model.cam_pos[cam_id] = pos
        self.model.cam_quat[cam_id] = quat

        # Exact pinhole intrinsics including principal point offset (verified pixel-accurate on DROID)
        px = 1e-5
        ox = -(calib.cx - calib.width / 2.0) * px
        oy = (calib.cy - calib.height / 2.0) * px
        self.model.cam_resolution[cam_id] = [calib.width, calib.height]
        self.model.cam_sensorsize[cam_id] = [calib.width * px, calib.height * px]
        self.model.cam_intrinsic[cam_id] = [calib.fx * px, calib.fy * px, ox, oy]
        mujoco.mj_forward(self.model, self.data)

        renderer.disable_segmentation_rendering()
        renderer.update_scene(self.data, camera=cam_id)
        rgb = renderer.render().copy()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(self.data, camera=cam_id)
        seg = renderer.render()[:, :, 0].copy()
        renderer.disable_segmentation_rendering()

        robot_mask = np.isin(seg, list(self.robot_geom_ids)) if self.robot_geom_ids else (seg >= 0)
        obj_mask = np.isin(seg, list(self.obj_geom_ids)) if self.obj_geom_ids else np.zeros_like(robot_mask)
        return rgb, robot_mask, obj_mask

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
