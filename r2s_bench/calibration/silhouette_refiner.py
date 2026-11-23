"""Two-view silhouette pose & dimension refiner (EasyHeC++-style optimization).

Provides two closed-book calibration primitives:
1. `fit_vessel_dims_on_public_ep`: Run STRICTLY on a public SysID episode to fit
   programmatic vessel geometry `(r_bot, r_top, height)` against calibrated multi-view
   2D silhouettes, anchored around the coarse 3D center.
2. `refine_object_xy_two_view`: With frozen object dimensions, refine tabletop `(x, y)`
   within a tight shift bound (`<= 10-12 mm`) using camera-quality-weighted multi-view
   Mask IoU so lower-accuracy cameras cannot pull the object away from physical contact.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import mujoco
import numpy as np
from scipy.optimize import minimize

from r2s_bench.backends.mujoco_backend import opencv_cam_to_mujoco_pose
from r2s_bench.grading.l1_visual import compute_mask_iou
from r2s_bench.robots.franka_robotiq import _add_hollow_vessel
from r2s_bench.schema import CameraCalib


def _set_cam(model: mujoco.MjModel, cam_id: int, calib: CameraCalib) -> None:
    pos, quat = opencv_cam_to_mujoco_pose(calib.T_base_cam)
    model.cam_pos[cam_id] = pos
    model.cam_quat[cam_id] = quat
    px = 1e-5
    ox = -(calib.cx - calib.width / 2.0) * px
    oy = (calib.cy - calib.height / 2.0) * px
    model.cam_resolution[cam_id] = [calib.width, calib.height]
    model.cam_sensorsize[cam_id] = [calib.width * px, calib.height * px]
    model.cam_intrinsic[cam_id] = [calib.fx * px, calib.fy * px, ox, oy]


def _normalize_weights(n: int, weights: Optional[List[float]]) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return w / float(np.sum(w))


def _build_single_vessel_model(
    xy: Tuple[float, float],
    table_z: float,
    dims: Tuple[float, float, float],
    wall_thickness: float,
    offwidth: int,
    offheight: int,
) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = offwidth
    spec.visual.global_.offheight = offheight
    spec.worldbody.add_camera(name="eval_cam", pos=[1.0, 0.0, 1.0], fovy=55.0)
    body = spec.worldbody.add_body(name="vessel", pos=[xy[0], xy[1], table_z + 0.001])
    _add_hollow_vessel(
        body,
        name_prefix="vessel",
        radius_bottom=dims[0],
        radius_top=dims[1],
        height=dims[2],
        wall_thickness=wall_thickness,
        rgba=(0.8, 0.4, 0.2, 1.0),
        total_mass=0.1,
        n_sides=16,
    )
    return spec.compile()


def fit_vessel_dims_on_public_ep(
    init_xy: Tuple[float, float],
    table_z: float,
    init_dims: Tuple[float, float, float],
    dims_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    calibs: List[CameraCalib],
    real_masks: List[np.ndarray],
    camera_weights: Optional[List[float]] = None,
    wall_thickness: float = 0.004,
    max_xy_shift_m: float = 0.015,
    max_iter: int = 60,
) -> Tuple[Tuple[float, float, float], Tuple[float, float], float, float]:
    """Fit vessel `(r_bot, r_top, height)` (and small anchor shift) on public Ep#0.

    Returns:
        (fitted_dims, fitted_xy, init_weighted_iou, fitted_weighted_iou)
    """
    h, w = real_masks[0].shape[:2]
    scaled_calibs = [c.scaled(w, h) for c in calibs]
    weights = _normalize_weights(len(calibs), camera_weights)
    (rb_lo, rb_hi), (rt_lo, rt_hi), (h_lo, h_hi) = dims_bounds

    def eval_params(params: np.ndarray) -> float:
        dx, dy, rb, rt, vh = [float(v) for v in params]
        if np.hypot(dx, dy) > max_xy_shift_m:
            return -1.0
        if not (rb_lo <= rb <= rb_hi and rt_lo <= rt <= rt_hi and h_lo <= vh <= h_hi):
            return -1.0
        if rt < rb - 0.004:
            return -1.0

        m = _build_single_vessel_model(
            (init_xy[0] + dx, init_xy[1] + dy),
            table_z,
            (rb, rt, vh),
            wall_thickness=wall_thickness,
            offwidth=w,
            offheight=h,
        )
        d = mujoco.MjData(m)
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "eval_cam")
        renderer = mujoco.Renderer(m, height=h, width=w)
        renderer.enable_segmentation_rendering()
        ious = []
        for c, rm in zip(scaled_calibs, real_masks):
            _set_cam(m, cam_id, c)
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam_id)
            seg = renderer.render()[:, :, 0]
            sim_m = seg >= 0
            ious.append(compute_mask_iou(rm, sim_m))
        renderer.close()
        return float(np.dot(weights, ious))

    p0 = np.array([0.0, 0.0, init_dims[0], init_dims[1], init_dims[2]], dtype=np.float64)
    init_iou = eval_params(p0)

    def objective(params: np.ndarray) -> float:
        val = eval_params(params)
        if val < 0.0:
            return 2.0
        return 1.0 - val

    # Construct explicit simplex with physical scale steps (~6mm)
    steps = np.array([0.006, 0.006, 0.008, 0.010, 0.010], dtype=np.float64)
    simplex = np.tile(p0, (len(p0) + 1, 1))
    for i in range(len(p0)):
        simplex[i + 1, i] += steps[i]

    res = minimize(
        objective,
        p0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3},
    )
    best = res.x
    best_iou = eval_params(best)
    fitted_dims = (float(best[2]), float(best[3]), float(best[4]))
    fitted_xy = (float(init_xy[0] + best[0]), float(init_xy[1] + best[1]))
    return fitted_dims, fitted_xy, init_iou, best_iou


def refine_object_xy_two_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    init_xy: Tuple[float, float],
    table_z: float,
    calibs: List[CameraCalib],
    real_masks: List[np.ndarray],
    camera_weights: Optional[List[float]] = None,
    max_shift_m: float = 0.012,
    max_iter: int = 45,
) -> Tuple[Tuple[float, float], float, float]:
    """Refine a tabletop freejoint body's `(x, y)` to maximize weighted multi-view Mask IoU.

    Returns:
        (refined_xy, init_weighted_iou, refined_weighted_iou)
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt_adr = int(model.body_jntadr[bid])
    qpos_adr = int(model.jnt_qposadr[jnt_adr])
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eval_cam")

    h, w = real_masks[0].shape[:2]
    scaled_calibs = [c.scaled(w, h) for c in calibs]
    weights = _normalize_weights(len(calibs), camera_weights)
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.enable_segmentation_rendering()

    def eval_weighted_iou(xy: np.ndarray) -> float:
        data.qpos[qpos_adr] = float(xy[0])
        data.qpos[qpos_adr + 1] = float(xy[1])
        data.qpos[qpos_adr + 2] = float(table_z + 0.001)
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        ious = []
        for c, rm in zip(scaled_calibs, real_masks):
            _set_cam(model, cam_id, c)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam_id)
            seg = renderer.render()[:, :, 0]
            sim_m = np.isin(seg, gids)
            ious.append(compute_mask_iou(rm, sim_m))
        return float(np.dot(weights, ious))

    x0 = np.array(init_xy, dtype=np.float64)
    init_iou = eval_weighted_iou(x0)

    def objective(delta: np.ndarray) -> float:
        norm = float(np.linalg.norm(delta))
        if norm > max_shift_m:
            return 1.0 + norm
        return 1.0 - eval_weighted_iou(x0 + delta)

    # Compact initial simplex (~6mm) inside the tight trust region
    step = min(0.006, max_shift_m * 0.5)
    init_simplex = np.array([[0.0, 0.0], [step, 0.0], [0.0, step]], dtype=np.float64)
    res = minimize(
        objective,
        np.zeros(2),
        method="Nelder-Mead",
        options={"initial_simplex": init_simplex, "maxiter": max_iter, "xatol": 5e-4, "fatol": 5e-4},
    )
    best_xy = x0 + res.x
    best_iou = eval_weighted_iou(best_xy)
    renderer.close()
    return (float(best_xy[0]), float(best_xy[1])), init_iou, best_iou
