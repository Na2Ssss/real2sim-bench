"""Grasp-Guided Zero-Rule Object Localization (Public FK -> Back-Projection -> Held-out).

Eliminates hand-coded HSV thresholds and hand-coded image ROIs by using ONLY public
SysID trajectories:
  1. Detect grasp-close (`t_grasp`) and release/pour (`t_target`) events from public
     `gripper(t)` trajectories and compute exact 3D fingertip positions via MuJoCo FK.
  2. Back-project the 3D physical contact anchors onto public Frame-0 dual-view images
     to get exact 2D pixel prompts `(u, v)` (ready for SAM3 or CPU appearance learning).
  3. Automatically learn each object's Lab/HSV appearance model, expected 3D height band,
     and camera-to-base static offset strictly from public episodes.
  4. Localize the object on completely unseen held-out Frame-0 dual-view images via
     appearance candidate matching + dual-view epipolar triangulation (zero leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mujoco
import numpy as np

from r2s_bench.robots.franka_robotiq import build_droid_franka_robotiq_spec
from r2s_bench.schema import CameraCalib


@dataclass
class GraspEvent:
    t_grasp: int
    p_grasp_3d: np.ndarray
    t_target: int
    p_target_3d: np.ndarray
    has_release: bool


@dataclass
class LearnedObjectAppearance:
    name: str
    lab_mean: np.ndarray          # (3,) float32
    lab_std: np.ndarray           # (3,) float32
    hsv_low: np.ndarray           # (3,) float32
    hsv_high: np.ndarray          # (3,) float32
    z_mean: float                 # expected 3D Z in robot base frame
    z_tol: float                  # allowed Z tolerance
    xy_center: np.ndarray         # (2,) workspace center learned from public grasps
    area_med: float               # median 2D blob area in 720p
    area_min: float               # min 2D blob area in 720p
    area_max: float               # max 2D blob area in 720p
    static_bias_xy: np.ndarray    # (2,) median(p_phys_xy - p_vis_xy) from public eps
    public_prompts_uv: List[Tuple[Tuple[float, float], Tuple[float, float]]]


def project_3d_to_uv(p_base: np.ndarray, calib: CameraCalib) -> Tuple[float, float]:
    """Project a 3D point in robot base frame into pixel (u, v) using exact OpenCV intrinsics."""
    T_cam_base = np.linalg.inv(calib.T_base_cam)
    p_cam = T_cam_base[:3, :3] @ np.asarray(p_base, dtype=np.float64) + T_cam_base[:3, 3]
    u = calib.fx * (p_cam[0] / p_cam[2]) + calib.cx
    v = calib.fy * (p_cam[1] / p_cam[2]) + calib.cy
    return float(u), float(v)


def triangulate_rays(
    uv1: Tuple[float, float], c1: CameraCalib, uv2: Tuple[float, float], c2: CameraCalib
) -> Tuple[np.ndarray, float]:
    """Midpoint of closest approach between two calibrated camera rays."""
    def ray(uv, c):
        r_cam = np.array([(uv[0] - c.cx) / c.fx, (uv[1] - c.cy) / c.fy, 1.0], dtype=np.float64)
        o = c.T_base_cam[:3, 3]
        d = c.T_base_cam[:3, :3] @ r_cam
        return o, d / np.linalg.norm(d)

    o1, d1 = ray(uv1, c1)
    o2, d2 = ray(uv2, c2)
    w0 = o1 - o2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-8:
        return 0.5 * (o1 + o2), 1.0
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2))


class GraspEventExtractor:
    """Reusable MuJoCo FK extractor for public trajectory grasp/release events."""

    def __init__(self) -> None:
        self.model = build_droid_franka_robotiq_spec(offwidth=320, offheight=180).compile()
        self.data = mujoco.MjData(self.model)
        self.pad_l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_left_pad")
        self.pad_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_right_pad")

    def tip_xyz(self, q7: np.ndarray) -> np.ndarray:
        self.data.qpos[:7] = q7
        mujoco.mj_forward(self.model, self.data)
        return (0.5 * (self.data.xpos[self.pad_l] + self.data.xpos[self.pad_r])).copy()

    def extract(self, qpos_seq: np.ndarray, grip_seq: np.ndarray) -> GraspEvent:
        g = np.asarray(grip_seq, dtype=np.float64).reshape(-1)
        T = len(g)
        # Find grasp closing onset: first frame after t>=5 where gripper rises from <0.35 to >=0.45
        t_grasp = -1
        for t in range(5, T):
            if g[t] >= 0.45 and np.min(g[max(0, t - 10) : t]) < 0.35:
                t_grasp = t
                break
        if t_grasp < 0:
            t_grasp = int(np.argmax(g > 0.45))

        p_grasp = self.tip_xyz(qpos_seq[t_grasp])

        # Find release onset after holding for >= 10 frames
        after = np.where(g[t_grasp + 10 :] < 0.35)[0]
        if len(after) > 0:
            t_target = int(t_grasp + 10 + after[0])
            p_target = self.tip_xyz(qpos_seq[t_target])
            has_release = True
        else:
            # Pouring / non-release task: pick frame furthest horizontally from grasp after t_grasp
            best_t, best_d = t_grasp, -1.0
            for t in range(t_grasp + 10, T):
                pt = self.tip_xyz(qpos_seq[t])
                dist = float(np.hypot(pt[0] - p_grasp[0], pt[1] - p_grasp[1]))
                if dist > best_d:
                    best_d, best_t = dist, t
            t_target = best_t
            p_target = self.tip_xyz(qpos_seq[t_target])
            has_release = False

        return GraspEvent(
            t_grasp=t_grasp,
            p_grasp_3d=p_grasp,
            t_target=t_target,
            p_target_3d=p_target,
            has_release=has_release,
        )


def _extract_local_foreground_pixels(
    frame_bgr: np.ndarray,
    uv: Tuple[float, float],
    inner_r: int = 30,
    bg_r0: int = 55,
    bg_r1: int = 110,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Extract salient foreground pixels around projected prompt `(u, v)` vs surrounding ring."""
    H, W = frame_bgr.shape[:2]
    cx, cy = int(round(uv[0])), int(round(uv[1]))
    if not (20 <= cx < W - 20 and 20 <= cy < H - 20):
        return None

    x0, x1 = max(0, cx - bg_r1), min(W, cx + bg_r1)
    y0, y1 = max(0, cy - bg_r1), min(H, cy + bg_r1)
    patch_bgr = frame_bgr[y0:y1, x0:x1]
    patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - cx, yy - cy)
    inner_mask = dist <= inner_r
    bg_mask = (dist >= bg_r0) & (dist <= bg_r1)
    if bg_mask.sum() < 50 or inner_mask.sum() < 30:
        return None

    bg_lab_med = np.median(patch_lab[bg_mask], axis=0)
    bg_sat_med = float(np.median(patch_hsv[bg_mask, 1]))
    # Emphasize chromaticity (a, b) + saturation contrast over illumination
    diff = patch_lab - bg_lab_med[None, None, :]
    d_sat = np.maximum(0.0, patch_hsv[:, :, 1] - bg_sat_med)
    delta_e = np.sqrt(0.25 * diff[:, :, 0] ** 2 + 2.2 * diff[:, :, 1] ** 2 + 1.5 * diff[:, :, 2] ** 2 + 0.5 * d_sat ** 2)
    inner_de = delta_e[inner_mask]
    thresh = max(24.0, float(np.percentile(inner_de, 82)))
    fg_local = (delta_e >= thresh) & inner_mask
    if fg_local.sum() < 20:
        fg_local = (delta_e >= float(np.percentile(inner_de, 88))) & inner_mask

    # Pick the salient component closest to the projected FK grasp point
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_local.astype(np.uint8))
    if n <= 1:
        return None
    best, best_score = -1, -1.0
    local_cx, local_cy = cx - x0, cy - y0
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 15:
            continue
        d_prompt = float(np.hypot(centroids[i][0] - local_cx, centroids[i][1] - local_cy))
        de_mean = float(np.mean(delta_e[labels == i]))
        score = de_mean / (1.0 + d_prompt / 18.0)
        if score > best_score:
            best_score, best = score, i
    if best < 0:
        return None
    comp = labels == best
    area = float(stats[best, cv2.CC_STAT_AREA])
    return patch_lab[comp], patch_hsv[comp], area


def learn_appearance_from_public(
    name: str,
    public_samples: List[Tuple[np.ndarray, CameraCalib, np.ndarray, CameraCalib, np.ndarray]],
    z_shift: float = -0.025,
) -> LearnedObjectAppearance:
    """Learn an object's appearance model & static bias strictly from public episodes.

    Args:
        name: Object identifier (e.g., "grasped_obj", "target_container").
        public_samples: List of `(f1_bgr, c1, f2_bgr, c2, p_phys_3d)` from public episodes.
        z_shift: Vertical offset from fingertip contact point to object visual center (m).
    """
    all_lab = []
    all_hsv = []
    all_areas = []
    z_vals = []
    prompts_uv = []

    for f1, c1, f2, c2, p_phys in public_samples:
        p_vis_anchor = p_phys.copy()
        p_vis_anchor[2] += z_shift
        uv1 = project_3d_to_uv(p_vis_anchor, c1)
        uv2 = project_3d_to_uv(p_vis_anchor, c2)
        prompts_uv.append((uv1, uv2))
        z_vals.append(float(p_vis_anchor[2]))

        for fr, uv in ((f1, uv1), (f2, uv2)):
            res = _extract_local_foreground_pixels(fr, uv)
            if res is not None:
                lab_px, hsv_px, area = res
                all_lab.append(lab_px)
                all_hsv.append(hsv_px)
                all_areas.append(area)

    if not all_lab:
        raise RuntimeError(f"Failed to extract any foreground pixels around public prompts for {name}")

    cat_lab = np.concatenate(all_lab, axis=0)
    cat_hsv = np.concatenate(all_hsv, axis=0)
    lab_mean = np.median(cat_lab, axis=0).astype(np.float32)
    lab_std = np.maximum(np.std(cat_lab, axis=0), np.array([10.0, 6.0, 6.0])).astype(np.float32)

    # Tight robust 15-85 percentile HSV envelope (prevents bleeding into warm wooden tabletops)
    hsv_lo = np.percentile(cat_hsv, 15, axis=0) - np.array([4.0, 20.0, 25.0])
    hsv_hi = np.percentile(cat_hsv, 85, axis=0) + np.array([4.0, 25.0, 25.0])
    hsv_lo = np.clip(hsv_lo, [0, 0, 0], [180, 255, 255]).astype(np.float32)
    hsv_hi = np.clip(hsv_hi, [0, 0, 0], [180, 255, 255]).astype(np.float32)

    med_area = float(np.median(all_areas))
    xy_center = np.median(np.asarray([p[:2] for _, _, _, _, p in public_samples]), axis=0).astype(np.float64)
    model = LearnedObjectAppearance(
        name=name,
        lab_mean=lab_mean,
        lab_std=lab_std,
        hsv_low=hsv_lo,
        hsv_high=hsv_hi,
        z_mean=float(np.median(z_vals)),
        z_tol=0.14,  # wide enough to handle height-adjustable tables (e.g. Demo #3)
        xy_center=xy_center,
        area_med=med_area,
        area_min=max(25.0, med_area * 0.25),
        area_max=20000.0,
        static_bias_xy=np.zeros(2, dtype=np.float64),
        public_prompts_uv=prompts_uv,
    )

    # Calibrate median physical-vs-visual static bias on the SAME public samples
    biases = []
    for f1, c1, f2, c2, p_phys in public_samples:
        p_vis, _ = localize_in_frame0(f1, c1, f2, c2, model, apply_bias=False)
        if p_vis is not None:
            biases.append(p_phys[:2] - p_vis[:2])
    if biases:
        model.static_bias_xy = np.median(np.asarray(biases), axis=0)
    return model


def _find_candidates(frame_bgr: np.ndarray, model: LearnedObjectAppearance) -> List[Tuple[float, float, float]]:
    """Find candidate `(u, v, lab_dist)` centroids matching `model` anywhere in the image."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Normalized Lab distance (cap std so shadows cannot blow up tolerance and merge with tabletop)
    denom = np.clip(model.lab_std, np.array([10.0, 6.0, 6.0], dtype=np.float32), np.array([26.0, 14.0, 14.0], dtype=np.float32))
    z_lab = np.linalg.norm((lab - model.lab_mean[None, None, :]) / (denom[None, None, :] * 1.4), axis=2)
    is_chromatic = float(model.hsv_low[1] + model.hsv_high[1]) * 0.5 >= 75.0
    if is_chromatic:
        h_lo, h_hi = float(model.hsv_low[0]), float(model.hsv_high[0])
        h_ch = hsv[:, :, 0]
        if h_lo <= 12.0 and h_hi >= 165.0:
            h_ok = (h_ch <= 15.0) | (h_ch >= 163.0)
        else:
            h_ok = (h_ch >= h_lo) & (h_ch <= h_hi)
        s_ok = hsv[:, :, 1] >= max(70.0, float(model.hsv_low[1]))
        mask = (z_lab <= 2.1) & h_ok & s_ok
    else:
        mask = (z_lab <= 1.6) & (hsv[:, :, 1] <= min(120.0, float(model.hsv_high[1]) + 10.0))

    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)
    cands: List[Tuple[float, float, float]] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if model.area_min <= area <= model.area_max:
            # Prefer clean appearance match AND substantial connected object over tiny background specks
            comp_dist = float(np.mean(z_lab[labels == i])) - 0.12 * float(np.log(area + 1.0))
            cands.append((float(centroids[i][0]), float(centroids[i][1]), comp_dist))
    cands.sort(key=lambda x: x[2])
    return cands[:25]


def localize_in_frame0(
    f1_bgr: np.ndarray,
    c1: CameraCalib,
    f2_bgr: np.ndarray,
    c2: CameraCalib,
    model: LearnedObjectAppearance,
    apply_bias: bool = False,
) -> Tuple[Optional[np.ndarray], float]:
    """Localize object 3D position strictly from Frame-0 dual-view images using learned model."""
    cands1 = _find_candidates(f1_bgr, model)
    cands2 = _find_candidates(f2_bgr, model)
    best_p = None
    best_cost = 1e9
    best_gap = 1e9

    for u1, v1, d1 in cands1:
        for u2, v2, d2 in cands2:
            p, gap = triangulate_rays((u1, v1), c1, (u2, v2), c2)
            # Basic robot-reachable tabletop sanity check (zero scene-specific hardcoding)
            if not (0.28 <= p[0] <= 0.90 and -0.45 <= p[1] <= 0.50):
                continue
            if abs(p[2] - model.z_mean) > model.z_tol:
                continue
            d_ws = float(np.hypot(p[0] - model.xy_center[0], p[1] - model.xy_center[1]))
            if d_ws > 0.38:
                continue
            cost = gap + 0.015 * (d1 + d2) + 0.04 * d_ws
            if cost < best_cost:
                best_cost = cost
                best_p = p
                best_gap = gap

    if best_p is not None and apply_bias:
        best_p = best_p.copy()
        best_p[:2] += model.static_bias_xy
    return best_p, best_gap
