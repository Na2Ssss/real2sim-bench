"""L2 Physics & Object Dynamics Tracking metrics (2D mask-based, GT-6D-free)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from r2s_bench.grading.l1_visual import compute_mask_iou, compute_smooth_mask_score


def mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def detect_motion_onset(centroids: List[Optional[np.ndarray]], pixel_thresh: float = 3.0) -> Optional[int]:
    c0 = next((c for c in centroids if c is not None), None)
    if c0 is None:
        return None
    for idx, c in enumerate(centroids):
        if c is not None and float(np.linalg.norm(c - c0)) >= pixel_thresh:
            return idx
    return None


def evaluate_l2_sequence(
    real_obj_masks: List[np.ndarray],
    sim_obj_masks: List[np.ndarray],
    fps: float = 15.0,
) -> Dict[str, float]:
    assert len(real_obj_masks) == len(sim_obj_masks) and len(real_obj_masks) > 0
    ious: List[float] = []
    smooth_scores: List[float] = []
    centroid_errs: List[float] = []
    real_c: List[Optional[np.ndarray]] = []
    sim_c: List[Optional[np.ndarray]] = []

    for rm, sm in zip(real_obj_masks, sim_obj_masks):
        ious.append(compute_mask_iou(rm, sm))
        smooth_scores.append(compute_smooth_mask_score(rm, sm, sigma_px=25.0))
        rc, sc = mask_centroid(rm), mask_centroid(sm)
        real_c.append(rc)
        sim_c.append(sc)
        if rc is not None and sc is not None:
            centroid_errs.append(float(np.linalg.norm(rc - sc)))

    t_r, t_s = detect_motion_onset(real_c), detect_motion_onset(sim_c)
    if t_r is not None and t_s is not None:
        onset_err = abs(t_r - t_s) / fps
    elif t_r is None and t_s is None:
        onset_err = 0.0
    else:
        onset_err = len(real_obj_masks) / fps

    return {
        "l2_obj_track_mean_iou": float(np.mean(ious)),
        "l2_obj_track_final_iou": float(ious[-1]),
        "l2_obj_smooth_mean_score": float(np.mean(smooth_scores)),
        "l2_obj_smooth_final_score": float(smooth_scores[-1]),
        "l2_centroid_mean_err_px": float(np.mean(centroid_errs)) if centroid_errs else float("inf"),
        "l2_contact_onset_err_sec": float(onset_err),
    }
