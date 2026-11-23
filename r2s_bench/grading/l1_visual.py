"""L1 Visual & Camera Alignment metrics."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def compute_smooth_mask_score(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    sigma_px: float = 25.0,
) -> float:
    """Small-object-friendly alignment score: 0.5 * IoU + 0.5 * exp(-centroid_err / sigma_px)."""
    iou = compute_mask_iou(mask_a, mask_b)
    ya, xa = np.nonzero(mask_a.astype(bool))
    yb, xb = np.nonzero(mask_b.astype(bool))
    if len(xa) == 0 and len(xb) == 0:
        return 1.0
    if len(xa) == 0 or len(xb) == 0:
        return 0.5 * iou
    err_px = float(np.hypot(float(xa.mean()) - float(xb.mean()), float(ya.mean()) - float(yb.mean())))
    return float(0.5 * iou + 0.5 * np.exp(-err_px / sigma_px))


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    if mask is not None:
        m = mask.astype(bool)
        if m.sum() == 0:
            return 0.0
        mse = float(np.mean((a[m] - b[m]) ** 2))
    else:
        mse = float(np.mean((a - b) ** 2))
    if mse < 1e-8:
        return 60.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def render_diagnostic_overlay(
    real_rgb: np.ndarray,
    sim_rgb: np.ndarray,
    robot_mask: np.ndarray,
    obj_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create crisp cyan-robot + orange-object overlay on real image for visual inspection."""
    out = real_rgb.copy().astype(np.float32)
    rm = robot_mask.astype(bool)
    if rm.any():
        cyan = np.zeros_like(out)
        cyan[:, :, 1] = 220.0
        cyan[:, :, 2] = 255.0
        blend = 0.55 * sim_rgb.astype(np.float32) + 0.45 * cyan
        out[rm] = 0.45 * out[rm] + 0.55 * blend[rm]
    if obj_mask is not None and obj_mask.astype(bool).any():
        om = obj_mask.astype(bool)
        orange = np.zeros_like(out)
        orange[:, :, 0] = 255.0
        orange[:, :, 1] = 140.0
        blend = 0.50 * sim_rgb.astype(np.float32) + 0.50 * orange
        out[om] = 0.40 * out[om] + 0.60 * blend[om]
    u8 = np.clip(out, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(rm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(u8, contours, -1, (0, 255, 255), 2)
    return u8


def evaluate_l1_frame(
    real_rgb: np.ndarray,
    sim_rgb: np.ndarray,
    real_robot_mask: np.ndarray,
    sim_robot_mask: np.ndarray,
    fg_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    return {
        "l1_robot_mask_iou": compute_mask_iou(real_robot_mask, sim_robot_mask),
        "l1_fg_psnr_db": compute_psnr(real_rgb, sim_rgb, mask=fg_mask),
    }
