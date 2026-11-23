"""Card 3: Multi-View Visual Hull Voxel Carving + Open-Top Hollow Shell + CoACD Convex Decomposition.

Reconstructs physically graspable, concave, collision-ready 3D MuJoCo mesh assets directly
from calibrated multi-view 2D object silhouettes — ZERO 3DGS, ZERO generative 3D models,
ZERO hand-typed cylinder/frustum dimensions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import coacd
import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

from r2s_bench.schema import CameraCalib


def carve_visual_hull_mesh(
    views: List[Tuple[np.ndarray, CameraCalib, np.ndarray]],
    bounds_xyz: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    voxel_res: int = 80,
    min_view_frac: float = 0.85,
    hollow_shell: bool = True,
    wall_voxels: int = 2,
    axial_symmetry: bool = True,
    smooth_sigma: float = 0.8,
) -> trimesh.Trimesh:
    """Carve a 3D Visual Hull from multi-view binary masks in object-local frame.

    Args:
        views: List of `(mask_bool_HW, calib, obj_origin_base_xyz)` tuples.
               Multiple episodes with different tabletop placements naturally act as
               distinct virtual viewpoints around the same rigid object!
        bounds_xyz: `((x0, x1), (y0, y1), (z0, z1))` in object-local frame (meters),
                    where `z = 0` is the tabletop contact plane.
        voxel_res: Voxel grid resolution along each axis.
        min_view_frac: Fraction of views in which a voxel must project inside the mask.
        hollow_shell: If True, carves out the interior while keeping the top open so
                      vessels (cups/bowls) are physically hollow and rim-graspable.
        wall_voxels: Shell wall thickness in voxels (~2 voxels ≈ 3.5-4.0 mm).
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds_xyz
    xs = np.linspace(x0, x1, voxel_res, dtype=np.float64)
    ys = np.linspace(y0, y1, voxel_res, dtype=np.float64)
    zs = np.linspace(z0, z1, voxel_res, dtype=np.float64)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts_local = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (N, 3)

    votes = np.zeros(len(pts_local), dtype=np.int32)
    for mask_hw, calib, obj_origin in views:
        H, W = mask_hw.shape[:2]
        sc = calib.scaled(W, H)
        T_cam_base = np.linalg.inv(sc.T_base_cam)
        pts_base = pts_local + np.asarray(obj_origin, dtype=np.float64)[None, :]
        pts_cam = (pts_base @ T_cam_base[:3, :3].T) + T_cam_base[:3, 3][None, :]

        z_pos = pts_cam[:, 2] > 0.05
        u = np.round(sc.fx * (pts_cam[:, 0] / np.maximum(pts_cam[:, 2], 1e-6)) + sc.cx).astype(np.int32)
        v = np.round(sc.fy * (pts_cam[:, 1] / np.maximum(pts_cam[:, 2], 1e-6)) + sc.cy).astype(np.int32)
        in_bounds = z_pos & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(len(pts_local), dtype=bool)
        hit[in_bounds] = mask_hw[v[in_bounds], u[in_bounds]]
        votes += hit.astype(np.int32)

    min_votes = max(1, int(np.ceil(len(views) * min_view_frac)))
    occ = (votes >= min_votes).reshape((voxel_res, voxel_res, voxel_res))

    # Keep only largest connected component and fill any internal holes
    labeled, n_comp = ndimage.label(occ)
    if n_comp == 0:
        raise RuntimeError("Visual hull carving produced an empty voxel grid")
    sizes = ndimage.sum(occ, labeled, range(1, n_comp + 1))
    occ = labeled == (int(np.argmax(sizes)) + 1)
    occ = ndimage.binary_fill_holes(occ)

    if axial_symmetry:
        # Remove 2-view diamond artifacts by revolving each Z-slice's carved radius around its XY centroid
        occ_sym = np.zeros_like(occ)
        z_indices = np.where(occ.any(axis=(0, 1)))[0]
        if len(z_indices) > 0:
            ix_all, iy_all, _ = np.nonzero(occ)
            cx_m = float(xs[ix_all].mean())
            cy_m = float(ys[iy_all].mean())
            rr_grid = np.hypot(xs[:, None] - cx_m, ys[None, :] - cy_m)
            dx_m = (x1 - x0) / (voxel_res - 1)
            dy_m = (y1 - y0) / (voxel_res - 1)
            radii = []
            for zi in z_indices:
                area_m2 = float(occ[:, :, zi].sum()) * dx_m * dy_m
                radii.append((zi, float(np.sqrt(area_m2 / np.pi))))
            # Cut off the 2-view visual hull conical cap above the widest rim slice!
            z_rim = max(radii, key=lambda x: x[1])[0]
            for zi, r_equiv in radii:
                if zi <= z_rim and r_equiv >= dx_m:
                    occ_sym[:, :, zi] = rr_grid <= r_equiv
            occ = occ_sym

    if hollow_shell:
        # Erode horizontally (X, Y) and from bottom (+Z), while leaving the top (+Z end) open!
        struct_xy = np.zeros((3, 3, 1), dtype=bool)
        struct_xy[1, :, 0] = True
        struct_xy[:, 1, 0] = True
        inner = occ.copy()
        for _ in range(wall_voxels):
            inner = ndimage.binary_erosion(inner, structure=struct_xy, border_value=0)
        # Keep bottom solid (wall_voxels layers above lowest occupied z)
        z_occ = np.where(occ.any(axis=(0, 1)))[0]
        if len(z_occ) > wall_voxels + 2:
            z_bot = int(z_occ[0])
            inner[:, :, : z_bot + wall_voxels] = False
        occ = occ & (~inner)

    # Pad by 1 voxel so Marching Cubes produces a watertight closed surface
    padded = np.pad(occ.astype(np.float32), pad_width=1, mode="constant", constant_values=0.0)
    if smooth_sigma > 0:
        padded = ndimage.gaussian_filter(padded, sigma=smooth_sigma)

    dx = (x1 - x0) / (voxel_res - 1)
    dy = (y1 - y0) / (voxel_res - 1)
    dz = (z1 - z0) / (voxel_res - 1)
    verts, faces, _, _ = marching_cubes(padded, level=0.42, spacing=(dx, dy, dz))
    # Undo 1-voxel padding offset back to object-local coordinates
    verts[:, 0] += x0 - dx
    verts[:, 1] += y0 - dy
    verts[:, 2] += z0 - dz
    # Shift bottom cleanly to z = 0
    verts[:, 2] -= float(verts[:, 2].min())

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return mesh


def decompose_mesh_coacd(
    mesh: trimesh.Trimesh,
    out_dir: Path,
    prefix: str,
    threshold: float = 0.045,
    max_convex_hull: int = 20,
) -> Tuple[Path, List[Path]]:
    """Save visual mesh + CoACD convex collision parts as OBJ files for MuJoCo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_path = out_dir / f"{prefix}_visual.obj"
    mesh.export(vis_path)

    coacd_mesh = coacd.Mesh(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32))
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode="auto",
        resolution=800,
    )
    part_paths: List[Path] = []
    for idx, (vs, fs) in enumerate(parts):
        pm = trimesh.Trimesh(vertices=vs, faces=fs, process=True)
        if len(pm.vertices) < 4 or len(pm.faces) < 4:
            continue
        pp = out_dir / f"{prefix}_col_{idx:02d}.obj"
        pm.export(pp)
        part_paths.append(pp)
    return vis_path, part_paths
