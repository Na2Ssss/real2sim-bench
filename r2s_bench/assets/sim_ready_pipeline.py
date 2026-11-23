"""Reusable Sim-Ready 3D Asset Post-Processing Pipeline.

Converts ANY raw `.glb/.obj` mesh (from Hunyuan3D-2, TRELLIS, GSO, etc.) into a
MuJoCo-ready physics + visual asset:
  1. Orientation & Metric Scale Normalization (Z-up, centered in XY, bottom flush at z=0).
  2. Automatic Open-Top Container Hollowing (if `is_container=True`, preserves the exact
     generated outer surface while carving a clean open-top hollow interior shell).
  3. Automatic CoACD Convex Decomposition (16-24 clean convex collision hulls for MuJoCo).
  4. Real-Camera Tabletop Homography Texture Baking (`extract_real_table_texture`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import coacd
import cv2
import numpy as np
import trimesh
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.measure import marching_cubes


@dataclass
class SimReadyAsset:
    name: str
    asset_dir: Path
    visual_mesh_path: Path
    collision_hull_paths: List[Path]
    size_xyz_m: Tuple[float, float, float]
    rgba: Tuple[float, float, float, float]
    is_container: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "visual_mesh_path": str(self.visual_mesh_path),
            "collision_hull_paths": [str(p) for p in self.collision_hull_paths],
            "size_xyz_m": [float(v) for v in self.size_xyz_m],
            "rgba": [float(v) for v in self.rgba],
            "is_container": bool(self.is_container),
        }


def _load_single_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle meshes found in {mesh_path}")
        return trimesh.util.concatenate(meshes)
    return loaded


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return 4x4 homogeneous rotation matrix mapping unit vector a to unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    T = np.eye(4, dtype=np.float64)
    if s < 1e-8:
        if c < 0.0:
            # 180 deg rotation around X
            T[1, 1] = -1.0
            T[2, 2] = -1.0
        return T
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    T[:3, :3] = R
    return T


def _normalize_mesh_z_up(
    mesh: trimesh.Trimesh,
    target_size_xyz: Tuple[float, float, float],
    y_up_to_z_up: bool = True,
    auto_upright_pca: bool = True,
    is_container: bool = True,
    taper_bot_scale: float = 1.0,
) -> trimesh.Trimesh:
    """Convert Y-up (GLB/Hunyuan3D convention) to Z-up, PCA-level camera tilt, sit at z=0, and scale."""
    m = mesh.copy()
    if y_up_to_z_up:
        # Rotate +90 deg around X: (x, y, z)_yup -> (x, -z, y)_zup
        R = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
        m.apply_transform(R)

    if auto_upright_pca:
        verts = np.asarray(m.vertices, dtype=np.float64)
        if len(verts) > 20000:
            idx = np.linspace(0, len(verts) - 1, 20000).astype(int)
            pts = verts[idx]
        else:
            pts = verts
        pts_c = pts - pts.mean(axis=0, keepdims=True)
        eigvals, eigvecs = np.linalg.eigh(np.cov(pts_c.T))
        tx, ty, tz = target_size_xyz
        if tz > max(tx, ty) * 1.05:
            # Tall object (cup, bottle): longest principal axis is vertical
            axis = eigvecs[:, -1]
        elif tz < min(tx, ty) * 0.90:
            # Flat object (bowl, plate): shortest principal axis is vertical
            axis = eigvecs[:, 0]
        else:
            axis = np.array([0.0, 0.0, 1.0])
        if axis[2] < 0.0:
            axis = -axis
        m.apply_transform(_rotation_from_a_to_b(axis, np.array([0.0, 0.0, 1.0])))

        # For open-top containers, ensure wider rim is at +Z (top)
        if is_container:
            v2 = np.asarray(m.vertices, dtype=np.float64)
            z_lo, z_hi = v2[:, 2].min(), v2[:, 2].max()
            span = max(z_hi - z_lo, 1e-6)
            bot = v2[v2[:, 2] < z_lo + 0.15 * span][:, :2]
            top = v2[v2[:, 2] > z_hi - 0.15 * span][:, :2]
            if len(bot) > 10 and len(top) > 10:
                r_bot = float(np.linalg.norm(bot - bot.mean(axis=0), axis=1).mean())
                r_top = float(np.linalg.norm(top - top.mean(axis=0), axis=1).mean())
                if r_bot > r_top * 1.05:
                    flip = np.diag([1.0, -1.0, -1.0, 1.0])
                    m.apply_transform(flip)

    bounds = m.bounds
    extents = bounds[1] - bounds[0]
    extents = np.maximum(extents, 1e-6)
    center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    z_min = float(bounds[0, 2])

    # Translate bottom-center to origin
    m.apply_translation([-center_xy[0], -center_xy[1], -z_min])

    # Apply target metric scale (sx, sy, sz)
    sx = target_size_xyz[0] / extents[0]
    sy = target_size_xyz[1] / extents[1]
    sz = target_size_xyz[2] / extents[2]
    S = np.diag([sx, sy, sz, 1.0])
    m.apply_transform(S)

    # Optional smooth linear radial taper along Z (corrects single/dual-view cylinder bias on tapered cups)
    if abs(taper_bot_scale - 1.0) > 1e-4:
        v = np.asarray(m.vertices, dtype=np.float64).copy()
        z_hi = max(float(v[:, 2].max()), 1e-6)
        frac = np.clip(v[:, 2] / z_hi, 0.0, 1.0)
        scale_xy = taper_bot_scale + (1.0 - taper_bot_scale) * frac
        v[:, 0] *= scale_xy
        v[:, 1] *= scale_xy
        m.vertices = v
    return m


def _carve_open_top_hollow_shell(
    mesh: trimesh.Trimesh,
    wall_thickness_m: float = 0.004,
    bottom_thickness_m: float = 0.005,
    grid_res: int = 72,
) -> trimesh.Trimesh:
    """Carve an open-top hollow cavity inside a container mesh while preserving its exact outer profile."""
    bounds = mesh.bounds
    ext = bounds[1] - bounds[0]
    pitch = float(np.max(ext)) / float(grid_res)
    vg = mesh.voxelized(pitch=pitch)
    occ = vg.matrix.copy()
    origin = vg.transform[:3, 3].copy()

    # Fill internal holes slice-by-slice in XY so the outer silhouette is solid first
    for iz in range(occ.shape[2]):
        occ[:, :, iz] = binary_fill_holes(occ[:, :, iz])

    # Trim sparse slanted top-cap artifact slices (< 40% area of the 85%-height slice)
    # so the open rim is a clean horizontal ring (critical for single-view 3D reconstructions)
    ref_z = max(1, int(0.85 * occ.shape[2]))
    ref_area = int(occ[:, :, ref_z].sum())
    if ref_area > 0:
        for iz in range(ref_z + 1, occ.shape[2]):
            if int(occ[:, :, iz].sum()) < int(0.65 * ref_area):
                occ[:, :, iz:] = False
                break

    # Erode in XY to compute inner cavity (strictly inside outer wall by `wall_thickness_m`)
    wall_vox = max(1, int(round(wall_thickness_m / pitch)))
    bot_vox = max(1, int(round(bottom_thickness_m / pitch)))
    cavity = np.zeros_like(occ, dtype=bool)
    # Use disk structuring element (accurate circular wall thickness instead of square corners)
    yy, xx = np.ogrid[-wall_vox : wall_vox + 1, -wall_vox : wall_vox + 1]
    struct_2d = (xx * xx + yy * yy) <= (wall_vox * wall_vox)
    for iz in range(bot_vox, occ.shape[2]):
        if occ[:, :, iz].any():
            cavity[:, :, iz] = binary_erosion(occ[:, :, iz], structure=struct_2d)

    shell = occ & (~cavity)
    # Pad by 1 empty voxel on all sides for clean closed Marching Cubes surface
    padded = np.pad(shell.astype(np.float32), pad_width=1, mode="constant", constant_values=0.0)
    verts, faces, _, _ = marching_cubes(padded, level=0.5)
    verts = (verts - 1.0) * pitch + origin
    # Marching cubes on occupancy=1 produces inward-facing winding; ensure positive outward volume
    out = trimesh.Trimesh(vertices=verts, faces=faces[:, ::-1], process=True)
    if out.volume < 0:
        out.invert()
    # Smooth slightly while preserving metric bounds
    trimesh.smoothing.filter_taubin(out, iterations=5)
    out.fix_normals()
    # Re-snap bottom to z=0, center XY, and preserve exact original Z height after top-cap trim
    b = out.bounds
    cxy = 0.5 * (b[0, :2] + b[1, :2])
    out.apply_translation([-float(cxy[0]), -float(cxy[1]), -float(b[0, 2])])
    cur_z = float(out.bounds[1, 2] - out.bounds[0, 2])
    if cur_z > 1e-6 and abs(cur_z - float(ext[2])) > 1e-4:
        out.apply_transform(np.diag([1.0, 1.0, float(ext[2]) / cur_z, 1.0]))
    return out


def build_sim_ready_asset(
    name: str,
    raw_mesh_path: Path,
    out_dir: Path,
    target_size_xyz: Tuple[float, float, float],
    rgba: Tuple[float, float, float, float],
    is_container: bool = True,
    y_up_to_z_up: bool = True,
    auto_upright_pca: bool = True,
    taper_bot_scale: float = 1.0,
    wall_thickness_m: float = 0.004,
    coacd_threshold: float = 0.04,
    max_convex_hulls: int = 22,
) -> SimReadyAsset:
    """Full Sim-Ready pipeline: Normalize -> Open-Top Hollow Shell -> CoACD Convex Decomposition."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_single_mesh(raw_mesh_path)
    norm_mesh = _normalize_mesh_z_up(
        raw,
        target_size_xyz=target_size_xyz,
        y_up_to_z_up=y_up_to_z_up,
        auto_upright_pca=auto_upright_pca,
        is_container=is_container,
        taper_bot_scale=taper_bot_scale,
    )

    if is_container:
        phys_mesh = _carve_open_top_hollow_shell(norm_mesh, wall_thickness_m=wall_thickness_m)
        vis_mesh = phys_mesh  # Hollow shell also looks open-top visually!
    else:
        phys_mesh = norm_mesh
        vis_mesh = norm_mesh

    vis_path = out_dir / f"{name}_visual.obj"
    vis_mesh.export(str(vis_path))

    # Run CoACD convex decomposition on the physics mesh
    coacd_in = coacd.Mesh(
        np.asarray(phys_mesh.vertices, dtype=np.float64),
        np.asarray(phys_mesh.faces, dtype=np.int32),
    )
    parts = coacd.run_coacd(
        coacd_in,
        threshold=coacd_threshold,
        max_convex_hull=max_convex_hulls,
        preprocess_mode="auto",
        resolution=2000,
    )

    hull_paths: List[Path] = []
    for idx, (vs, fs) in enumerate(parts):
        if len(vs) < 4 or len(fs) < 4:
            continue
        hp = out_dir / f"{name}_col_{idx:02d}.obj"
        trimesh.Trimesh(vertices=vs, faces=fs, process=False).export(str(hp))
        hull_paths.append(hp)

    asset = SimReadyAsset(
        name=name,
        asset_dir=out_dir,
        visual_mesh_path=vis_path,
        collision_hull_paths=hull_paths,
        size_xyz_m=target_size_xyz,
        rgba=rgba,
        is_container=is_container,
    )
    (out_dir / "asset_meta.json").write_text(json.dumps(asset.to_dict(), indent=2))
    return asset


def extract_real_table_texture(
    frame_bgr: np.ndarray,
    calib,
    table_center_xy: Tuple[float, float],
    table_half_xy: Tuple[float, float],
    table_z: float,
    out_png: Path,
    exclude_masks: Optional[List[np.ndarray]] = None,
    tex_size: int = 512,
) -> Path:
    """Warp the real wooden tabletop from Cam1 via Homography + inpaint out foreground objects."""
    h, w = frame_bgr.shape[:2]
    sc = calib.scaled(w, h)
    T_cam_base = np.linalg.inv(sc.T_base_cam)

    # Clean out cup/bowl/robot pixels on the table before warping via Navier-Stokes inpainting
    clean_bgr = frame_bgr.copy()
    if exclude_masks:
        union = np.zeros((h, w), dtype=np.uint8)
        for m in exclude_masks:
            m_rs = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            union |= (m_rs > 0).astype(np.uint8) * 255
        union = cv2.dilate(union, np.ones((15, 15), np.uint8))
        clean_bgr = cv2.inpaint(clean_bgr, union, 9, cv2.INPAINT_TELEA)

    cx, cy = table_center_xy
    hx, hy = table_half_xy
    # 4 corners of tabletop box top surface in robot base frame -> project to Cam1 pixels
    corners_3d = np.array([
        [cx - hx, cy - hy, table_z],
        [cx + hx, cy - hy, table_z],
        [cx + hx, cy + hy, table_z],
        [cx - hx, cy + hy, table_z],
    ], dtype=np.float64)
    src_pts = []
    for p in corners_3d:
        pc = (T_cam_base @ np.append(p, 1.0))[:3]
        u = sc.fx * (pc[0] / pc[2]) + sc.cx
        v = sc.fy * (pc[1] / pc[2]) + sc.cy
        src_pts.append([u, v])
    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.array([[0, tex_size - 1], [tex_size - 1, tex_size - 1], [tex_size - 1, 0], [0, 0]], dtype=np.float32)

    H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(clean_bgr, H_mat, (tex_size, tex_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), warped)
    return out_png
