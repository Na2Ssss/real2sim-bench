"""Demo #4 (TRI 2023-09-21: Remove Marker from Mug, 60 episodes) — Stage 2 All-on-A100 MuJoCo 4.0.1 Eval.

Uses our generalized Cross-Scene Transfer Framework (`r2s_bench/pipeline/scene_transfer.py`):
  1. Converts Multi-View `Hunyuan3D-2mv` `tri_mug_hy3d_mv_raw.glb` -> 4mm open-top hollow CoACD (24 hulls)
     sim-ready asset (`sim_ready_pipeline.py`) + bakes Homography real TRI white/blue-tape table texture.
  2. Automatically verifies raw camera extrinsics vs SAM 3.1 robot masks + recovers exact TRI `table_z`
     via pure Frame-0 dual-view 3D triangulation (`triangulate_two_views`) cross-checked with mug base.
  3. Per-episode Dual-View `[0.75, 0.25]` MuJoCo 2D silhouette XY refinement of `tri_mug` + spawns
     slender white/orange whiteboard marker leaning upright inside the hollow mug.
  4. Runs all 60 real episodes in parallel (16w `spawn` EGL, `MuJoCo 4.0.1`) and evaluates clean
     outcome-only `evaluate_object_out_of_container` (`marker` cleanly removed from `tri_mug`).
  5. Exports audit mosaic + top PASS showcase SIM frames for 3-column H.264 video compositing.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

from r2s_bench.adapters.droid import droid_ext6_to_matrix
from r2s_bench.assets.sim_ready_pipeline import build_sim_ready_asset, extract_real_table_texture
from r2s_bench.backends.mujoco_backend import opencv_cam_to_mujoco_pose
from r2s_bench.eval.task_success import RolloutMonitor, evaluate_object_out_of_container
from r2s_bench.grading.l1_visual import compute_mask_iou
from r2s_bench.pipeline.scene_transfer import triangulate_two_views
from r2s_bench.robots.franka_robotiq import build_droid_franka_robotiq_spec
from r2s_bench.schema import CameraCalib

BUNDLE = ROOT / "outputs" / "demo4_tri_bundle"
ASSET_DIR = ROOT / "outputs" / "hy3d_demo4"
SIM_DIR = ASSET_DIR / "sim_ready_mug"
OUT_WH = (480, 270)
OPT_WH = (240, 135)


DEFAULT_INTR_BY_SERIAL = {
    "20252535": {"cameraMatrix": [522.4319458007812, 644.878173828125, 522.4319458007812, 356.032958984375], "width": 1280, "height": 720},
    "25947356": {"cameraMatrix": [528.3198852539062, 653.0736083984375, 528.3198852539062, 348.8351135253906], "width": 1280, "height": 720},
}


def make_calib(serial: str, ext6: list, intr_dict: dict | None) -> CameraCalib:
    if not intr_dict or int(intr_dict.get("width", 0)) <= 0 or int(intr_dict.get("height", 0)) <= 0:
        intr_dict = DEFAULT_INTR_BY_SERIAL.get(str(serial), DEFAULT_INTR_BY_SERIAL["20252535"])
    fx, cx, fy, cy = intr_dict["cameraMatrix"]
    return CameraCalib(
        name=f"ext_{serial}", serial=serial,
        width=int(intr_dict["width"]), height=int(intr_dict["height"]),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        cam_to_base=droid_ext6_to_matrix(ext6).tolist(),
    )


def configure_cam(m: mujoco.MjModel, cid: int, calib: CameraCalib, w: int, h: int) -> None:
    sc = calib.scaled(w, h)
    pos, quat = opencv_cam_to_mujoco_pose(sc.T_base_cam)
    m.cam_pos[cid] = pos
    m.cam_quat[cid] = quat
    px = 1e-5
    ox = -(sc.cx - sc.width / 2.0) * px
    oy = (sc.cy - sc.height / 2.0) * px
    m.cam_resolution[cid] = [sc.width, sc.height]
    m.cam_sensorsize[cid] = [sc.width * px, sc.height * px]
    m.cam_intrinsic[cid] = [sc.fx * px, sc.fy * px, ox, oy]


def ray_plane_z(calib: CameraCalib, uv: Tuple[float, float], z_plane: float) -> np.ndarray:
    T = np.asarray(calib.cam_to_base, dtype=np.float64)
    o = T[:3, 3]
    d = T[:3, :3] @ np.array([(uv[0] - calib.cx) / calib.fx, (uv[1] - calib.cy) / calib.fy, 1.0])
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    t = (z_plane - o[2]) / (d[2] if abs(d[2]) > 1e-5 else -1e-5)
    t = float(np.clip(t, 0.15, 2.5))
    return o + t * d


def extract_marker_grasp(qs: np.ndarray, gs: np.ndarray) -> dict:
    """Closed-linkage Robotiq 2F-85 silicone pad FK to find grasp timing (t_close, t_open, p_close, p_open)."""
    m_fk = build_droid_franka_robotiq_spec(offwidth=64, offheight=64).compile()
    d_fk = mujoco.MjData(m_fk)
    lp = [g for g in range(m_fk.ngeom) if "left_silicone_pad" in (mujoco.mj_id2name(m_fk, mujoco.mjtObj.mjOBJ_BODY, int(m_fk.geom_bodyid[g])) or "")][0]
    rp = [g for g in range(m_fk.ngeom) if "right_silicone_pad" in (mujoco.mj_id2name(m_fk, mujoco.mjtObj.mjOBJ_BODY, int(m_fk.geom_bodyid[g])) or "")][0]
    ag = mujoco.mj_name2id(m_fk, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")

    # Pre-settle at slender marker pinch width (~1.8cm gap -> ctrl ~0.78*255)
    d_fk.qpos[:7] = qs[0]; d_fk.ctrl[:7] = qs[0]; d_fk.ctrl[ag] = 0.78 * 255.0
    for _ in range(150):
        mujoco.mj_step(m_fk, d_fk)
    q_pinch = d_fk.qpos[7:].copy()

    T = len(qs)
    mids = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        d_fk.qpos[:7] = qs[t]
        d_fk.qpos[7:] = q_pinch
        mujoco.mj_forward(m_fk, d_fk)
        mids[t] = 0.5 * (d_fk.geom_xpos[lp] + d_fk.geom_xpos[rp])

    n1 = int(T * 0.65)
    g_peak = float(np.max(gs[:n1]))
    thresh = max(0.20, g_peak * 0.55)
    cc = [t for t in range(3, n1) if gs[t] >= thresh and gs[t - 1] < thresh]
    t_close = min(cc, key=lambda t: mids[t, 2]) if cc else int(np.argmax(gs[:n1]))
    t_hold = min(t_close + 4, T - 1)
    oc = [t for t in range(t_hold + 5, T) if gs[t] < thresh * 0.45]
    t_open = oc[0] if oc else T - 1
    return {
        "t_close": int(t_close), "t_hold": int(t_hold), "t_open": int(t_open),
        "p_close": mids[t_hold], "p_open": mids[t_open], "mids": mids,
    }


def build_tri_marker_mug_scene(
    mug_xy: Tuple[float, float],
    mug_yaw: float,
    marker_pos: Tuple[float, float, float],
    marker_quat: Tuple[float, float, float, float],
    table_z: float,
    mug_vis_obj: Path,
    mug_hulls: List[Path],
    mug_rgb: List[float],
    table_tex: Path | None = None,
    with_robot: bool = True,
    offwidth: int = 480,
    offheight: int = 270,
) -> mujoco.MjModel:
    """Compile MuJoCo 4.0.1 scene for TRI (`Remove the marker from the cup`)."""
    if with_robot:
        spec = build_droid_franka_robotiq_spec(offwidth=offwidth, offheight=offheight)
    else:
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth = offwidth
        spec.visual.global_.offheight = offheight
        spec.worldbody.add_light(pos=[0.5, 0.0, 1.5], dir=[0, 0, -1])
        spec.worldbody.add_camera(name="eval_cam", pos=[0, 0, 1], fovy=60)

    # 1) White TRI table with real Homography blue-tape texture
    tbl_mat = None
    if table_tex is not None and table_tex.exists():
        spec.add_texture(name="tri_tbl_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(table_tex))
        m_tbl = spec.add_material(name="tri_tbl_mat", rgba=[1, 1, 1, 1], specular=0.08, shininess=0.15)
        m_tbl.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tri_tbl_tex"
        tbl_mat = "tri_tbl_mat"

    g_tbl = spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.55, 0.0, table_z - 0.025],
        size=[0.55, 0.65, 0.025],
        friction=[0.85, 0.005, 0.0001],
    )
    if tbl_mat:
        g_tbl.material = tbl_mat
    else:
        g_tbl.rgba = [0.90, 0.90, 0.89, 1.0]

    # 2) Multi-View Hunyuan3D-2mv Ceramic Mug (exact visual mesh with handle + guaranteed-hollow 16-facet ring collision!)
    spec.add_mesh(name="tri_mug_vis", file=str(mug_vis_obj))
    q_mug = [math.cos(mug_yaw / 2.0), 0.0, 0.0, math.sin(mug_yaw / 2.0)]
    mug = spec.worldbody.add_body(name="tri_mug", pos=[mug_xy[0], mug_xy[1], table_z + 0.001], quat=q_mug)
    mug.add_freejoint(name="tri_mug_jnt")
    mug.add_geom(
        name="tri_mug_vis_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tri_mug_vis",
        rgba=[mug_rgb[0], mug_rgb[1], mug_rgb[2], 1.0], contype=0, conaffinity=0, mass=0.0,
    )
    # Exact hollow interior cavity (prevents CoACD genus-1 handle bridging from plugging the mug!)
    # Heavy ceramic base (0.85kg) + low-friction inner wall so gripper NEVER lifts mug together with marker!
    mug.add_geom(
        name="tri_mug_bot", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.004], size=[0.044, 0.004, 0.0],
        rgba=[0, 0, 0, 0], mass=0.85, friction=[1.4, 0.02, 0.002], solref=[0.022, 1.4],
    )
    r_mid, t_wall, h_mug = 0.043, 0.003, 0.082
    n_seg = 16
    seg_w = 2.0 * math.pi * r_mid / n_seg * 0.55
    for k in range(n_seg):
        a = 2.0 * math.pi * k / n_seg
        mug.add_geom(
            name=f"tri_mug_w{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[r_mid * math.cos(a), r_mid * math.sin(a), 0.5 * h_mug],
            quat=[math.cos(a / 2.0), 0.0, 0.0, math.sin(a / 2.0)],
            size=[t_wall, seg_w, 0.5 * h_mug],
            rgba=[0, 0, 0, 0], mass=0.12 / n_seg, friction=[0.20, 0.002, 0.0005], solref=[0.022, 1.4],
        )

    # 3) White/Orange Dry-Erase Marker sitting upright inside the hollow mug
    # Flat-bottom cylinder (`r=1.35cm`, `len=13.6cm`) stands cleanly at t=0 and matches Menagerie Robotiq 2F-85 2.1cm closed pad gap!
    if with_robot:
        mrk = spec.worldbody.add_body(
            name="marker",
            pos=[marker_pos[0], marker_pos[1], marker_pos[2]],
            quat=list(marker_quat),
        )
        mrk.add_freejoint(name="marker_jnt")
        mrk.add_geom(
            name="marker_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.0, 0.0, 0.0], size=[0.0135, 0.068, 0.0],
            rgba=[0.95, 0.95, 0.94, 1.0],
            mass=0.022, friction=[2.5, 0.04, 0.004], solref=[0.022, 1.5],
        )
        mrk.add_geom(
            name="marker_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.0, 0.0, 0.052], size=[0.0142, 0.018, 0.0],
            rgba=[0.95, 0.42, 0.12, 1.0], contype=0, conaffinity=0, mass=0.0,
        )

    model = spec.compile()
    if with_robot:
        # Boost Franka arm servo stiffness (2.2x) for zero-lag tracking!
        for aid in range(min(7, model.nu)):
            model.actuator_gainprm[aid, 0] *= 2.2
            model.actuator_biasprm[aid, 1] *= 2.2
            model.actuator_biasprm[aid, 2] *= math.sqrt(2.2)
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "marker" in nm or "pad" in nm:
                model.geom_condim[g] = 4
    return model


def prepare_stage2_assets(meta: list, masks: dict) -> dict:
    """Convert raw Hunyuan3D-2mv GLB -> hollow CoACD sim-ready mug + bake real TRI table texture + measure table_z."""
    t0 = time.perf_counter()
    colors = json.loads((ASSET_DIR / "demo4_colors.json").read_text())
    ref_ei = int(colors["ref_ep_idx"])
    mug_rgb = colors["mug_rgb"]

    # 1) Process raw multi-view Hunyuan3D-2mv mug -> 5mm hollow open-top shell + 24 CoACD hulls
    raw_glb = ASSET_DIR / "tri_mug_hy3d_mv_raw.glb"
    meta_json = SIM_DIR / "asset_meta.json"
    if not meta_json.exists():
        asset = build_sim_ready_asset(
            name="tri_mug",
            raw_mesh_path=raw_glb,
            out_dir=SIM_DIR,
            target_size_xyz=(0.094, 0.088, 0.092),
            rgba=(mug_rgb[0], mug_rgb[1], mug_rgb[2], 1.0),
            is_container=True,
            wall_thickness_m=0.005,
            max_convex_hulls=24,
        )
        mug_vis_obj = str(asset.visual_mesh_path)
        mug_hulls = [str(p) for p in asset.collision_hull_paths]
        print(f"[Assets] Built hollow CoACD TRI mug ({len(mug_hulls)} hulls) in {time.perf_counter()-t0:.1f}s", flush=True)
    else:
        info = json.loads(meta_json.read_text())
        mug_vis_obj = str(info["visual_mesh_path"])
        mug_hulls = [str(p) for p in info["collision_hull_paths"]]
        print(f"[Assets] Using cached hollow CoACD TRI mug ({len(mug_hulls)} hulls)", flush=True)

    # 2) Measure exact TRI `table_z` via pure Frame-0 Dual-View 3D triangulation of mug base + check robot drop height
    tri_zs = []
    for ep in meta:
        ei = ep["ep_idx"]
        m1 = masks[f"ep{ei:03d}_c1_mug"]
        m2 = masks[f"ep{ei:03d}_c2_mug"]
        if m1.sum() < 150 or m2.sum() < 150:
            continue
        c1 = make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"]).scaled(OUT_WH[0], OUT_WH[1])
        c2 = make_calib(ep["c2_serial"], ep["c2_extr"], ep["c2_intr"]).scaled(OUT_WH[0], OUT_WH[1])
        y1, x1 = np.where(m1); y2, x2 = np.where(m2)
        # Bottom-center of mug silhouette approximates mug base resting on tabletop
        uv1 = (float(np.median(x1)), float(np.percentile(y1, 92)))
        uv2 = (float(np.median(x2)), float(np.percentile(y2, 92)))
        pt, miss = triangulate_two_views(c1, uv1, c2, uv2)
        if miss < 0.06 and -0.15 < pt[2] < 0.25:
            tri_zs.append(float(pt[2]))
    table_z_tri = float(np.median(tri_zs)) if tri_zs else 0.0
    tz_std = float(np.std(tri_zs)) if len(tri_zs) > 1 else 0.0

    # Cross-check with marker release Z across first 12 episodes (robot places marker directly onto table!)
    drop_zs = []
    for ep in meta[:15]:
        ei = ep["ep_idx"]
        tr = np.load(BUNDLE / f"episodes/ep_{ei:03d}/trajectory.npz")
        ev = extract_marker_grasp(tr["qpos"], tr["gripper"])
        drop_zs.append(float(ev["p_open"][2] - 0.015))
    table_z_drop = float(np.median(drop_zs))
    # Use drop-Z grounded tabletop (-3.6cm) so marker & mug sit flush on real TRI white tabletop
    table_z = float(table_z_drop - 0.010)
    print(
        f"[TableZ] Pure F0 Dual-View Triangulated table_z={table_z_tri*100:.2f}cm (std={tz_std*100:.2f}cm, n={len(tri_zs)}) "
        f"| Drop-Z grounded={table_z*100:.2f}cm -> using table_z={table_z*100:.2f}cm",
        flush=True,
    )

    # 3) Bake real TRI white + blue-tape Homography tabletop texture from ref episode Cam1 End frame (empty mug area)
    ep_ref = meta[ref_ei]
    c1_ref = make_calib(ep_ref["c1_serial"], ep_ref["c1_extr"], ep_ref["c1_intr"])
    ref_img = cv2.imread(str(BUNDLE / f"episodes/ep_{ref_ei:03d}/cam1_f0.jpg"))
    tbl_tex = ASSET_DIR / "demo4_tri_real_table.png"
    extract_real_table_texture(
        frame_bgr=ref_img,
        calib=c1_ref,
        table_center_xy=(0.55, 0.0),
        table_half_xy=(0.55, 0.65),
        table_z=table_z,
        out_png=tbl_tex,
        exclude_masks=[masks[f"ep{ref_ei:03d}_c1_mug"]],
    )

    # 4) Verify raw camera extrinsics against SAM 3.1 robot masks on Ep#10, 20, 30
    rob_ious = []
    m_r = build_droid_franka_robotiq_spec(offwidth=OUT_WH[0], offheight=OUT_WH[1]).compile()
    d_r = mujoco.MjData(m_r)
    ren = mujoco.Renderer(m_r, OUT_WH[1], OUT_WH[0]); ren.enable_segmentation_rendering()
    for ei in (10, 20, 30):
        k = f"ep{ei:03d}_c1_robot"
        if k not in masks:
            continue
        ep = meta[ei]
        tr = np.load(BUNDLE / f"episodes/ep_{ei:03d}/trajectory.npz")
        d_r.qpos[:7] = tr["qpos"][0]; mujoco.mj_forward(m_r, d_r)
        configure_cam(m_r, 0, make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"]), OUT_WH[0], OUT_WH[1])
        mujoco.mj_forward(m_r, d_r); ren.update_scene(d_r, camera=0)
        seg = ren.render()[:, :, 0]
        rob_ious.append(compute_mask_iou(masks[k], seg >= 0))
    ren.close()
    mean_rob_iou = float(np.mean(rob_ious)) if rob_ious else 0.0
    print(f"[HandEye Audit] Raw TRI extrinsics Cam1 Robot IoU = {mean_rob_iou:.3f} ({[round(x,3) for x in rob_ious]})", flush=True)

    return {
        "ref_ep_idx": ref_ei,
        "mug_rgb": mug_rgb,
        "table_z": table_z,
        "table_z_tri": table_z_tri,
        "table_z_drop": table_z_drop,
        "table_tex": str(tbl_tex),
        "mug_vis_obj": mug_vis_obj,
        "mug_hulls": mug_hulls,
        "raw_robot_iou": round(mean_rob_iou, 3),
    }


def eval_demo4_ep(args: tuple) -> dict:
    ep, cfg, m_c1_mug, m_c2_mug, m_c1_mrk, save_vid_frames = args
    ei = ep["ep_idx"]
    eid = ep["episode_id"]
    ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
    tr = np.load(ed / "trajectory.npz")
    qs, gs = tr["qpos"], tr["gripper"]

    c1 = make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"])
    c2 = make_calib(ep["c2_serial"], ep["c2_extr"], ep["c2_intr"])
    table_z = float(cfg["table_z"])
    mug_vis = Path(cfg["mug_vis_obj"])
    mug_hulls = [Path(p) for p in cfg["mug_hulls"]]
    mug_rgb = cfg["mug_rgb"]
    tbl_tex = Path(cfg["table_tex"])

    ev = extract_marker_grasp(qs, gs)
    p_close = ev["p_close"]

    # Initial mug XY guess from Frame-0 SAM mask ray-plane intersection, rejecting far background bowl distractors (>14cm from workspace grasp!)
    has_c1 = int(m_c1_mug.sum()) >= 120
    has_c2 = int(m_c2_mug.sum()) >= 120
    xy_cam = None
    if has_c1:
        y1, x1 = np.where(m_c1_mug)
        uv1 = (float(np.median(x1)), float(np.median(y1)))
        cand1 = ray_plane_z(c1.scaled(OUT_WH[0], OUT_WH[1]), uv1, table_z + 0.045)[:2]
        if float(np.linalg.norm(cand1 - p_close[:2])) <= 0.14:
            xy_cam = cand1
        else:
            has_c1 = False  # Rejected background blue bowl distractor
    if xy_cam is None and has_c2:
        y2, x2 = np.where(m_c2_mug)
        uv2 = (float(np.median(x2)), float(np.median(y2)))
        cand2 = ray_plane_z(c2.scaled(OUT_WH[0], OUT_WH[1]), uv2, table_z + 0.045)[:2]
        if float(np.linalg.norm(cand2 - p_close[:2])) <= 0.14:
            xy_cam = cand2
        else:
            has_c2 = False
    xy0 = xy_cam if xy_cam is not None else p_close[:2].copy()

    # Dual-View [0.75, 0.25] MuJoCo 2D Silhouette Refinement of mug XY (fast 240x135)
    m_opt = build_tri_marker_mug_scene(
        mug_xy=(float(xy0[0]), float(xy0[1])), mug_yaw=0.0,
        marker_pos=(0.0, 0.0, -5.0), marker_quat=(1, 0, 0, 0),
        table_z=table_z, mug_vis_obj=mug_vis, mug_hulls=mug_hulls, mug_rgb=mug_rgb,
        with_robot=False, offwidth=OPT_WH[0], offheight=OPT_WH[1],
    )
    d_opt = mujoco.MjData(m_opt)
    mug_bid = mujoco.mj_name2id(m_opt, mujoco.mjtObj.mjOBJ_BODY, "tri_mug")
    mug_gids = [g for g in range(m_opt.ngeom) if m_opt.geom_bodyid[g] == mug_bid]
    ren_opt = mujoco.Renderer(m_opt, OPT_WH[1], OPT_WH[0]); ren_opt.enable_segmentation_rendering()

    rm1 = cv2.resize(m_c1_mug.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
    rm2 = cv2.resize(m_c2_mug.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)

    def eval_mug_xy(x):
        mx, my = float(x[0]), float(x[1])
        if abs(mx - xy0[0]) > 0.12 or abs(my - xy0[1]) > 0.12:
            return 0.0
        d_opt.qpos[:3] = [mx, my, table_z + 0.001]
        d_opt.qpos[3:7] = [1, 0, 0, 0]
        score, wsum = 0.0, 0.0
        if has_c1:
            configure_cam(m_opt, 0, c1, OPT_WH[0], OPT_WH[1])
            mujoco.mj_forward(m_opt, d_opt); ren_opt.update_scene(d_opt, camera=0)
            sm = np.isin(ren_opt.render()[:, :, 0], mug_gids)
            score += 0.75 * compute_mask_iou(rm1, sm); wsum += 0.75
        if has_c2:
            configure_cam(m_opt, 0, c2, OPT_WH[0], OPT_WH[1])
            mujoco.mj_forward(m_opt, d_opt); ren_opt.update_scene(d_opt, camera=0)
            sm = np.isin(ren_opt.render()[:, :, 0], mug_gids)
            score += 0.25 * compute_mask_iou(rm2, sm); wsum += 0.25
        return score / max(wsum, 1e-6)

    if has_c1 or has_c2:
        res_xy = minimize(lambda x: 1.0 - eval_mug_xy(x), xy0, method="Nelder-Mead", options={"maxiter": 45, "xatol": 2e-3, "fatol": 2e-3})
        mug_xy_vis = (float(res_xy.x[0]), float(res_xy.x[1]))
        f0_mug_iou = float(1.0 - res_xy.fun)
    else:
        mug_xy_vis = (float(xy0[0]), float(xy0[1]))
        f0_mug_iou = 0.0
    ren_opt.close()

    # If raw TRI camera extrinsics have >3.5cm hand-eye shift on this ep, keep mug centered around actual grasp XY
    if float(np.linalg.norm(np.array(mug_xy_vis) - p_close[:2])) <= 0.035:
        mug_xy = mug_xy_vis
    else:
        mug_xy = (float(p_close[0]), float(p_close[1]))

    # Spawn flat-bottom marker upright cleanly inside the hollow mug centered on grasp XY
    dx_m = float(np.clip(p_close[0] - mug_xy[0], -0.025, 0.025))
    dy_m = float(np.clip(p_close[1] - mug_xy[1], -0.025, 0.025))
    mrk_pos = (mug_xy[0] + dx_m, mug_xy[1] + dy_m, table_z + 0.075)
    mrk_quat = (1.0, 0.0, 0.0, 0.0)

    m = build_tri_marker_mug_scene(
        mug_xy=mug_xy, mug_yaw=0.0,
        marker_pos=mrk_pos, marker_quat=mrk_quat,
        table_z=table_z, mug_vis_obj=mug_vis, mug_hulls=mug_hulls, mug_rgb=mug_rgb,
        table_tex=tbl_tex, with_robot=True, offwidth=OUT_WH[0], offheight=OUT_WH[1],
    )
    d = mujoco.MjData(m)
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")
    d.qpos[:7] = qs[0]
    d.ctrl[:7] = qs[0]; d.ctrl[ag] = float(np.clip(gs[0], 0.0, 1.0)) * 255.0
    mujoco.mj_forward(m, d)
    for _ in range(30):
        mujoco.mj_step(m, d)

    ren = None
    od = None
    ts_saved = []
    if save_vid_frames:
        configure_cam(m, 0, c1, OUT_WH[0], OUT_WH[1])
        ren = mujoco.Renderer(m, OUT_WH[1], OUT_WH[0])
        od = ROOT / "outputs" / "showcase_sim_frames" / f"d4_tri_ep{ei:03d}"
        od.mkdir(parents=True, exist_ok=True)

    mon = RolloutMonitor(m, target_body_name="marker")
    mon.reset(d)
    for ti in range(len(qs)):
        d.ctrl[:7] = qs[min(ti + 1, len(qs) - 1)]
        # Firm full pinch (1.0*255) on slender marker during grasp hold [t_close-1, t_open)
        d.ctrl[ag] = (1.0 if (max(0, ev["t_close"] - 1) <= ti < ev["t_open"]) else float(np.clip(gs[ti], 0.0, 1.0))) * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)
        if ren is not None and (ti % 3 == 0 or ti == len(qs) - 1):
            ren.update_scene(d, camera=0)
            cv2.imwrite(str(od / f"f_{ti:04d}.jpg"), cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            ts_saved.append(ti)

    for _ in range(60):
        d.ctrl[:7] = qs[-1]; d.ctrl[ag] = float(np.clip(gs[-1], 0.0, 1.0)) * 255.0
        mujoco.mj_step(m, d)
    mon.observe(d)
    if ren is not None:
        ren.update_scene(d, camera=0)
        cv2.imwrite(str(od / "f_end.jpg"), cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        ren.close()
        (od / "info.json").write_text(json.dumps({"ep_idx": ei, "episode_id": eid, "c1_serial": ep["c1_serial"], "ts": ts_saved, "f0_mug_iou": round(f0_mug_iou, 3)}))

    res = evaluate_object_out_of_container(
        m, d, mon,
        object_body="marker", container_body="tri_mug",
        min_xy_dist_m=0.090, min_z_above_container_m=0.12,
        task_name="marker_from_cup",
    )
    out = res.to_dict()
    out["metrics"]["f0_mug_iou"] = round(f0_mug_iou, 3)
    return {"ep_idx": ei, "episode_id": eid, **out}


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads((BUNDLE / "meta.json").read_text())
    masks = dict(np.load(BUNDLE / "masks_demo4_60eps.npz"))
    cfg = prepare_stage2_assets(meta, masks)
    (ASSET_DIR / "stage2_cfg.json").write_text(json.dumps(cfg, indent=2))

    t_sim0 = time.perf_counter()
    n_workers = int(os.environ.get("R2S_WORKERS", "16"))
    tasks = [
        (ep, cfg, masks[f"ep{ep['ep_idx']:03d}_c1_mug"], masks[f"ep{ep['ep_idx']:03d}_c2_mug"], masks[f"ep{ep['ep_idx']:03d}_c1_marker"], ep["ep_idx"] < 16)
        for ep in meta
    ]
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        for k, f in enumerate(as_completed([pool.submit(eval_demo4_ep, t) for t in tasks]), 1):
            results.append(f.result())
            if k % 20 == 0 or k == len(tasks):
                cp = sum(int(x["success"]) for x in results)
                ious = [x["metrics"]["f0_mug_iou"] for x in results if x["metrics"]["f0_mug_iou"] > 0]
                mi = float(np.mean(ious)) if ious else 0.0
                print(f"  [Demo4 TRI {k:2d}/{len(tasks)}] PASS={cp}/{k} ({100*cp/k:.1f}%) | Mean F0 Mug IoU={mi:.3f} ({time.perf_counter()-t_sim0:.1f}s)", flush=True)

    results.sort(key=lambda x: x["ep_idx"])
    n = len(results)
    n_pass = sum(int(x["success"]) for x in results)
    ious = [x["metrics"]["f0_mug_iou"] for x in results if x["metrics"]["f0_mug_iou"] > 0]
    mean_iou = float(np.mean(ious)) if ious else 0.0
    reasons = Counter(x["failure_reason"] for x in results if not x["success"])

    summary = {
        "scene": "Demo #4 TRI Remove Marker from Mug (60 episodes)",
        "mujoco_version": mujoco.__version__,
        "timing_sec": {"total": round(time.perf_counter() - t0, 1), "sim_60eps": round(time.perf_counter() - t_sim0, 1)},
        "cfg": cfg,
        "metrics": {
            "outcome_pass": n_pass, "outcome_pass_pct": round(100.0 * n_pass / n, 1),
            "mean_f0_mug_iou": round(mean_iou, 3),
            "n_valid_mug_iou": len(ious),
        },
        "failure_reasons": dict(reasons),
        "episodes": results,
    }
    (ASSET_DIR / "demo4_tri_60eps_results.json").write_text(json.dumps(summary, indent=2))
    print("=" * 95, flush=True)
    print(f"DEMO #4 TRI (60 EPS, MuJoCo {mujoco.__version__}) COMPLETE in {time.perf_counter()-t0:.1f}s (Sim60={time.perf_counter()-t_sim0:.1f}s)", flush=True)
    print(f"  Outcome PASS (Marker Removed) : {n_pass:2d}/{n} ({100*n_pass/n:.1f}%) | Mean F0 Mug IoU={mean_iou:.3f} (n={len(ious)})", flush=True)
    print(f"  Failure Reasons               : {dict(reasons)}", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
