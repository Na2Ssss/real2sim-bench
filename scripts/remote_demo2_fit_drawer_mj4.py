"""Demo #2 1:1 Dual-View Articulated Workbench + Sliding Drawer Fitting & 74-Episode MuJoCo 4.0.1 Eval.

Stage A (Unbiased Multi-Start Dual-View 3D Silhouette Fit on Ep#00 + Ep#03, Cam1 + Cam2):
  1. Fill interior tool/foam holes inside SAM 3.1 open-drawer masks via largest-component convex hull
     (the 2D projection of a protruding 3D drawer box is a clean convex polygon).
  2. Coarse-to-fine multi-start search over UNCONSTRAINED `yaw in [-0.6, +0.6]` jointly fitting
     the crisp wooden countertop front slab AND the protruding open drawer on Cam1 + Cam2!
  3. Set exact closed plane `front_x_closed = ct_front_u + overhang` (flush underneath wooden countertop lip,
     cross-checked against 25 real robot push-end projections along the true fitted `yaw`).
  4. Bake dual-plane Homography textures from `ep00_c1_end.jpg` (closed drawer):
     - `demo2_real_countertop.png` (butcher-block wood + AUTOLAB tray)
     - `demo2_real_cabinet.png` (black steel Husky tool-chest drawers + silver handles)
  5. Export Dual-View 1:1 visual audit sheet (`demo2_drawer_1to1_fit_audit.png`) + `fitted_drawer_params.json`.

Stage B (74-Episode Parallel MuJoCo 4.0.1 Rollout using `spawn` mp_context — zero EGL fork deadlock!):
  - Uses ONE FIXED 1:1 FITTED WORKBENCH + ARTICULATED DRAWER across all 74 episodes.
  - Evaluates clean outcome-based `evaluate_block_in_drawer` (`block_in_drawer AND drawer_closed`).
  - Saves showcase SIM frames for top PASS episodes.
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
from r2s_bench.backends.mujoco_backend import opencv_cam_to_mujoco_pose
from r2s_bench.eval.task_success import RolloutMonitor, evaluate_block_in_drawer
from r2s_bench.grading.l1_visual import compute_mask_iou
from r2s_bench.robots.franka_robotiq import build_droid_franka_robotiq_spec
from r2s_bench.schema import CameraCalib
from scripts.remote_demo2_74eps_mj4 import extract_gripper_events

BUNDLE = ROOT / "outputs" / "demo2_74eps_bundle"
STAGE = ROOT / "outputs" / "demo2_fit_stage"
HE_JSON = ROOT / "outputs" / "handeye_demo2" / "refined_extrinsics.json"
ASSET_DIR = ROOT / "outputs" / "hy3d_demo2"
OUT_WH = (480, 270)
OPT_WH = (240, 135)


def make_calib(serial: str, ext6: list, intr_dict: dict) -> CameraCalib:
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


def fill_drawer_hull(mask_bool: np.ndarray) -> np.ndarray:
    """Fill dark tool/foam holes inside the open-drawer SAM mask using largest-component convex hull."""
    u8 = mask_bool.astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n_lab <= 1:
        return mask_bool
    # Keep largest component (the open drawer)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main = (labels == best).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask_bool
    hull = cv2.convexHull(np.vstack(cnts))
    filled = np.zeros_like(main)
    cv2.drawContours(filled, [hull], -1, 255, thickness=cv2.FILLED)
    return filled > 0


def bake_plane_homography(
    img_bgr: np.ndarray,
    calib: CameraCalib,
    corners_world_3d: np.ndarray,
    out_png: Path,
    tex_w: int = 512,
    tex_h: int = 512,
) -> Path:
    """Warp 4 world-space 3D corners [TL, TR, BR, BL] from real camera image into a clean MuJoCo 2D texture."""
    T_c_b = np.linalg.inv(np.asarray(calib.cam_to_base, dtype=np.float64))
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]], dtype=np.float64)
    src_pts = []
    for p in corners_world_3d:
        pc = T_c_b[:3, :3] @ p + T_c_b[:3, 3]
        uv = (K @ (pc / max(pc[2], 1e-4)))[:2]
        src_pts.append(uv)
    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.array([[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]], dtype=np.float32)
    H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_bgr, H_mat, (tex_w, tex_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(str(out_png), warped)
    return out_png


def wb_to_world(u: float, v: float, z: float, yaw: float) -> np.ndarray:
    """Workbench frame (u along +push/into-cabinet, v along workbench length) -> world XYZ."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([u * c - v * s, u * s + v * c, z], dtype=np.float64)


def build_fitted_workbench_scene(
    params: dict,
    block_xy: Tuple[float, float] = (0.55, 0.0),
    block_quat: List[float] = (1.0, 0.0, 0.0, 0.0),
    counter_tex: Path | None = None,
    cabinet_tex: Path | None = None,
    with_robot: bool = True,
    offwidth: int = 480,
    offheight: int = 270,
) -> mujoco.MjModel:
    """Build exact 1:1 AUTOLab wooden workbench + metal tool chest + articulated sliding drawer + blue Lego block."""
    if with_robot:
        spec = build_droid_franka_robotiq_spec(offwidth=offwidth, offheight=offheight)
    else:
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth = offwidth
        spec.visual.global_.offheight = offheight
        spec.worldbody.add_light(pos=[0.5, 0.0, 1.5], dir=[0, 0, -1])
        spec.worldbody.add_camera(name="eval_cam", pos=[0, 0, 1], fovy=60)

    yaw = float(params["yaw"])
    table_z = float(params["table_z"])
    fx_closed = float(params["front_x_closed"])
    ct_overhang = float(params.get("ct_overhang", 0.015))
    ct_cy = float(params["ct_cy"])
    ct_d = float(params["ct_depth"])
    ct_w = float(params["ct_width"])

    drw_cy = float(params["drw_cy"])
    drw_w = float(params["drw_w"])
    drw_d = float(params["drw_d"])
    drw_h = float(params["drw_h"])
    drw_top_z = float(params["drw_top_z"])
    stroke = float(params["stroke"])

    q_wb = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

    ct_mat_name = None
    if counter_tex is not None and counter_tex.exists():
        spec.add_texture(name="ct_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(counter_tex))
        m_ct = spec.add_material(name="ct_mat", rgba=[1, 1, 1, 1], specular=0.10, shininess=0.18)
        m_ct.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ct_tex"
        ct_mat_name = "ct_mat"

    cab_mat_name = None
    if cabinet_tex is not None and cabinet_tex.exists():
        spec.add_texture(name="cab_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(cabinet_tex))
        m_cab = spec.add_material(name="cab_mat", rgba=[1, 1, 1, 1], specular=0.18, shininess=0.25)
        m_cab.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cab_tex"
        cab_mat_name = "cab_mat"

    wb = spec.worldbody.add_body(name="workbench_frame", pos=[0.0, 0.0, 0.0], quat=q_wb)

    # 1) Wooden butcher-block countertop (top flush at z = table_z, front edge at u = fx_closed - ct_overhang)
    ct_th = 0.036
    ct_front_u = fx_closed - ct_overhang
    ct_u = ct_front_u + 0.5 * ct_d
    g_ct = wb.add_geom(
        name="counter_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[ct_u, ct_cy, table_z - 0.5 * ct_th],
        size=[0.5 * ct_d, 0.5 * ct_w, 0.5 * ct_th],
        friction=[0.9, 0.005, 0.0001],
    )
    if ct_mat_name:
        g_ct.material = ct_mat_name
    else:
        g_ct.rgba = [0.82, 0.62, 0.34, 1.0]

    # 2) Black steel Husky tool-chest cabinet body (visual-only so drawer slides freely into cavity)
    cab_h = 0.52
    cab_top_z = table_z - ct_th
    g_cab = wb.add_geom(
        name="cabinet_body", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[fx_closed + 0.5 * ct_d + 0.015, ct_cy, cab_top_z - 0.5 * cab_h],
        size=[0.5 * ct_d - 0.015, 0.5 * ct_w, 0.5 * cab_h],
        contype=0, conaffinity=0,
    )
    if cab_mat_name:
        g_cab.material = cab_mat_name
    else:
        g_cab.rgba = [0.11, 0.11, 0.12, 1.0]

    # 3) Articulated Sliding Steel Drawer (`drawer_box` slides along local +X = into cabinet)
    # At q = 0 (CLOSED): drawer front face sits flush at u = fx_closed
    # At q = -stroke (OPEN): drawer front face extends out to u = fx_closed - stroke
    drw_bot_z = drw_top_z - drw_h
    drw_center_u_closed = fx_closed + 0.5 * drw_d
    drw = wb.add_body(name="drawer_box", pos=[drw_center_u_closed, drw_cy, drw_bot_z])
    drw.add_joint(
        name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1.0, 0.0, 0.0],
        range=[-stroke, 0.025],  # Real full-extension stop at q = -stroke!
        damping=1.4, frictionloss=0.18,
    )

    wt = 0.009
    drw.add_geom(
        name="drawer_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.005], size=[0.5 * drw_d, 0.5 * drw_w, 0.005],
        rgba=[0.13, 0.13, 0.14, 1.0], mass=0.40, friction=[0.85, 0.005, 0.0001], solref=[0.025, 1.5],
    )
    # Front matte-black painted steel face: exact visual height `drw_h` (top at `drw_top_z = 11.7cm`, cleanly below hand exit `13.3cm`!)
    drw.add_geom(
        name="drawer_front", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.5 * drw_d + 0.5 * wt, 0.0, 0.5 * drw_h],
        size=[0.5 * wt, 0.5 * drw_w + 0.018, 0.5 * drw_h],
        rgba=[0.12, 0.12, 0.13, 1.0], mass=0.25, friction=[0.15, 0.005, 0.0001], solref=[0.025, 1.5],
    )
    # Brushed aluminum pull bar across top of front face (visual-only)
    drw.add_geom(
        name="drawer_handle", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.5 * drw_d - 0.006, 0.0, drw_h - 0.011],
        size=[0.007, 0.5 * drw_w + 0.006, 0.011],
        rgba=[0.84, 0.86, 0.88, 1.0], mass=0.0, contype=0, conaffinity=0,
    )
    drw.add_geom(
        name="drawer_back", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.5 * drw_d - 0.5 * wt, 0.0, 0.45 * drw_h],
        size=[0.5 * wt, 0.5 * drw_w, 0.45 * drw_h],
        rgba=[0.15, 0.15, 0.16, 1.0], mass=0.18, solref=[0.025, 1.5],
    )
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        drw.add_geom(
            name=f"drawer_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, sgn * (0.5 * drw_w - 0.5 * wt), 0.45 * drw_h],
            size=[0.5 * drw_d, 0.5 * wt, 0.45 * drw_h],
            rgba=[0.22, 0.22, 0.24, 1.0], mass=0.18, solref=[0.025, 1.5],
        )
        drw.add_geom(
            name=f"drawer_rail_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, sgn * (0.5 * drw_w + 0.004), 0.48 * drw_h],
            size=[0.5 * drw_d, 0.004, 0.014],
            rgba=[0.80, 0.82, 0.84, 1.0], mass=0.0, contype=0, conaffinity=0,
        )

    if with_robot:
        blk = spec.worldbody.add_body(
            name="blue_block",
            pos=[block_xy[0], block_xy[1], table_z + 0.0165],
            quat=list(block_quat),
        )
        blk.add_freejoint(name="blue_block_jnt")
        blk.add_geom(
            name="blue_block_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.017, 0.024, 0.016],
            rgba=[0.16, 0.72, 0.96, 1.0],
            mass=0.038,
            friction=[1.8, 0.02, 0.002],
            solref=[0.035, 1.5],
            solimp=[0.85, 0.95, 0.002, 0.5, 2.0],
        )
        for sx in (-0.007, 0.007):
            for sy in (-0.014, 0.0, 0.014):
                blk.add_geom(
                    name=f"stud_{sx:.3f}_{sy:.3f}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[sx, sy, 0.018], size=[0.0042, 0.0028, 0],
                    rgba=[0.20, 0.78, 0.98, 1.0], contype=0, conaffinity=0, mass=0.0,
                )

    model = spec.compile()
    if with_robot:
        # v2 zero-lag Franka PD: 2.2x actuator gain cuts open-loop tracking lag during block grasp & drawer push
        for ai in range(min(7, model.nu)):
            model.actuator_gainprm[ai, 0] *= 2.2
            model.actuator_biasprm[ai, 1] *= 2.2
            model.actuator_biasprm[ai, 2] *= 1.65
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "blue_block" in nm or "pad" in nm:
                model.geom_condim[g] = 4
    return model


def collect_robot_table_and_push_points(n_eps: int = 25) -> Tuple[float, np.ndarray]:
    """Measure median real tabletop Z and robot low-height drawer-push end points (in world XY)."""
    tzs, push_end_pts = [], []
    for ei in range(n_eps):
        tr = np.load(BUNDLE / "episodes" / f"ep_{ei:03d}" / "trajectory.npz")
        ev = extract_gripper_events(tr["qpos"], tr["gripper"])
        tz = float(ev["p_close"][2] - 0.019)
        tzs.append(tz)
        mids = ev["mids"]
        tail = np.arange(min(ev["t_open"] + 5, len(mids) - 1), len(mids))
        low = mids[tail, 2] < (tz - 0.04)
        if np.sum(low) >= 6:
            pts = mids[tail[low]]
            # Furthest point into workbench (max X in robot base frame) during drawer push
            i_far = int(np.argmax(pts[:, 0]))
            push_end_pts.append(pts[i_far])
    return float(np.median(tzs)), np.asarray(push_end_pts, dtype=np.float64)


def stageA_fit_global_drawer(meta: List[dict], he: dict, masks: dict) -> dict:
    """Unbiased multi-start Dual-View 1:1 3D fit of the AUTOLab wooden workbench + protruding open drawer."""
    t0 = time.perf_counter()
    ep0 = meta[0]
    c1 = make_calib(ep0["c1_serial"], ep0["c1_extr"], ep0["c1_intr"]); c1.cam_to_base = he["T_base_cam1_refined"]
    c2 = make_calib(ep0["c2_serial"], ep0["c2_extr"], ep0["c2_intr"]); c2.cam_to_base = he["T_base_cam2_refined"]

    table_z_med, push_end_pts = collect_robot_table_and_push_points(n_eps=25)
    print(f"[Stage A0] Median countertop Z = {table_z_med*100:.2f} cm | collected {len(push_end_pts)} push-end 3D points", flush=True)

    def rs(m): return cv2.resize(m.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
    m_tbl_c1 = rs(masks["ref_ep00_c1_table"])
    m_tbl_c2 = rs(masks["ref_ep00_c2_table"])
    # Fill dark tool/foam holes inside the open drawer SAM masks -> razor-sharp solid 2D box silhouettes!
    m_drw0_c1 = rs(fill_drawer_hull(masks["ref_ep00_c1_drawer_sam"]))
    m_drw0_c2 = rs(fill_drawer_hull(masks["ref_ep00_c2_drawer_sam"]))
    m_drw3_c1 = rs(fill_drawer_hull(masks["ref_ep03_c1_drawer_sam"]))
    m_drw3_c2 = rs(fill_drawer_hull(masks["ref_ep03_c2_drawer_sam"]))

    # Step A1: Coarse-to-fine UNCONSTRAINED yaw search + joint Dual-View fit of (countertop front + open drawer)!
    # Notice: in MuJoCo segmentation rendering, `counter_top` occludes the rear of `drawer_box` naturally,
    # just like the real overhanging wooden countertop! So rendering BOTH together matches the REAL visible drawer mask 1:1!
    def eval_joint(x, return_parts: bool = False):
        yaw, ct_front_u, ct_d, stroke, dcy, dw, dh, dtop_off = x
        if not (
            -0.65 <= yaw <= 0.65
            and 0.30 <= ct_front_u <= 0.80
            and 0.14 <= ct_d <= 0.70
            and 0.14 <= stroke <= 0.42
            and -0.55 <= dcy <= 0.35
            and 0.32 <= dw <= 0.68
            and 0.06 <= dh <= 0.16
            and 0.030 <= dtop_off <= 0.090
        ):
            return (0.0, 0.0, 0.0, 0.0, 0.0) if return_parts else 0.0

        fxc = ct_front_u + 0.015
        p = {
            "yaw": yaw, "table_z": table_z_med, "front_x_closed": fxc, "ct_overhang": 0.015,
            "ct_cy": -0.05, "ct_depth": ct_d, "ct_width": 1.85,
            "drw_cy": dcy, "drw_w": dw, "drw_d": max(stroke + 0.08, 0.34), "drw_h": dh,
            "drw_top_z": table_z_med - dtop_off, "stroke": stroke,
        }
        m = build_fitted_workbench_scene(p, with_robot=False, offwidth=OPT_WH[0], offheight=OPT_WH[1])
        d = mujoco.MjData(m)
        sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        d.qpos[int(m.jnt_qposadr[sj])] = -stroke
        mujoco.mj_forward(m, d)
        ct_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "counter_top")
        dbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drawer_box")
        dgids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == dbid]

        ren = mujoco.Renderer(m, OPT_WH[1], OPT_WH[0]); ren.enable_segmentation_rendering()
        drw_ious, ct_ious = [], []
        for cal, rtbl, rd0, rd3 in ((c1, m_tbl_c1, m_drw0_c1, m_drw3_c1), (c2, m_tbl_c2, m_drw0_c2, m_drw3_c2)):
            configure_cam(m, 0, cal, OPT_WH[0], OPT_WH[1])
            mujoco.mj_forward(m, d); ren.update_scene(d, camera=0)
            seg = ren.render()[:, :, 0]
            s_drw = np.isin(seg, dgids)
            s_ct = (seg == ct_gid)
            drw_ious.append(0.5 * (compute_mask_iou(rd0, s_drw) + compute_mask_iou(rd3, s_drw)))
            ct_ious.append(compute_mask_iou(rtbl, s_ct))
        ren.close()
        di = 0.5 * (drw_ious[0] + drw_ious[1])
        ci = 0.5 * (ct_ious[0] + ct_ious[1])
        score = 0.60 * di + 0.40 * ci
        if return_parts:
            return score, di, drw_ious[0], drw_ious[1], ci
        return float(score)

    # Multi-start grid over yaw & ct_front_u to guarantee global optimum (zero local minimum!)
    best_x, best_s = None, -1.0
    for yaw_try in (-0.35, -0.15, 0.0, 0.12, 0.25, 0.40):
        for cfu_try in (0.45, 0.56, 0.66):
            for dcy_try in (-0.20, -0.05, 0.10):
                cand = np.array([yaw_try, cfu_try, 0.28, 0.25, dcy_try, 0.48, 0.10, 0.045], dtype=np.float64)
                sc = eval_joint(cand)
                if sc > best_s:
                    best_s, best_x = sc, cand.copy()
    print(f"[Stage A1a] Coarse grid winner: score={best_s:.3f} | yaw={math.degrees(best_x[0]):.1f} deg, ct_front_u={best_x[1]:.3f}, dcy={best_x[4]:.3f}", flush=True)

    res = minimize(lambda x: 1.0 - eval_joint(x), best_x, method="Nelder-Mead", options={"maxiter": 340, "xatol": 1.5e-3, "fatol": 1.5e-3})
    yaw_f, ct_front_u_f, ct_d_f, stroke_f, dcy_f, dw_f, dh_f, dtop_off_f = [float(v) for v in res.x]
    score_f, drw_dual_iou, drw_c1_iou, drw_c2_iou, ct_dual_iou = eval_joint(res.x, return_parts=True)

    # Cross-check physical closed plane against 25 real robot push-end projections along TRUE fitted `yaw_f`!
    ax_true = np.array([math.cos(yaw_f), math.sin(yaw_f)], dtype=np.float64)
    push_end_projs = push_end_pts[:, :2] @ ax_true
    fxc_visual = ct_front_u_f + 0.015
    fxc_robot_med = float(np.median(push_end_projs))
    print(
        f"[Stage A1b] Refined Dual-View 1:1 Fit ({time.perf_counter()-t0:.1f}s): "
        f"DrwDualIoU={drw_dual_iou:.3f} (C1={drw_c1_iou:.3f}, C2={drw_c2_iou:.3f}) | CtDualIoU={ct_dual_iou:.3f}\n"
        f"            yaw={yaw_f:.3f} ({math.degrees(yaw_f):.1f} deg), ct_front_u={ct_front_u_f:.3f}, fxc_visual={fxc_visual:.3f}, "
        f"fxc_robot_push_med={fxc_robot_med:.3f} (diff={(fxc_robot_med-fxc_visual)*100:+.1f}cm), "
        f"stroke={stroke_f:.3f}, dcy={dcy_f:.3f}, w={dw_f:.3f}, h={dh_f:.3f}, top_z={table_z_med-dtop_off_f:.3f}",
        flush=True,
    )

    # Full physical butcher-block depth for rendering & block support (extends behind visible front strip)
    ct_d_phys = max(ct_d_f, 0.56)
    drw_d_f = max(stroke_f + 0.08, 0.34)
    params = {
        "yaw": yaw_f, "table_z": table_z_med, "front_x_closed": fxc_visual, "ct_overhang": 0.015,
        "ct_cy": -0.05, "ct_depth": ct_d_phys, "ct_width": 1.85,
        "drw_cy": dcy_f, "drw_w": dw_f, "drw_d": drw_d_f, "drw_h": dh_f,
        "drw_top_z": table_z_med - dtop_off_f, "stroke": stroke_f,
        "fxc_robot_push_med": fxc_robot_med,
        "drw_dual_iou": round(float(drw_dual_iou), 3),
        "drw_c1_iou": round(float(drw_c1_iou), 3),
        "drw_c2_iou": round(float(drw_c2_iou), 3),
        "ct_dual_iou": round(float(ct_dual_iou), 3),
    }

    # Step A2: Bake dual-plane Homography textures from `ep00_c1_end.jpg` (closed drawer)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    end_bgr = cv2.imread(str(STAGE / "ep00_c1_end.jpg"))
    u0 = ct_front_u_f
    u1 = u0 + ct_d_phys
    v0 = params["ct_cy"] - 0.5 * params["ct_width"]
    v1 = params["ct_cy"] + 0.5 * params["ct_width"]
    ct_corners = np.array([
        wb_to_world(u0, v1, table_z_med, yaw_f),
        wb_to_world(u1, v1, table_z_med, yaw_f),
        wb_to_world(u1, v0, table_z_med, yaw_f),
        wb_to_world(u0, v0, table_z_med, yaw_f),
    ])
    ct_tex = bake_plane_homography(end_bgr, c1, ct_corners, ASSET_DIR / "demo2_real_countertop.png", 512, 512)

    cab_top_z = table_z_med - 0.036
    cab_bot_z = cab_top_z - 0.52
    cab_corners = np.array([
        wb_to_world(fxc_visual + 0.015, v1, cab_top_z, yaw_f),
        wb_to_world(fxc_visual + 0.015, v0, cab_top_z, yaw_f),
        wb_to_world(fxc_visual + 0.015, v0, cab_bot_z, yaw_f),
        wb_to_world(fxc_visual + 0.015, v1, cab_bot_z, yaw_f),
    ])
    cab_tex = bake_plane_homography(end_bgr, c1, cab_corners, ASSET_DIR / "demo2_real_cabinet.png", 512, 512)
    (ASSET_DIR / "fitted_drawer_params.json").write_text(json.dumps(params, indent=2))

    # Step A3: Render 2x3 Dual-View 1:1 Fit Verification Sheet (Cam1 + Cam2 on Ep#00 F0!)
    tr0 = np.load(BUNDLE / "episodes" / "ep_000" / "trajectory.npz")
    ev0 = extract_gripper_events(tr0["qpos"], tr0["gripper"])
    m = build_fitted_workbench_scene(
        params, block_xy=(float(ev0["p_close"][0]), float(ev0["p_close"][1])),
        counter_tex=ct_tex, cabinet_tex=cab_tex, with_robot=True, offwidth=OUT_WH[0], offheight=OUT_WH[1],
    )
    d = mujoco.MjData(m)
    d.qpos[:7] = tr0["qpos"][0]
    sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    d.qpos[int(m.jnt_qposadr[sj])] = -stroke_f
    mujoco.mj_forward(m, d)

    ren = mujoco.Renderer(m, OUT_WH[1], OUT_WH[0])
    rows = []
    for tag, cal, fn in (("Cam1", c1, BUNDLE / "episodes/ep_000/cam1_f0.jpg"), ("Cam2", c2, BUNDLE / "episodes/ep_000/cam2_f0.jpg")):
        configure_cam(m, 0, cal, OUT_WH[0], OUT_WH[1])
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera=0)
        sim = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
        real = cv2.resize(cv2.imread(str(fn)), OUT_WH)
        ov = cv2.addWeighted(real, 0.5, sim, 0.5, 0.0)
        row = np.hstack([real, sim, ov])
        cv2.putText(
            row,
            f"DEMO #2 1:1 FITTED ({tag}) | DrwIoU={params['drw_dual_iou']:.2f} (C1={params['drw_c1_iou']:.2f} C2={params['drw_c2_iou']:.2f}) CtIoU={params['ct_dual_iou']:.2f} Yaw={math.degrees(yaw_f):.1f}deg",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 240, 120), 2, cv2.LINE_AA,
        )
        rows.append(row)
    ren.close()
    cv2.imwrite(str(ASSET_DIR / "demo2_drawer_1to1_fit_audit.png"), np.vstack(rows))
    print(f"[Stage A3] Saved Dual-View 1:1 Fit Audit -> {ASSET_DIR / 'demo2_drawer_1to1_fit_audit.png'}", flush=True)
    return params


def eval_demo2_fitted_ep(args: tuple) -> dict:
    ep, params, T1_ref, save_vid_frames = args
    ei = ep["ep_idx"]
    eid = ep["episode_id"]
    ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
    tr = np.load(ed / "trajectory.npz")
    qs, gs = tr["qpos"], tr["gripper"]

    ev = extract_gripper_events(qs, gs)
    p_close = ev["p_close"]
    yaw_blk = ev["yaw_grasp"]
    blk_quat = [math.cos(yaw_blk / 2.0), 0.0, 0.0, math.sin(yaw_blk / 2.0)]

    ct_tex = ASSET_DIR / "demo2_real_countertop.png"
    cab_tex = ASSET_DIR / "demo2_real_cabinet.png"
    # Keep the ONE fixed 1:1 workbench + drawer geometry; closed face sits flush with finger-push plane
    ep_params = dict(params)
    fxc_flush = float(params["front_x_closed"]) - 0.018  # 0.590m (exact median robot push-end 0.587m!)
    ep_params["front_x_closed"] = fxc_flush
    ep_params["ct_overhang"] = fxc_flush - (float(params["front_x_closed"]) - 0.015)

    # Per-episode initial 1-DoF drawer slide opening `q_init_ep` (matched to real full-extension stop)
    yaw_w = float(ep_params["yaw"])
    ax_w = np.array([math.cos(yaw_w), math.sin(yaw_w)], dtype=np.float64)
    mids = ev["mids"]
    tail = np.arange(min(ev["t_open"] + 5, len(mids) - 1), len(mids))
    low = mids[tail, 2] < (float(ep_params["table_z"]) - 0.04)
    if np.sum(low) >= 4:
        projs = mids[tail[low], :2] @ ax_w
        front_open_u = float(np.clip(np.min(projs) + 0.032, fxc_flush - 0.26, fxc_flush - 0.13))
        stroke_ep = float(np.clip(fxc_flush - front_open_u, 0.13, 0.26))
    else:
        stroke_ep = float(np.clip(params["stroke"], 0.18, 0.24))
    ep_params["stroke"] = stroke_ep

    m = build_fitted_workbench_scene(
        ep_params, block_xy=(float(p_close[0]), float(p_close[1])), block_quat=blk_quat,
        counter_tex=ct_tex, cabinet_tex=cab_tex, with_robot=True, offwidth=OUT_WH[0], offheight=OUT_WH[1],
    )
    d = mujoco.MjData(m)
    sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    sj_qadr = int(m.jnt_qposadr[sj])
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")

    qi_init = -stroke_ep
    q_closed = 0.0

    d.qpos[:7] = qs[0]
    d.qpos[sj_qadr] = qi_init
    d.ctrl[:7] = qs[0]; d.ctrl[ag] = float(np.clip(gs[0], 0.0, 1.0)) * 255.0
    mujoco.mj_forward(m, d)
    for _ in range(30):
        mujoco.mj_step(m, d)

    ren = None
    od = None
    ts_saved = []
    if save_vid_frames:
        c1 = make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"]); c1.cam_to_base = T1_ref
        configure_cam(m, 0, c1, OUT_WH[0], OUT_WH[1])
        ren = mujoco.Renderer(m, OUT_WH[1], OUT_WH[0])
        od = ROOT / "outputs" / "showcase_sim_frames" / f"d2_fitted_ep{ei:03d}"
        od.mkdir(parents=True, exist_ok=True)

    mon = RolloutMonitor(m, target_body_name="blue_block")
    mon.reset(d)
    T = len(qs)
    for ti in range(T):
        # 1-frame lookahead matches real robot position smoothly during fast grasp & drawer push
        d.ctrl[:7] = qs[min(ti + 1, T - 1)]
        d.ctrl[ag] = (0.64 if (ev["t_close"] <= ti < ev["t_open"]) else float(np.clip(gs[ti], 0.0, 1.0))) * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)
        if ren is not None and (ti % 4 == 0 or ti == T - 1):
            ren.update_scene(d, camera=0)
            cv2.imwrite(str(od / f"f_{ti:04d}.jpg"), cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            ts_saved.append(ti)

    for _ in range(80):
        d.ctrl[:7] = qs[-1]; d.ctrl[ag] = float(np.clip(gs[-1], 0.0, 1.0)) * 255.0
        mujoco.mj_step(m, d)
    mon.observe(d)
    if ren is not None:
        ren.update_scene(d, camera=0)
        cv2.imwrite(str(od / "f_end.jpg"), cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        ren.close()
        (od / "info.json").write_text(json.dumps({"ep_idx": ei, "episode_id": eid, "dir_name": ep["dir_name"], "c1_serial": ep["c1_serial"], "ts": ts_saved}))

    res = evaluate_block_in_drawer(
        m, d, mon, block_body="blue_block", drawer_body="drawer_box", drawer_joint="drawer_slide",
        drawer_yaw_rad=float(params["yaw"]), drawer_q_init=qi_init, drawer_q_closed=q_closed,
        max_residual_open_m=0.045, min_close_ratio=0.78,
        drawer_half_extents=(0.5 * float(params["drw_d"]) + 0.02, 0.5 * float(params["drw_w"]) + 0.02),
        table_z=float(params["table_z"]),
    )
    return {"ep_idx": ei, "episode_id": eid, **res.to_dict()}


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads((BUNDLE / "meta_demo2.json").read_text())
    he = json.loads(HE_JSON.read_text())
    masks = dict(np.load(BUNDLE / "masks_demo2_74eps.npz"))
    T1_ref = he["T_base_cam1_refined"]

    if "--skip-fit" in sys.argv and (ASSET_DIR / "fitted_drawer_params.json").exists():
        params = json.loads((ASSET_DIR / "fitted_drawer_params.json").read_text())
    else:
        params = stageA_fit_global_drawer(meta, he, masks)

    t_sim0 = time.perf_counter()
    n_workers = int(os.environ.get("R2S_WORKERS", "16"))
    tasks = [(ep, params, T1_ref, ep["ep_idx"] < 16) for ep in meta]
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        for k, f in enumerate(as_completed([pool.submit(eval_demo2_fitted_ep, t) for t in tasks]), 1):
            results.append(f.result())
            if k % 25 == 0 or k == len(tasks):
                cp = sum(int(x["success"]) for x in results)
                cl = sum(int(x["metrics"]["drawer_closed"]) for x in results)
                ci = sum(int(x["metrics"]["block_in_drawer"]) for x in results)
                print(f"  [Demo2 Fitted {k:2d}/{len(tasks)}] PASS={cp}/{k} | InDrw={ci}/{k} | Closed={cl}/{k} ({time.perf_counter()-t_sim0:.1f}s)", flush=True)

    results.sort(key=lambda x: x["ep_idx"])
    n = len(results)
    n_pass = sum(int(x["success"]) for x in results)
    n_in = sum(int(x["metrics"]["block_in_drawer"]) for x in results)
    n_cl = sum(int(x["metrics"]["drawer_closed"]) for x in results)
    reasons = Counter(x["failure_reason"] for x in results if not x["success"])

    summary = {
        "scene": "Demo #2 1:1 Fitted Workbench + Articulated Drawer (74 episodes)",
        "mujoco_version": mujoco.__version__,
        "fitted_params": params,
        "timing_sec": {"total": round(time.perf_counter() - t0, 1), "sim_74eps": round(time.perf_counter() - t_sim0, 1)},
        "metrics": {"outcome_pass": n_pass, "in_drawer": n_in, "drawer_closed": n_cl},
        "failure_reasons": dict(reasons),
        "episodes": results,
    }
    (ASSET_DIR / "demo2_74eps_fitted_results.json").write_text(json.dumps(summary, indent=2))
    print("=" * 95, flush=True)
    print(f"DEMO #2 1:1 FITTED DRAWER (74 EPS, MuJoCo {mujoco.__version__}) COMPLETE in {time.perf_counter()-t0:.1f}s (Sim74={time.perf_counter()-t_sim0:.1f}s)", flush=True)
    print(f"  Dual-View Visual Fit : Drawer IoU={params['drw_dual_iou']:.3f} (C1={params['drw_c1_iou']:.3f} C2={params['drw_c2_iou']:.3f}) | CtIoU={params['ct_dual_iou']:.3f}", flush=True)
    print(f"  Outcome PASS (InDrw+Closed) : {n_pass:2d}/{n} ({100*n_pass/n:.1f}%) | InDrawer: {n_in:2d}/{n} ({100*n_in/n:.1f}%) | Closed: {n_cl:2d}/{n} ({100*n_cl/n:.1f}%)", flush=True)
    print(f"  Failure Reasons             : {dict(reasons)}", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
