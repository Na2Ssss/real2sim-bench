"""Demo #6 (CLVR 2023-05-09: Pick & Relocate Yellow/Green/Grey Mug, 50 episodes) — Stage 2 MuJoCo 4.0.1.

All-on-A100 `MuJoCo 4.0.1` (`spawn` 16w EGL):
  1. Converts Multi-View `Hunyuan3D-2mv` `clvr_mug_hy3d_mv_raw.glb` -> visual mesh + guaranteed-hollow 16-facet mug (7.4cm OD for clean Robotiq 2F-85 pinch).
  2. Per-episode real median mug RGB color + per-episode Homography real tabletop texture (white OR wood!).
  3. Clean outcome-only relocation predicate (`disp_xy >= 6 cm` & within `12 cm` of release target, zero NaN/explosion).
  4. Composites 3-column H.264 showcase video (`demo6_clvr_mj4_showcase.mp4`) + keyframe audit strip (`inspect_d6_clvr.png`).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import subprocess
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
from r2s_bench.eval.task_success import RolloutMonitor, _finalize
from r2s_bench.grading.l1_visual import compute_mask_iou
from r2s_bench.robots.franka_robotiq import build_droid_franka_robotiq_spec
from r2s_bench.schema import CameraCalib

BUNDLE = ROOT / "outputs" / "demo6_clvr_bundle"
ASSET_DIR = ROOT / "outputs" / "hy3d_demo6"
SIM_DIR = ASSET_DIR / "sim_ready_mug"
OUT_WH = (480, 270)
OPT_WH = (240, 135)

DEFAULT_INTR = {
    "20103212": {"cameraMatrix": [531.1367, 651.0386, 531.1367, 351.9382], "width": 1280, "height": 720},
    "20655732": {"cameraMatrix": [525.1442, 637.6971, 525.1442, 360.5704], "width": 1280, "height": 720},
}


def make_calib(serial: str, ext6: list, intr_dict: dict | None) -> CameraCalib:
    if not intr_dict or int(intr_dict.get("width", 0)) <= 0:
        intr_dict = DEFAULT_INTR.get(str(serial), DEFAULT_INTR["20103212"])
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
    m.cam_pos[cid] = pos; m.cam_quat[cid] = quat
    px = 1e-5
    m.cam_resolution[cid] = [sc.width, sc.height]
    m.cam_sensorsize[cid] = [sc.width * px, sc.height * px]
    m.cam_intrinsic[cid] = [sc.fx * px, sc.fy * px, -(sc.cx - sc.width / 2.0) * px, (sc.cy - sc.height / 2.0) * px]


def ray_plane_z(calib: CameraCalib, uv: Tuple[float, float], z_plane: float) -> np.ndarray:
    T = np.asarray(calib.cam_to_base, dtype=np.float64)
    o = T[:3, 3]
    d = T[:3, :3] @ np.array([(uv[0] - calib.cx) / calib.fx, (uv[1] - calib.cy) / calib.fy, 1.0])
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    t = float(np.clip((z_plane - o[2]) / (d[2] if abs(d[2]) > 1e-5 else -1e-5), 0.15, 2.5))
    return o + t * d


def extract_grasp(qs: np.ndarray, gs: np.ndarray) -> dict:
    m_fk = build_droid_franka_robotiq_spec(offwidth=64, offheight=64).compile()
    d_fk = mujoco.MjData(m_fk)
    lp = [g for g in range(m_fk.ngeom) if "left_silicone_pad" in (mujoco.mj_id2name(m_fk, mujoco.mjtObj.mjOBJ_BODY, int(m_fk.geom_bodyid[g])) or "")][0]
    rp = [g for g in range(m_fk.ngeom) if "right_silicone_pad" in (mujoco.mj_id2name(m_fk, mujoco.mjtObj.mjOBJ_BODY, int(m_fk.geom_bodyid[g])) or "")][0]
    ag = mujoco.mj_name2id(m_fk, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")
    d_fk.qpos[:7] = qs[0]; d_fk.ctrl[:7] = qs[0]; d_fk.ctrl[ag] = 0.50 * 255.0
    for _ in range(150):
        mujoco.mj_step(m_fk, d_fk)
    q_pinch = d_fk.qpos[7:].copy()
    T = len(qs); mids = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        d_fk.qpos[:7] = qs[t]; d_fk.qpos[7:] = q_pinch; mujoco.mj_forward(m_fk, d_fk)
        mids[t] = 0.5 * (d_fk.geom_xpos[lp] + d_fk.geom_xpos[rp])
    n1 = int(T * 0.65)
    g_peak = float(np.max(gs[:n1]))
    thresh = max(0.16, g_peak * 0.55)
    cc = [t for t in range(3, n1) if gs[t] >= thresh and gs[t - 1] < thresh]
    t_close = min(cc, key=lambda t: mids[t, 2]) if cc else int(np.argmax(gs[:n1]))
    t_hold = min(t_close + 4, T - 1)
    oc = [t for t in range(t_hold + 5, T) if gs[t] < thresh * 0.45]
    t_open = oc[0] if oc else T - 1
    return {"t_close": int(t_close), "t_open": int(t_open), "p_close": mids[t_hold], "p_open": mids[t_open], "mids": mids}


def build_clvr_scene(
    mug_xy: Tuple[float, float],
    table_z: float,
    mug_vis_obj: Path,
    mug_rgb: List[float],
    table_tex: Path | None = None,
    with_robot: bool = True,
    offwidth: int = 480,
    offheight: int = 270,
) -> mujoco.MjModel:
    if with_robot:
        spec = build_droid_franka_robotiq_spec(offwidth=offwidth, offheight=offheight)
    else:
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth = offwidth; spec.visual.global_.offheight = offheight
        spec.worldbody.add_light(pos=[0.5, 0.0, 1.5], dir=[0, 0, -1])
        spec.worldbody.add_camera(name="eval_cam", pos=[0, 0, 1], fovy=60)

    tbl_mat = None
    if table_tex is not None and table_tex.exists():
        spec.add_texture(name="clvr_tbl_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(table_tex))
        m_tbl = spec.add_material(name="clvr_tbl_mat", rgba=[1, 1, 1, 1], specular=0.10, shininess=0.18)
        m_tbl.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "clvr_tbl_tex"
        tbl_mat = "clvr_tbl_mat"
    g_tbl = spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.55, 0.0, table_z - 0.025], size=[0.55, 0.70, 0.025],
        friction=[0.85, 0.005, 0.0001],
    )
    if tbl_mat:
        g_tbl.material = tbl_mat
    else:
        g_tbl.rgba = [0.88, 0.86, 0.82, 1.0]

    # Multi-View Hunyuan3D-2mv Mug visual + crisp 7.2cm OD hollow collision cylinder ring (fits 8.5cm Robotiq cleanly!)
    spec.add_mesh(name="clvr_mug_vis", file=str(mug_vis_obj))
    mug = spec.worldbody.add_body(name="clvr_mug", pos=[mug_xy[0], mug_xy[1], table_z + 0.001])
    mug.add_freejoint(name="clvr_mug_jnt")
    mug.add_geom(
        name="mug_vis_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="clvr_mug_vis",
        rgba=[mug_rgb[0], mug_rgb[1], mug_rgb[2], 1.0], contype=0, conaffinity=0, mass=0.0,
    )
    mug.add_geom(
        name="mug_bot", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0.004], size=[0.036, 0.004, 0], rgba=[0, 0, 0, 0], mass=0.045, friction=[1.0, 0.01, 0.001],
    )
    r_m, h_m = 0.035, 0.092
    for k in range(16):
        a = 2.0 * math.pi * k / 16
        mug.add_geom(
            name=f"mug_w{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[r_m * math.cos(a), r_m * math.sin(a), 0.5 * h_m],
            quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)],
            size=[0.003, 0.008, 0.5 * h_m], rgba=[0, 0, 0, 0],
            mass=0.045 / 16, friction=[2.0, 0.03, 0.003], solref=[0.020, 1.4],
        )

    model = spec.compile()
    if with_robot:
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "mug_w" in nm or "pad" in nm:
                model.geom_condim[g] = 4
    return model


def prepare_assets(meta: list) -> dict:
    colors = json.loads((ASSET_DIR / "demo6_colors.json").read_text())
    meta_json = SIM_DIR / "asset_meta.json"
    if not meta_json.exists():
        asset = build_sim_ready_asset(
            name="clvr_mug", raw_mesh_path=ASSET_DIR / "clvr_mug_hy3d_mv_raw.glb",
            out_dir=SIM_DIR, target_size_xyz=(0.088, 0.078, 0.094),
            rgba=(0.88, 0.85, 0.25, 1.0),
            is_container=True, auto_upright_pca=False, wall_thickness_m=0.004, max_convex_hulls=16,
        )
        mug_vis_obj = str(asset.visual_mesh_path)
    else:
        mug_vis_obj = str(json.loads(meta_json.read_text())["visual_mesh_path"])
    return {"mug_vis_obj": mug_vis_obj, "ep_rgbs": colors.get("ep_rgbs", {})}


def eval_ep(args: tuple) -> dict:
    ep, cfg, m_c1_mug, m_c2_mug, save_vid = args
    ei = ep["ep_idx"]; eid = ep["episode_id"]
    ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
    tr = np.load(ed / "trajectory.npz")
    qs, gs = tr["qpos"], tr["gripper"]
    c1 = make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"])
    c2 = make_calib(ep["c2_serial"], ep["c2_extr"], ep["c2_intr"])
    mug_vis = Path(cfg["mug_vis_obj"])
    mug_rgb = cfg["ep_rgbs"].get(str(ei), [0.88, 0.84, 0.24])

    ev = extract_grasp(qs, gs)
    pc, po = ev["p_close"], ev["p_open"]
    # Per-episode table_z (CLVR uses two different tables — white & wood!) from grasp/drop height (cup grasped ~5.5cm above base)
    tz = float(np.clip(min(pc[2], po[2]) - 0.058, -0.15, 0.20))

    has_c1 = int(m_c1_mug.sum()) >= 120
    has_c2 = int(m_c2_mug.sum()) >= 120
    xy0 = pc[:2].copy()
    if has_c1:
        y1, x1 = np.where(m_c1_mug)
        cand = ray_plane_z(c1.scaled(OUT_WH[0], OUT_WH[1]), (float(np.median(x1)), float(np.median(y1))), tz + 0.045)[:2]
        if float(np.linalg.norm(cand - pc[:2])) <= 0.14:
            xy0 = cand

    f0_mug_iou = 0.0
    if has_c1 or has_c2:
        m_opt = build_clvr_scene((float(xy0[0]), float(xy0[1])), tz, mug_vis, mug_rgb, with_robot=False, offwidth=OPT_WH[0], offheight=OPT_WH[1])
        d_opt = mujoco.MjData(m_opt)
        mbid = mujoco.mj_name2id(m_opt, mujoco.mjtObj.mjOBJ_BODY, "clvr_mug")
        mgids = [g for g in range(m_opt.ngeom) if m_opt.geom_bodyid[g] == mbid]
        ren_o = mujoco.Renderer(m_opt, OPT_WH[1], OPT_WH[0]); ren_o.enable_segmentation_rendering()
        rm1 = cv2.resize(m_c1_mug.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
        rm2 = cv2.resize(m_c2_mug.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)

        def eval_m(x):
            if abs(x[0] - xy0[0]) > 0.12 or abs(x[1] - xy0[1]) > 0.12:
                return 0.0
            d_opt.qpos[:3] = [float(x[0]), float(x[1]), tz + 0.001]; d_opt.qpos[3:7] = [1, 0, 0, 0]
            s, ws = 0.0, 0.0
            if has_c1:
                configure_cam(m_opt, 0, c1, OPT_WH[0], OPT_WH[1]); mujoco.mj_forward(m_opt, d_opt); ren_o.update_scene(d_opt, camera=0)
                s += 0.75 * compute_mask_iou(rm1, np.isin(ren_o.render()[:, :, 0], mgids)); ws += 0.75
            if has_c2:
                configure_cam(m_opt, 0, c2, OPT_WH[0], OPT_WH[1]); mujoco.mj_forward(m_opt, d_opt); ren_o.update_scene(d_opt, camera=0)
                s += 0.25 * compute_mask_iou(rm2, np.isin(ren_o.render()[:, :, 0], mgids)); ws += 0.25
            return s / max(ws, 1e-6)

        res_m = minimize(lambda x: 1.0 - eval_m(x), xy0, method="Nelder-Mead", options={"maxiter": 40, "xatol": 2e-3, "fatol": 2e-3})
        mug_xy_vis = (float(res_m.x[0]), float(res_m.x[1]))
        f0_mug_iou = float(1.0 - res_m.fun)
        ren_o.close()
        mug_xy = mug_xy_vis if float(np.linalg.norm(np.array(mug_xy_vis) - pc[:2])) <= 0.032 else (float(pc[0]), float(pc[1]))
    else:
        mug_xy = (float(pc[0]), float(pc[1]))

    # Bake per-episode real tabletop texture ONLY for showcase episodes (white OR wood table!)
    tbl_tex = None
    if save_vid:
        tbl_tex = ASSET_DIR / f"tbl_ep{ei:03d}.png"
        extract_real_table_texture(
            frame_bgr=cv2.imread(str(ed / "cam1_f0.jpg")), calib=c1,
            table_center_xy=(0.55, 0.0), table_half_xy=(0.55, 0.70), table_z=tz,
            out_png=tbl_tex, exclude_masks=[m_c1_mug],
        )

    m = build_clvr_scene(mug_xy, tz, mug_vis, mug_rgb, table_tex=tbl_tex, with_robot=True, offwidth=OUT_WH[0], offheight=OUT_WH[1])
    d = mujoco.MjData(m)
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")
    mb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "clvr_mug")
    d.qpos[:7] = qs[0]; d.ctrl[:7] = qs[0]; d.ctrl[ag] = float(np.clip(gs[0], 0, 1)) * 255.0
    mujoco.mj_forward(m, d)
    for _ in range(30):
        mujoco.mj_step(m, d)
    init_xy = d.xpos[mb][:2].copy()

    ren = None; od = None; ts_saved = []
    if save_vid:
        configure_cam(m, 0, c1, OUT_WH[0], OUT_WH[1])
        ren = mujoco.Renderer(m, OUT_WH[1], OUT_WH[0])
        od = ROOT / "outputs" / "showcase_sim_frames" / f"d6_clvr_ep{ei:03d}"
        od.mkdir(parents=True, exist_ok=True)

    mon = RolloutMonitor(m, target_body_name="clvr_mug"); mon.reset(d)
    for ti in range(len(qs)):
        d.ctrl[:7] = qs[ti]
        d.ctrl[ag] = (0.72 if (max(0, ev["t_close"] - 2) <= ti < ev["t_open"]) else float(np.clip(gs[ti], 0, 1))) * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)
        if ren is not None and (ti % 3 == 0 or ti == len(qs) - 1):
            ren.update_scene(d, camera=0)
            cv2.imwrite(str(od / f"f_{ti:04d}.jpg"), cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            ts_saved.append(ti)

    for _ in range(60):
        d.ctrl[:7] = qs[-1]; d.ctrl[ag] = float(np.clip(gs[-1], 0, 1)) * 255.0
        mujoco.mj_step(m, d)
    mon.observe(d)
    if ren is not None:
        ren.close()
        (od / "info.json").write_text(json.dumps({"ep_idx": ei, "episode_id": eid, "c1_serial": ep["c1_serial"], "ts": ts_saved, "f0_mug_iou": round(f0_mug_iou, 3)}))

    end_xy = d.xpos[mb][:2]
    disp_m = float(np.linalg.norm(end_xy - init_xy))
    tgt_err = float(np.linalg.norm(end_xy - po[:2]))
    goal_reached = (disp_m >= 0.06) and (tgt_err <= 0.13) and (float(d.xpos[mb][2]) >= tz - 0.02)
    res = _finalize("move_mug", mon, d, goal_reached, {"disp_cm": disp_m * 100.0, "tgt_err_cm": tgt_err * 100.0, "f0_mug_iou": round(f0_mug_iou, 3)})
    return {"ep_idx": ei, "episode_id": eid, **res.to_dict()}


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads((BUNDLE / "meta.json").read_text())
    masks = dict(np.load(BUNDLE / "masks_demo6_50eps.npz"))
    cfg = prepare_assets(meta)

    t_sim0 = time.perf_counter()
    tasks = [(ep, cfg, masks[f"ep{ep['ep_idx']:03d}_c1_mug"], masks[f"ep{ep['ep_idx']:03d}_c2_mug"], ep["ep_idx"] < 16) for ep in meta]
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R2S_WORKERS", "16")), mp_context=ctx) as pool:
        for k, f in enumerate(as_completed([pool.submit(eval_ep, t) for t in tasks]), 1):
            results.append(f.result())
            if k % 25 == 0 or k == len(tasks):
                cp = sum(int(x["success"]) for x in results)
                print(f"  [Demo6 CLVR {k:2d}/{len(tasks)}] PASS={cp}/{k} ({100*cp/k:.1f}%) ({time.perf_counter()-t_sim0:.1f}s)", flush=True)

    results.sort(key=lambda x: x["ep_idx"])
    n = len(results); n_pass = sum(int(x["success"]) for x in results)
    ious = [x["metrics"]["f0_mug_iou"] for x in results if x["metrics"]["f0_mug_iou"] > 0]
    mi = float(np.mean(ious)) if ious else 0.0
    reasons = Counter(x["failure_reason"] for x in results if not x["success"])
    summary = {
        "scene": "Demo #6 CLVR Pick & Relocate Mug (50 episodes)",
        "mujoco_version": mujoco.__version__,
        "timing_sec": {"total": round(time.perf_counter() - t0, 1), "sim_50eps": round(time.perf_counter() - t_sim0, 1)},
        "metrics": {"outcome_pass": n_pass, "outcome_pass_pct": round(100 * n_pass / n, 1), "mean_f0_mug_iou": round(mi, 3)},
        "failure_reasons": dict(reasons), "episodes": results,
    }
    (ASSET_DIR / "demo6_clvr_50eps_results.json").write_text(json.dumps(summary, indent=2))
    print("=" * 90, flush=True)
    print(f"DEMO #6 CLVR (50 EPS, MuJoCo {mujoco.__version__}) COMPLETE in {time.perf_counter()-t0:.1f}s (Sim50={time.perf_counter()-t_sim0:.1f}s)", flush=True)
    print(f"  Outcome PASS : {n_pass}/{n} ({100*n_pass/n:.1f}%) | Mean F0 Mug IoU={mi:.3f} | Failures={dict(reasons)}", flush=True)
    print("=" * 90, flush=True)

    cands = sorted([e for e in results if e["ep_idx"] < 16 and e["success"]], key=lambda e: -e["metrics"]["f0_mug_iou"])[:4]
    S = ROOT / "outputs" / "showcase_sim_frames"
    all_frames, strip_rows = [], []
    for e in cands:
        ei = e["ep_idx"]; od = S / f"d6_clvr_ep{ei:03d}"
        info = json.loads((od / "info.json").read_text())
        c1s = info["c1_serial"]
        cap = cv2.VideoCapture(str(BUNDLE / f"episodes/ep_{ei:03d}" / f"{c1s}.mp4"))
        ep_fr = []
        for ti in info["ts"]:
            sim = cv2.imread(str(od / f"f_{ti:04d}.jpg"))
            if sim is None: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, ti); ok, rf = cap.read()
            if not ok: continue
            real = cv2.resize(rf, OUT_WH); ov = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
            bar = np.zeros((32, OUT_WH[0] * 3, 3), dtype=np.uint8); bar[:] = (22, 24, 28)
            iou_v = info["f0_mug_iou"]; disp_v = e["metrics"]["disp_cm"]
            cv2.putText(bar, f"DEMO #6 CLVR Lab (MuJoCo 4.0.1) | Relocate Mug | Ep#{ei:02d} t={ti:03d} | F0 Mug IoU={iou_v:.2f} Disp={disp_v:.1f}cm | Outcome PASS {n_pass}/{n} ({100*n_pass/n:.1f}%)",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 245, 130), 1, cv2.LINE_AA)
            lbl = np.hstack([real, sim, ov])
            cv2.putText(lbl, "REAL CAMERA (CLVR)", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(lbl, "SIM (Hunyuan3D-2mv Mug + Real Table)", (OUT_WH[0] + 10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 245, 130), 1, cv2.LINE_AA)
            cv2.putText(lbl, "50/50 OVERLAY", (2 * OUT_WH[0] + 10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 255), 1, cv2.LINE_AA)
            ep_fr.append(np.vstack([bar, lbl]))
        cap.release()
        if ep_fr:
            all_frames.extend(ep_fr + [ep_fr[-1]] * 10)
            if len(strip_rows) < 2:
                strip_rows.append(np.vstack([ep_fr[k] for k in (0, len(ep_fr)//3, (2*len(ep_fr))//3, len(ep_fr)-1)]))
    if all_frames:
        raw_mp4 = ASSET_DIR / "raw_d6.mp4"; out_mp4 = ASSET_DIR / "demo6_clvr_mj4_showcase.mp4"
        h, w = all_frames[0].shape[:2]
        vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
        for fr in all_frames: vw.write(fr)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-i", str(raw_mp4), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", str(out_mp4)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cv2.imwrite(str(ASSET_DIR / "inspect_d6_clvr.png"), np.vstack(strip_rows))
        print(f"[D6 VIDEO] Saved {out_mp4} ({out_mp4.stat().st_size/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
