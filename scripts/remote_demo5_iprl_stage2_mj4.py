"""Demo #5 (IPRL 2024-01-01: Put/Remove Glass Lid on/off Kitchen Pot, 50 episodes) — Stage 2 MuJoCo 4.0.1.

All-on-A100 `MuJoCo 4.0.1` (`spawn` 16w EGL):
  1. Converts Multi-View `Hunyuan3D-2mv` `iprl_pot_hy3d_mv_raw.glb` -> visual mesh + guaranteed-hollow 16-facet pan collision.
  2. Builds translucent glass lid + dark rim + black center knob (`d=3.4cm` for clean Robotiq 2F-85 pinch).
  3. Runs all 50 episodes with clean outcome-only evaluation (`evaluate_object_on_target` / `evaluate_object_out_of_container`).
  4. Composites 3-column H.264 showcase video (`demo5_iprl_mj4_showcase.mp4`) + keyframe audit strip (`inspect_d5_iprl.png`).
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
from r2s_bench.eval.task_success import RolloutMonitor, evaluate_object_on_target, evaluate_object_out_of_container
from r2s_bench.grading.l1_visual import compute_mask_iou
from r2s_bench.robots.franka_robotiq import build_droid_franka_robotiq_spec
from r2s_bench.schema import CameraCalib

BUNDLE = ROOT / "outputs" / "demo5_iprl_bundle"
ASSET_DIR = ROOT / "outputs" / "hy3d_demo5"
SIM_DIR = ASSET_DIR / "sim_ready_pot"
OUT_WH = (480, 270)
OPT_WH = (240, 135)

DEFAULT_INTR = {
    "27432424": {"cameraMatrix": [524.22528, 645.3158, 524.22528, 365.4808], "width": 1280, "height": 720},
    "28221883": {"cameraMatrix": [522.4032, 646.5189, 522.4032, 365.7996], "width": 1280, "height": 720},
}


def make_calib(serial: str, ext6: list, intr_dict: dict | None) -> CameraCalib:
    if not intr_dict or int(intr_dict.get("width", 0)) <= 0:
        intr_dict = DEFAULT_INTR.get(str(serial), DEFAULT_INTR["27432424"])
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
    d_fk.qpos[:7] = qs[0]; d_fk.ctrl[:7] = qs[0]; d_fk.ctrl[ag] = 0.65 * 255.0
    for _ in range(150):
        mujoco.mj_step(m_fk, d_fk)
    q_pinch = d_fk.qpos[7:].copy()
    T = len(qs)
    mids = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        d_fk.qpos[:7] = qs[t]; d_fk.qpos[7:] = q_pinch; mujoco.mj_forward(m_fk, d_fk)
        mids[t] = 0.5 * (d_fk.geom_xpos[lp] + d_fk.geom_xpos[rp])
    n1 = int(T * 0.65)
    g_peak = float(np.max(gs[:n1]))
    thresh = max(0.18, g_peak * 0.55)
    cc = [t for t in range(3, n1) if gs[t] >= thresh and gs[t - 1] < thresh]
    t_close = min(cc, key=lambda t: mids[t, 2]) if cc else int(np.argmax(gs[:n1]))
    t_hold = min(t_close + 4, T - 1)
    oc = [t for t in range(t_hold + 5, T) if gs[t] < thresh * 0.45]
    t_open = oc[0] if oc else T - 1
    return {"t_close": int(t_close), "t_open": int(t_open), "p_close": mids[t_hold], "p_open": mids[t_open], "mids": mids}


def build_iprl_scene(
    pot_xy: Tuple[float, float],
    lid_pos: Tuple[float, float, float],
    table_z: float,
    pot_vis_obj: Path,
    pot_rgb: List[float],
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

    # Clean matte black dining table (matches IPRL black kitchen table without window-glare warp artifacts)
    g_tbl = spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.58, 0.0, table_z - 0.025], size=[0.52, 0.025, 0.0],
        rgba=[0.11, 0.11, 0.12, 1.0],
        friction=[0.85, 0.005, 0.0001],
    )

    # 1) Kitchen Pan / Pot (`Hunyuan3D-2mv` visual + heavy guaranteed-hollow 16-facet rim)
    spec.add_mesh(name="iprl_pot_vis", file=str(pot_vis_obj))
    pot = spec.worldbody.add_body(name="cooking_pot", pos=[pot_xy[0], pot_xy[1], table_z + 0.001])
    pot.add_freejoint(name="cooking_pot_jnt")
    pot.add_geom(
        name="pot_vis_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="iprl_pot_vis",
        rgba=[pot_rgb[0], pot_rgb[1], pot_rgb[2], 1.0], contype=0, conaffinity=0, mass=0.0,
    )
    pot.add_geom(name="pot_bot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.005], size=[0.095, 0.005, 0], rgba=[0, 0, 0, 0], mass=0.65, friction=[1.1, 0.01, 0.001])
    r_p, h_p = 0.096, 0.056
    for k in range(16):
        a = 2.0 * math.pi * k / 16
        pot.add_geom(
            name=f"pot_w{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[r_p * math.cos(a), r_p * math.sin(a), 0.5 * h_p],
            quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)],
            size=[0.004, 0.021, 0.5 * h_p], rgba=[0, 0, 0, 0], mass=0.015, solref=[0.022, 1.4],
        )

    # 2) Glass Pot Lid (translucent glass disk + dark steel rim + black graspable center knob)
    if with_robot:
        lid = spec.worldbody.add_body(name="pot_lid", pos=list(lid_pos))
        lid.add_freejoint(name="pot_lid_jnt")
        # Glass dome body
        lid.add_geom(
            name="lid_glass", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0, 0, 0.006], size=[0.098, 0.006, 0],
            rgba=[0.72, 0.78, 0.82, 0.48], mass=0.095, friction=[0.7, 0.005, 0.001], solref=[0.024, 1.5],
        )
        # Visual stainless-steel rim ring
        lid.add_geom(
            name="lid_rim", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0, 0, 0.005], size=[0.100, 0.004, 0],
            rgba=[0.22, 0.23, 0.25, 0.85], contype=0, conaffinity=0, mass=0.0,
        )
        # Raised black center knob (d=3.6cm, h=3.4cm — right in Robotiq 2F-85 sweet spot!)
        lid.add_geom(
            name="lid_knob", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0, 0, 0.028], size=[0.018, 0.018, 0],
            rgba=[0.10, 0.10, 0.11, 1.0], mass=0.030, friction=[2.5, 0.04, 0.004], solref=[0.022, 1.5],
        )

    model = spec.compile()
    if with_robot:
        for _aid in range(min(7, model.nu)):
            model.actuator_gainprm[_aid, 0] *= 2.2
            model.actuator_biasprm[_aid, 1] *= 2.2
            model.actuator_biasprm[_aid, 2] *= 1.4832
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "lid_knob" in nm or "pad" in nm:
                model.geom_condim[g] = 4
    return model


def prepare_assets(meta: list, masks: dict) -> dict:
    colors = json.loads((ASSET_DIR / "demo5_colors.json").read_text())
    ref_ei = int(colors["ref_ep_idx"])
    pot_rgb = colors["pot_rgb"]

    meta_json = SIM_DIR / "asset_meta.json"
    if not meta_json.exists():
        asset = build_sim_ready_asset(
            name="iprl_pot", raw_mesh_path=ASSET_DIR / "iprl_pot_hy3d_mv_raw.glb",
            out_dir=SIM_DIR, target_size_xyz=(0.205, 0.205, 0.062),
            rgba=(pot_rgb[0], pot_rgb[1], pot_rgb[2], 1.0),
            is_container=True, auto_upright_pca=False, wall_thickness_m=0.006, max_convex_hulls=16,
        )
        pot_vis_obj = str(asset.visual_mesh_path)
    else:
        pot_vis_obj = str(json.loads(meta_json.read_text())["visual_mesh_path"])

    # Measure exact round black kitchen table Z from "put" episodes (lid resting directly on table -> p_close_z - 0.032)
    tzs = []
    for ep in meta[:25]:
        ei = ep["ep_idx"]
        tr = np.load(BUNDLE / f"episodes/ep_{ei:03d}/trajectory.npz")
        ev = extract_grasp(tr["qpos"], tr["gripper"])
        ins = ep["instruction"].lower()
        if "remove" in ins or "take" in ins:
            tzs.append(float(ev["p_open"][2] - 0.030))
        else:
            tzs.append(float(ev["p_close"][2] - 0.030))
    table_z = float(np.median(tzs))
    print(f"[D5 IPRL] Kitchen table_z = {table_z*100:.2f} cm (from {len(tzs)} grasps/drops)", flush=True)

    ep_ref = meta[ref_ei]
    c1_ref = make_calib(ep_ref["c1_serial"], ep_ref["c1_extr"], ep_ref["c1_intr"])
    ref_img = cv2.imread(str(BUNDLE / f"episodes/ep_{ref_ei:03d}/cam1_f0.jpg"))
    tbl_tex = ASSET_DIR / "demo5_iprl_real_table.png"
    extract_real_table_texture(
        frame_bgr=ref_img, calib=c1_ref,
        table_center_xy=(0.58, 0.0), table_half_xy=(0.58, 0.58), table_z=table_z,
        out_png=tbl_tex,
        exclude_masks=[masks[f"ep{ref_ei:03d}_c1_pot"], masks[f"ep{ref_ei:03d}_c1_lid"]],
    )
    return {"ref_ep_idx": ref_ei, "pot_rgb": pot_rgb, "table_z": table_z, "pot_vis_obj": pot_vis_obj, "table_tex": str(tbl_tex)}


def eval_ep(args: tuple) -> dict:
    ep, cfg, m_c1_pot, m_c2_pot, m_c1_lid, save_vid = args
    ei = ep["ep_idx"]
    eid = ep["episode_id"]
    ins = ep["instruction"].lower()
    is_remove = ("remove" in ins) or ("take" in ins)

    ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
    tr = np.load(ed / "trajectory.npz")
    qs, gs = tr["qpos"], tr["gripper"]
    c1 = make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"])
    c2 = make_calib(ep["c2_serial"], ep["c2_extr"], ep["c2_intr"])
    tz = float(cfg["table_z"])
    pot_vis = Path(cfg["pot_vis_obj"])
    pot_rgb = cfg["pot_rgb"]
    tbl_tex = Path(cfg["table_tex"])

    ev = extract_grasp(qs, gs)
    pc, po = ev["p_close"], ev["p_open"]

    # Localize pot via Dual-View silhouette IoU around expected pot region (p_close for remove, p_open for put)
    pot_anchor = pc[:2] if is_remove else po[:2]
    has_c1 = int(m_c1_pot.sum()) >= 200
    has_c2 = int(m_c2_pot.sum()) >= 200
    xy0 = pot_anchor.copy()
    if has_c1:
        y1, x1 = np.where(m_c1_pot)
        cand = ray_plane_z(c1.scaled(OUT_WH[0], OUT_WH[1]), (float(np.median(x1)), float(np.median(y1))), tz + 0.030)[:2]
        if float(np.linalg.norm(cand - pot_anchor)) <= 0.16:
            xy0 = cand

    f0_pot_iou = 0.0
    if has_c1 or has_c2:
        m_opt = build_iprl_scene((float(xy0[0]), float(xy0[1])), (0, 0, -5), tz, pot_vis, pot_rgb, with_robot=False, offwidth=OPT_WH[0], offheight=OPT_WH[1])
        d_opt = mujoco.MjData(m_opt)
        pbid = mujoco.mj_name2id(m_opt, mujoco.mjtObj.mjOBJ_BODY, "cooking_pot")
        pgids = [g for g in range(m_opt.ngeom) if m_opt.geom_bodyid[g] == pbid]
        ren_o = mujoco.Renderer(m_opt, OPT_WH[1], OPT_WH[0]); ren_o.enable_segmentation_rendering()
        rm1 = cv2.resize(m_c1_pot.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
        rm2 = cv2.resize(m_c2_pot.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)

        def eval_p(x):
            if abs(x[0] - xy0[0]) > 0.12 or abs(x[1] - xy0[1]) > 0.12:
                return 0.0
            d_opt.qpos[:3] = [float(x[0]), float(x[1]), tz + 0.001]; d_opt.qpos[3:7] = [1, 0, 0, 0]
            s, ws = 0.0, 0.0
            if has_c1:
                configure_cam(m_opt, 0, c1, OPT_WH[0], OPT_WH[1]); mujoco.mj_forward(m_opt, d_opt); ren_o.update_scene(d_opt, camera=0)
                s += 0.75 * compute_mask_iou(rm1, np.isin(ren_o.render()[:, :, 0], pgids)); ws += 0.75
            if has_c2:
                configure_cam(m_opt, 0, c2, OPT_WH[0], OPT_WH[1]); mujoco.mj_forward(m_opt, d_opt); ren_o.update_scene(d_opt, camera=0)
                s += 0.25 * compute_mask_iou(rm2, np.isin(ren_o.render()[:, :, 0], pgids)); ws += 0.25
            return s / max(ws, 1e-6)

        res_p = minimize(lambda x: 1.0 - eval_p(x), xy0, method="Nelder-Mead", options={"maxiter": 40, "xatol": 2e-3, "fatol": 2e-3})
        pot_xy_vis = (float(res_p.x[0]), float(res_p.x[1]))
        f0_pot_iou = float(1.0 - res_p.fun)
        ren_o.close()
        pot_xy = pot_xy_vis if float(np.linalg.norm(np.array(pot_xy_vis) - pot_anchor)) <= 0.06 else (float(pot_anchor[0]), float(pot_anchor[1]))
    else:
        pot_xy = (float(pot_anchor[0]), float(pot_anchor[1]))

    # Spawn lid: if "remove", resting on top of pot; if "put", resting on kitchen table at p_close XY
    if is_remove:
        lid_pos = (float(pc[0]), float(pc[1]), tz + 0.060)
    else:
        lid_pos = (float(pc[0]), float(pc[1]), tz + 0.002)

    m = build_iprl_scene(pot_xy, lid_pos, tz, pot_vis, pot_rgb, table_tex=tbl_tex, with_robot=True, offwidth=OUT_WH[0], offheight=OUT_WH[1])
    d = mujoco.MjData(m)
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")
    d.qpos[:7] = qs[0]; d.ctrl[:7] = qs[0]; d.ctrl[ag] = float(np.clip(gs[0], 0, 1)) * 255.0
    mujoco.mj_forward(m, d)
    for _ in range(30):
        mujoco.mj_step(m, d)

    ren = None; od = None; ts_saved = []
    if save_vid:
        configure_cam(m, 0, c1, OUT_WH[0], OUT_WH[1])
        ren = mujoco.Renderer(m, OUT_WH[1], OUT_WH[0])
        od = ROOT / "outputs" / "showcase_sim_frames" / f"d5_iprl_ep{ei:03d}"
        od.mkdir(parents=True, exist_ok=True)

    mon = RolloutMonitor(m, target_body_name="pot_lid"); mon.reset(d)
    for ti in range(len(qs)):
        d.ctrl[:7] = qs[min(ti + 1, len(qs) - 1)]
        d.ctrl[ag] = (0.85 if (max(0, ev["t_close"] - 2) <= ti < ev["t_open"]) else float(np.clip(gs[ti], 0, 1))) * 255.0
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
        (od / "info.json").write_text(json.dumps({"ep_idx": ei, "episode_id": eid, "c1_serial": ep["c1_serial"], "ts": ts_saved, "f0_pot_iou": round(f0_pot_iou, 3), "is_remove": is_remove}))

    if is_remove:
        res = evaluate_object_out_of_container(m, d, mon, "pot_lid", "cooking_pot", min_xy_dist_m=0.11, task_name="lid_off_pot")
    else:
        res = evaluate_object_on_target(m, d, mon, "pot_lid", "cooking_pot", max_xy_dist_m=0.10, z_range_rel=(0.01, 0.14), task_name="lid_on_pot")
    out = res.to_dict()
    out["metrics"]["f0_pot_iou"] = round(f0_pot_iou, 3)
    out["is_remove"] = is_remove
    return {"ep_idx": ei, "episode_id": eid, **out}


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads((BUNDLE / "meta.json").read_text())
    masks = dict(np.load(BUNDLE / "masks_demo5_50eps.npz"))
    cfg = prepare_assets(meta, masks)

    t_sim0 = time.perf_counter()
    tasks = [(ep, cfg, masks[f"ep{ep['ep_idx']:03d}_c1_pot"], masks[f"ep{ep['ep_idx']:03d}_c2_pot"], masks[f"ep{ep['ep_idx']:03d}_c1_lid"], ep["ep_idx"] < 16) for ep in meta]
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R2S_WORKERS", "16")), mp_context=ctx) as pool:
        for k, f in enumerate(as_completed([pool.submit(eval_ep, t) for t in tasks]), 1):
            results.append(f.result())
            if k % 25 == 0 or k == len(tasks):
                cp = sum(int(x["success"]) for x in results)
                print(f"  [Demo5 IPRL {k:2d}/{len(tasks)}] PASS={cp}/{k} ({100*cp/k:.1f}%) ({time.perf_counter()-t_sim0:.1f}s)", flush=True)

    results.sort(key=lambda x: x["ep_idx"])
    n = len(results); n_pass = sum(int(x["success"]) for x in results)
    ious = [x["metrics"]["f0_pot_iou"] for x in results if x["metrics"]["f0_pot_iou"] > 0]
    mi = float(np.mean(ious)) if ious else 0.0
    reasons = Counter(x["failure_reason"] for x in results if not x["success"])
    summary = {
        "scene": "Demo #5 IPRL Glass Lid on/off Kitchen Pot (50 episodes)",
        "mujoco_version": mujoco.__version__,
        "timing_sec": {"total": round(time.perf_counter() - t0, 1), "sim_50eps": round(time.perf_counter() - t_sim0, 1)},
        "metrics": {"outcome_pass": n_pass, "outcome_pass_pct": round(100 * n_pass / n, 1), "mean_f0_pot_iou": round(mi, 3)},
        "failure_reasons": dict(reasons), "episodes": results,
    }
    (ASSET_DIR / "demo5_iprl_50eps_results.json").write_text(json.dumps(summary, indent=2))
    print("=" * 90, flush=True)
    print(f"DEMO #5 IPRL (50 EPS, MuJoCo {mujoco.__version__}) COMPLETE in {time.perf_counter()-t0:.1f}s (Sim50={time.perf_counter()-t_sim0:.1f}s)", flush=True)
    print(f"  Outcome PASS : {n_pass}/{n} ({100*n_pass/n:.1f}%) | Mean F0 Pot IoU={mi:.3f} | Failures={dict(reasons)}", flush=True)
    print("=" * 90, flush=True)

    # Composite showcase video + audit strip on A100 directly!
    cands = sorted([e for e in results if e["ep_idx"] < 16 and e["success"]], key=lambda e: -e["metrics"]["f0_pot_iou"])[:4]
    S = ROOT / "outputs" / "showcase_sim_frames"
    all_frames, strip_rows = [], []
    for e in cands:
        ei = e["ep_idx"]; od = S / f"d5_iprl_ep{ei:03d}"
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
            iou_v = info["f0_pot_iou"]; act = "Lid OFF Pot" if info["is_remove"] else "Lid ON Pot"
            cv2.putText(bar, f"DEMO #5 IPRL Kitchen (MuJoCo 4.0.1) | {act} | Ep#{ei:02d} t={ti:03d} | F0 Pot IoU={iou_v:.2f} | Outcome PASS {n_pass}/{n} ({100*n_pass/n:.1f}%)",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 245, 130), 1, cv2.LINE_AA)
            lbl = np.hstack([real, sim, ov])
            cv2.putText(lbl, "REAL CAMERA (IPRL)", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(lbl, "SIM (Hunyuan3D-2mv Pot + Glass Lid)", (OUT_WH[0] + 10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 245, 130), 1, cv2.LINE_AA)
            cv2.putText(lbl, "50/50 OVERLAY", (2 * OUT_WH[0] + 10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 255), 1, cv2.LINE_AA)
            ep_fr.append(np.vstack([bar, lbl]))
        cap.release()
        if ep_fr:
            all_frames.extend(ep_fr + [ep_fr[-1]] * 10)
            if len(strip_rows) < 2:
                strip_rows.append(np.vstack([ep_fr[k] for k in (0, len(ep_fr)//3, (2*len(ep_fr))//3, len(ep_fr)-1)]))
    if all_frames:
        raw_mp4 = ASSET_DIR / "raw_d5.mp4"; out_mp4 = ASSET_DIR / "demo5_iprl_mj4_showcase.mp4"
        h, w = all_frames[0].shape[:2]
        vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
        for fr in all_frames: vw.write(fr)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-i", str(raw_mp4), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", str(out_mp4)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cv2.imwrite(str(ASSET_DIR / "inspect_d5_iprl.png"), np.vstack(strip_rows))
        print(f"[D5 VIDEO] Saved {out_mp4} ({out_mp4.stat().st_size/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
