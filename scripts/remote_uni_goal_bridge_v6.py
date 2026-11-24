"""Uni-Goal-Bridge v6: Full 17-Task x 11-Lab DROID Grand Unified Action Render (27.7M, OBS=47 Proven SE(2)).

Builds directly on our scientifically verified `v4` (`OBS=47`, yaw-invariant `t_ee` + object nematic `(cos 2phi, sin 2phi)`
+ 2-step position DLS-IK) after ablating `v5` (which proved raw `R_ee[:, 0]` causes wrist-yaw causal confusion, R2S-12):
  - Extends from 11 tasks (448 real demos) -> 17 tasks across 11 robotics labs (adding ILIAD #10, REAL #11,
    WEIRD #12, AUTOLab block->cup, TRI marker->cup, PennPAL plush toy->plate -> ~560+ verified real demos!).
  - Scales DiT capacity cleanly: `d_model=384, n_layers=8, n_heads=8` (~27.7M params, pure TF32 + QK-Norm + SDPA).
  - Supports BOTH:
      1. `v6_real` (100% pure real 17-task x 11-lab scaling)
      2. `harvest_v4` (crisp 2-step DLS-IK OOD trajectory harvesting using 78.4% `v4` across Tasks 0..10)
      3. `v6_mix` (85% Real 17-Task + 15% Top-50% Ultra-Crisp `v4`-Harvested OOD trajectories)
  - Evaluates apples-to-apples on `orig5` (125 poses, seed=2026), `new6` (150 poses, seed=2027),
    `new6_randyaw` (150 poses, 360-deg random marker yaw), and `brandnew6` (Demo #17-#22, 150 poses, seed=2028)!
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

import scripts.remote_uni_goal_bridge_scaling as uni
import scripts.remote_uni_goal_bridge_v4 as v4

V4_DIR = v4.OUT_DIR
OUT_DIR = ROOT / "outputs" / "uni_goal_bridge_v6"
HORIZON = uni.HORIZON
ACT_DIM = uni.ACT_DIM
OBS_DIM = v4.OBS_DIM  # 47 (proven yaw-invariant t_ee + nematic object orientation!)
BRIDGE_SIGMA = uni.BRIDGE_SIGMA
ODE_STEPS = uni.ODE_STEPS
PORT_V6 = 18972

BRANDNEW_SCENES = [
    (11, "demo17_iliad_marker_cup",  "12. ILIAD | Marker -> Black Cup",       "configs/scenes/demo17_iliad_marker_cup.json"),
    (12, "demo18_real_marker_mug",   "13. REAL | Marker -> Mug",              "configs/scenes/demo18_real_marker_mug.json"),
    (13, "demo19_weird_obj_bowl",    "14. WEIRD | Object -> Bowl/Cup",        "configs/scenes/demo19_weird_obj_bowl.json"),
    (14, "demo20_auto_block_cup",    "15. AUTOLab | Yellow Block -> Blue Cup","configs/scenes/demo20_auto_block_cup.json"),
    (15, "demo21_tri_marker_cup",    "16. TRI | Marker -> Paper Cup",         "configs/scenes/demo21_tri_marker_cup.json"),
    (16, "demo22_pennpal_toy_plate", "17. PennPAL | Plush Toy -> Plate",      "configs/scenes/demo22_pennpal_toy_plate.json"),
]


def build_all_task_specs_v6() -> List[dict]:
    specs = v4.build_all_task_specs_v4()
    for tid, name, label, cfg_rel in BRANDNEW_SCENES:
        cfg_p = ROOT / cfg_rel
        if not cfg_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text())
        src_o, tgt_o = cfg["objects"][0], cfg["objects"][1]
        s_xyz = [float(v) for v in src_o["size_xyz"]]
        t_xyz = [float(v) for v in tgt_o["size_xyz"]]
        is_hol = 1.0 if src_o.get("grasp_mode") == "rim_pinch" else 0.0
        pick_z = float(src_o.get("grasp_height_m", 0.5 * s_xyz[2]))
        place_z = float(t_xyz[2] + 0.032) if cfg["outcome"]["type"] == "in_container" else float(t_xyz[2] + 0.025)
        aniso = float(np.clip((max(s_xyz[0], s_xyz[1]) - min(s_xyz[0], s_xyz[1])) / max(s_xyz[0], s_xyz[1], 1e-4), 0.0, 1.0))
        specs.append({
            "tid": tid, "name": name, "label": label, "kind": "universal", "cfg": cfg_rel,
            "src_body": src_o["name"], "tgt_body": tgt_o["name"],
            "src_size": s_xyz, "tgt_size": t_xyz, "is_hollow": is_hol, "aniso": round(aniso, 4),
            "pick_z_off": round(pick_z, 4), "place_z_off": round(place_z, 4),
        })
    return specs


# ---------------------------------------------------------------------------
# Stage 1: Collect Brand-New Tasks 11..16 & Merge with Clean Cached Tasks 0..10
# ---------------------------------------------------------------------------
def stage_collect_v6(min_demos: int = 5) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    base_ds = dict(np.load(V4_DIR / "uni_real_v4_11tasks.npz"))
    base_meta = json.loads((V4_DIR / "task_meta_v4.json").read_text())
    parts = [base_ds]
    task_meta = dict(base_meta["tasks"])
    active_tids = list(base_meta["active_tids"])

    all_specs = build_all_task_specs_v6()
    ctx = mp.get_context("spawn")
    n_w = int(os.environ.get("R2S_WORKERS", "24"))

    for tspec in all_specs[11:]:
        tid = tspec["tid"]
        cfg = json.loads((ROOT / tspec["cfg"]).read_text())
        sid = cfg["scene_id"]; bundle = Path(cfg["bundle_dir"]); asset_dir = Path(cfg["asset_dir"])
        res_p = asset_dir / f"{sid}_results.json"
        if not res_p.exists():
            continue
        res_prev = json.loads(res_p.read_text())
        pass_ids = [e["ep_idx"] for e in res_prev["episodes"] if e["success"]]
        if len(pass_ids) < min_demos:
            print(f"[Collect v6 Task {tid:2d} {sid}] SKIP (only {len(pass_ids)} < {min_demos})", flush=True)
            continue

        meta = json.loads((bundle / "meta.json").read_text())
        masks = dict(np.load(bundle / "masks.npz"))
        s1_colors = json.loads((asset_dir / "stage1_colors.json").read_text())
        fb_c1 = next(m["c1_intr"] for m in meta if m.get("c1_intr") and int(m["c1_intr"].get("width", 0)) > 0)
        fb_c2 = next(m["c2_intr"] for m in meta if m.get("c2_intr") and int(m["c2_intr"].get("width", 0)) > 0)
        fallback_intr = {"c1": fb_c1, "c2": fb_c2}
        vis_paths = {}
        for o in cfg["objects"]:
            mj = asset_dir / f"sim_ready_{o['name']}" / "asset_meta.json"
            if mj.exists():
                vis_paths[o["name"]] = Path(json.loads(mj.read_text())["visual_mesh_path"])

        meta_by_id = {m["ep_idx"]: m for m in meta}
        tasks = [
            (tspec, meta_by_id[ei], cfg, vis_paths, s1_colors, fallback_intr,
             {f"{cam}_{o['name']}": masks[f"ep{ei:03d}_{cam}_{o['name']}"] for o in cfg["objects"] for cam in ("c1", "c2")})
            for ei in pass_ids
        ]

        demos = []
        with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
            for f in as_completed([pool.submit(v4._replay_universal_ep_v4, t) for t in tasks]):
                r = f.result()
                if r is not None:
                    demos.append(r)

        if len(demos) < min_demos:
            print(f"[Collect v6 Task {tid:2d} {sid}] SKIP (replayed {len(demos)} < {min_demos})", flush=True)
            continue

        demos.sort(key=lambda x: x["ep_idx"])
        part = v4._demos_to_transitions_v4(demos, tspec)
        parts.append(part)
        active_tids.append(tid)

        all_xy = np.vstack([dm["src_xy0"] for dm in demos] + [dm["tgt_xy0"] for dm in demos])
        x_lo = float(np.clip(np.percentile(all_xy[:, 0], 5) - 0.06, 0.36, 0.50))
        x_hi = float(np.clip(np.percentile(all_xy[:, 0], 95) + 0.06, 0.58, 0.72))
        y_lo = float(np.clip(np.percentile(all_xy[:, 1], 5) - 0.08, -0.27, -0.08))
        y_hi = float(np.clip(np.percentile(all_xy[:, 1], 95) + 0.08, 0.08, 0.27))
        tz_med = float(np.median([dm["table_z"] for dm in demos]))
        q_home = np.median(np.stack([dm["q_home"] for dm in demos], axis=0), axis=0)
        min_sep = max(0.14, 0.5 * (max(tspec["src_size"][:2]) + max(tspec["tgt_size"][:2])) + 0.045)
        task_meta[str(tid)] = {
            "name": sid, "n_real_demos": len(demos), "n_samples": len(part["act"]),
            "q_home": q_home.tolist(), "table_z_med": tz_med,
            "xy_bounds": [round(x_lo, 3), round(x_hi, 3), round(y_lo, 3), round(y_hi, 3)],
            "dist_range": [round(min_sep, 3), round(min(0.36, min_sep + 0.20), 3)],
            "ref_ep_idx": int(demos[0]["ep_idx"]), "aniso": tspec["aniso"],
        }
        print(f"[Collect v6 Task {tid:2d} {sid}] {len(demos)}/{len(pass_ids)} demos -> {len(part['act'])} chunks | aniso={tspec['aniso']:.2f}", flush=True)

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}
    np.savez_compressed(OUT_DIR / "uni_real_v6_17tasks.npz", **merged)
    (OUT_DIR / "task_meta_v6.json").write_text(json.dumps({"active_tids": active_tids, "tasks": task_meta}, indent=2))
    tot = sum(task_meta[str(t)]["n_real_demos"] for t in active_tids)
    print(f"[Collect v6 COMPLETE] {len(active_tids)} active tasks | {tot} verified real demos -> {len(merged['act'])} chunks in {time.perf_counter()-t0:.1f}s!", flush=True)


# ---------------------------------------------------------------------------
# Stage 2: Harvest 1,650 Ultra-Crisp OOD Trajectories from 78.4% v4!
# ---------------------------------------------------------------------------
def _run_one_v4_harvest_ep(args: tuple) -> dict:
    import mujoco
    from r2s_bench.eval.task_success import RolloutMonitor, evaluate_object_in_container, evaluate_object_on_target

    tspec, ep_id, src_xy, tgt_xy, src_yaw, tz, q_home, vlm_ec, vlm_eb, seed, port = args
    if tspec["kind"] == "demo1":
        m, _, oc = uni._build_task_scene_and_cam(tspec, src_xy, tgt_xy, tz, render_frames=False)
    else:
        from scripts.remote_universal_stage2_mj4 import build_universal_scene
        cfg = json.loads((ROOT / tspec["cfg"]).read_text())
        asset_dir = Path(cfg["asset_dir"])
        s1_colors = json.loads((asset_dir / "stage1_colors.json").read_text())
        vis_paths, obj_rgbs = {}, {}
        for o in cfg["objects"]:
            nm = o["name"]
            mj = asset_dir / f"sim_ready_{nm}" / "asset_meta.json"
            if mj.exists(): vis_paths[nm] = Path(json.loads(mj.read_text())["visual_mesh_path"])
            obj_rgbs[nm] = s1_colors["med_rgbs"].get(nm, o.get("default_rgb", [0.6, 0.6, 0.6]))
        m = build_universal_scene(
            cfg, {tspec["src_body"]: tuple(src_xy), tspec["tgt_body"]: tuple(tgt_xy)},
            {tspec["src_body"]: float(src_yaw), tspec["tgt_body"]: 0.0},
            obj_rgbs, vis_paths, tz, table_tex=None, with_robot=True, offwidth=64, offheight=64,
        )
        oc = cfg["outcome"]

    d = mujoco.MjData(m)
    lp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_left_silicone_pad")
    rp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_right_silicone_pad")
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base_mount")
    s_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tspec["src_body"])
    t_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tspec["tgt_body"])
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")
    q_lo, q_hi = m.jnt_range[:7, 0].copy(), m.jnt_range[:7, 1].copy()
    q_ref = np.asarray(q_home, dtype=np.float64)
    gv = uni.task_geom_vec(tspec)

    d.qpos[:7] = q_ref; d.ctrl[:7] = q_ref; d.ctrl[ag] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(35): mujoco.mj_step(m, d)

    p_src0 = d.xpos[s_bid].copy(); p_tgt0 = d.xpos[t_bid].copy()
    p_pick_vlm = np.array([p_src0[0], p_src0[1], tz + tspec["pick_z_off"]], dtype=np.float32) + vlm_ec
    p_place_vlm = np.array([p_tgt0[0], p_tgt0[1], tz + tspec["place_z_off"]], dtype=np.float32) + vlm_eb

    mon = RolloutMonitor(m, target_body_name=tspec["src_body"]); mon.reset(d)
    client = v4.V4BridgeClient(port=port)

    max_ticks = 200; exec_k = 6
    q_cmd = q_ref.copy(); p_ee_prev = 0.5 * (d.xpos[lp] + d.xpos[rp]).copy()
    chunk_p_targets = None; chunk_R_targets = None; chunk_g_targets = None; chunk_idx = exec_k
    success_hold = 0
    rec_p_ee, rec_R_ee, rec_g_st, rec_g_cmd, rec_p_src, rec_p_tgt, rec_u_src = [], [], [], [], [], [], []

    for tick in range(max_ticks):
        p_ee = 0.5 * (d.xpos[lp] + d.xpos[rp]).copy(); R_ee = d.xmat[hb].reshape(3, 3).copy()
        g_st = uni.pad_dist_to_g_state(float(np.linalg.norm(d.xpos[lp] - d.xpos[rp])))
        p_src = d.xpos[s_bid].copy(); p_tgt = d.xpos[t_bid].copy()
        ee_vel = (p_ee - p_ee_prev) / uni.DT_CTRL; p_ee_prev = p_ee.copy()
        y_ax = d.xmat[s_bid].reshape(3, 3)[:2, 1]; yn = float(np.linalg.norm(y_ax))
        u_src = (y_ax / yn).astype(np.float32) if yn > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)

        if chunk_idx >= exec_k or chunk_p_targets is None:
            obs = v4.build_obs_v4_np(p_ee, R_ee, g_st, ee_vel, p_src, p_tgt, u_src, p_pick_vlm, p_place_vlm, tz, gv)
            a_line = uni.make_straight_line_prior_np(p_ee, p_pick_vlm, p_place_vlm, g_st)
            act_chunk = client.query(obs, a_line, seed=seed + tick * 7)
            chunk_p_targets = p_ee[None, :] + act_chunk[:, :3]
            chunk_R_targets = [Rotation.from_rotvec(act_chunk[k, 3:6]).as_matrix() @ R_ee for k in range(HORIZON)]
            chunk_g_targets = np.clip(act_chunk[:, 6], 0.0, 1.0)
            chunk_idx = 0

        p_des = chunk_p_targets[chunk_idx]; R_des = chunk_R_targets[chunk_idx]; g_des = float(chunk_g_targets[chunk_idx])
        chunk_idx += 1

        rec_p_ee.append(p_ee.astype(np.float32)); rec_R_ee.append(R_ee.astype(np.float32))
        rec_g_st.append(np.float32(g_st)); rec_g_cmd.append(np.float32(g_des))
        rec_p_src.append(p_src.astype(np.float32)); rec_p_tgt.append(p_tgt.astype(np.float32)); rec_u_src.append(u_src)

        for _ in range(2):
            p_cur = 0.5 * (d.xpos[lp] + d.xpos[rp]); R_cur = d.xmat[hb].reshape(3, 3)
            ep_err = p_des - p_cur; en = float(np.linalg.norm(ep_err))
            if en > 0.045: ep_err = ep_err * (0.045 / en)
            er = Rotation.from_matrix(R_des @ R_cur.T).as_rotvec(); rn = float(np.linalg.norm(er))
            if rn > 0.35: er = er * (0.35 / rn)
            jp_l = np.zeros((3, m.nv)); jp_r = np.zeros((3, m.nv)); jr_h = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp_l, None, lp); mujoco.mj_jacBody(m, d, jp_r, None, rp); mujoco.mj_jacBody(m, d, None, jr_h, hb)
            J = np.vstack([0.5 * (jp_l[:, :7] + jp_r[:, :7]), jr_h[:, :7]])
            err = np.concatenate([ep_err, er])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.0025 * np.eye(6), err)
            dq_null = 0.02 * (q_ref - d.qpos[:7])
            dq += (np.eye(7) - J.T @ np.linalg.solve(J @ J.T + 0.0025 * np.eye(6), J)) @ dq_null
            q_cmd = np.clip(d.qpos[:7] + np.clip(dq, -0.18, 0.18), q_lo + 0.02, q_hi - 0.02)

        d.ctrl[:7] = q_cmd; d.ctrl[ag] = g_des * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0: mon.observe(d)

        if g_st < 0.25 and tick > 45:
            dxy = float(np.linalg.norm(d.xpos[s_bid][:2] - d.xpos[t_bid][:2]))
            if dxy <= float(oc.get("max_xy_m", 0.095)) and float(p_ee[2] - tz) > 0.14:
                success_hold += 1
                if success_hold >= 12: break
            else:
                success_hold = 0

    for _ in range(45): d.ctrl[:7] = q_cmd; d.ctrl[ag] = 0.0; mujoco.mj_step(m, d)
    mon.observe(d); client.close()

    if oc["type"] == "in_container":
        res = evaluate_object_in_container(m, d, mon, tspec["src_body"], tspec["tgt_body"], max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [-0.02, 0.12]), table_z=tz)
    else:
        res = evaluate_object_on_target(m, d, mon, tspec["src_body"], tspec["tgt_body"], max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [0.005, 0.14]))

    out = {"tid": tspec["tid"], "success": bool(res.success), "ticks": len(rec_p_ee)}
    if res.success and len(rec_p_ee) >= 35:
        out["traj"] = {
            "tid": tspec["tid"], "table_z": tz,
            "p_ee": np.stack(rec_p_ee), "R_ee": np.stack(rec_R_ee),
            "g_state": np.asarray(rec_g_st), "g_cmd": np.asarray(rec_g_cmd),
            "p_src": np.stack(rec_p_src), "p_tgt": np.stack(rec_p_tgt), "u_src": np.stack(rec_u_src),
        }
    return out


def stage_harvest_v4(n_per_task: int = 160, port: int = 18964) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_w = int(os.environ.get("R2S_WORKERS", "24"))
    all_specs = v4.build_all_task_specs_v4()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((V4_DIR / "task_meta_v4.json").read_text())
    active_tids = meta_obj["active_tids"]

    rng = np.random.default_rng(88888)
    tasks = []
    for t in active_tids:
        ts = spec_by_tid[t]; tm = meta_obj["tasks"][str(t)]
        x0, x1, y0, y1 = tm["xy_bounds"]; dmin, dmax = tm["dist_range"]
        tz = float(tm["table_z_med"]); qh = np.array(tm["q_home"], dtype=np.float32)
        for i in range(n_per_task):
            for _ in range(200):
                sxy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                txy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                if dmin <= math.hypot(sxy[0] - txy[0], sxy[1] - txy[1]) <= dmax: break
            ec = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
            eb = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
            syaw = float(rng.uniform(-math.pi, math.pi)) if ts["aniso"] > 0.2 else 0.0
            tasks.append((ts, i, sxy, txy, syaw, tz, qh, ec, eb, 700000 + t * 10000 + i * 31, port))

    print(f"[v4 Crisp Harvest] Launching {len(tasks)} OOD rollouts ({n_per_task}/task x 11 tasks)...", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
        for k, f in enumerate(as_completed([pool.submit(_run_one_v4_harvest_ep, x) for x in tasks]), 1):
            results.append(f.result())
            if k % 350 == 0 or k == len(tasks):
                ok_n = sum(int(r["success"]) for r in results)
                print(f"  [v4 Harvest {k:4d}/{len(tasks)}] PASS={ok_n} ({100*ok_n/k:.1f}%) ({time.perf_counter()-t0:.1f}s)", flush=True)

    v4._shutdown_v4(port)

    parts, cnts = [], {}
    for t in active_tids:
        ts = spec_by_tid[t]
        ok_rs = [r for r in results if r["tid"] == t and r["success"] and "traj" in r]
        cnts[ts["name"]] = len(ok_rs)
        for r in ok_rs:
            p = v4._demos_to_transitions_v4([r["traj"]], ts)
            p["ep_len"] = np.full((len(p["act"]),), int(r["ticks"]), dtype=np.int32)
            parts.append(p)

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}
    np.savez_compressed(OUT_DIR / "v4_sim_harvest_11tasks.npz", **merged)
    summary = {"total_rollouts": len(tasks), "passing_trajs": sum(cnts.values()), "total_chunks": int(len(merged["act"])), "per_task": cnts, "elapsed_sec": round(time.perf_counter() - t0, 1)}
    (OUT_DIR / "v4_harvest_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[v4 HARVEST COMPLETE] {sum(cnts.values())}/{len(tasks)} passing trajs -> {len(merged['act'])} chunks in {summary['elapsed_sec']}s | {cnts}", flush=True)


# ---------------------------------------------------------------------------
# Stage 3: Train 27.7M 17-Task x 11-Lab `v6_real` or `v6_mix` (OBS=47 Proven SE(2)!)
# ---------------------------------------------------------------------------
def stage_train_v6(mode: str = "real", n_steps: int = 10000, sim_ratio: float = 0.15, elite_pct: float = 50.0) -> None:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_specs = build_all_task_specs_v6()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((OUT_DIR / "task_meta_v6.json").read_text())
    active_tids = meta_obj["active_tids"]

    real_ds = dict(np.load(OUT_DIR / "uni_real_v6_17tasks.npz"))
    n_real = len(real_ds["act"])
    if mode == "mix":
        sim = dict(np.load(OUT_DIR / "v4_sim_harvest_11tasks.npz"))
        ep_len = sim["ep_len"]; stid = sim["tid"]
        keep = np.zeros(len(stid), dtype=bool)
        for t in sorted(np.unique(stid).tolist()):
            m_t = (stid == t) & (ep_len >= 35)
            if not m_t.any(): continue
            cut = float(np.percentile(ep_len[m_t], elite_pct))
            sel = m_t & (ep_len <= cut)
            keep |= sel
            print(f"  [v6 Elite Filter tid={t:2d}] kept {int(sel.sum()):5d}/{int(m_t.sum()):5d} chunks (cutoff<={cut:.1f}t)", flush=True)
        sim_ds = {k: v[keep] for k, v in sim.items() if k != "ep_len"}
        ds = {k: np.concatenate([real_ds[k], sim_ds[k]], axis=0) for k in real_ds.keys()}
        n_sim = len(sim_ds["act"])
        is_sim_flag = np.concatenate([np.zeros(n_real, dtype=bool), np.ones(n_sim, dtype=bool)])
        ckpt_name = "uni_bridge_v6_mix.pt"
    else:
        ds = real_ds; n_sim = 0; is_sim_flag = np.zeros(n_real, dtype=bool)
        ckpt_name = "uni_bridge_v6_real.pt"

    p_ee = torch.from_numpy(ds["p_ee"]).to(device)
    R_ee = torch.from_numpy(ds["R_ee"]).to(device)
    g_st = torch.from_numpy(ds["g_st"]).to(device)
    vel = torch.from_numpy(ds["vel"]).to(device)
    p_src = torch.from_numpy(ds["p_src"]).to(device)
    p_tgt = torch.from_numpy(ds["p_tgt"]).to(device)
    u_src = torch.from_numpy(ds["u_src"]).to(device)
    tz_t = torch.from_numpy(ds["table_z"]).to(device)
    geom_t = torch.from_numpy(ds["geom"]).to(device)
    act = torch.from_numpy(ds["act"]).to(device)
    tid_np = ds["tid"]
    tid_t = torch.from_numpy(tid_np).to(device)

    max_tid = max(active_tids) + 1
    pick_z_lut = torch.zeros((max_tid,), device=device, dtype=torch.float32)
    place_z_lut = torch.zeros((max_tid,), device=device, dtype=torch.float32)
    aniso_lut = torch.zeros((max_tid,), device=device, dtype=torch.float32)
    for t in active_tids:
        pick_z_lut[t] = float(spec_by_tid[t]["pick_z_off"])
        place_z_lut[t] = float(spec_by_tid[t]["place_z_off"])
        aniso_lut[t] = float(spec_by_tid[t]["aniso"])

    pools_r = {t: torch.from_numpy(np.where((tid_np == t) & (~is_sim_flag))[0]).to(device) for t in active_tids}
    pools_s = {t: torch.from_numpy(np.where((tid_np == t) & is_sim_flag)[0]).to(device) for t in active_tids}
    step_ladder = (torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32) * 0.018)[None, :, None]

    def sample_balanced_idx(batch_size: int) -> torch.Tensor:
        per_t = max(8, batch_size // len(active_tids))
        chunks = []
        for t in active_tids:
            pr, ps = pools_r[t], pools_s[t]
            if mode == "mix" and len(ps) >= 32 and sim_ratio > 0.0:
                ns = max(1, int(round(per_t * sim_ratio))); nr = per_t - ns
                chunks.append(pr[torch.randint(0, len(pr), (nr,), device=device)])
                chunks.append(ps[torch.randint(0, len(ps), (ns,), device=device)])
            else:
                chunks.append(pr[torch.randint(0, len(pr), (per_t,), device=device)])
        return torch.cat(chunks, dim=0)

    def make_batch(idx: torch.Tensor):
        B = len(idx)
        tz_b = tz_t[idx]; tids = tid_t[idx]; g_b = geom_t[idx]; is_hol = g_b[:, 6]; an = aniso_lut[tids]

        vlm_ec = torch.cat([torch.randn((B, 2), device=device) * 0.022, torch.randn((B, 1), device=device) * 0.010], dim=-1).clamp(-0.042, 0.042)
        vlm_eb = torch.cat([torch.randn((B, 2), device=device) * 0.022, torch.randn((B, 1), device=device) * 0.010], dim=-1).clamp(-0.042, 0.042)
        p_pick_vlm = torch.stack([p_src[idx, 0], p_src[idx, 1], tz_b + pick_z_lut[tids]], dim=-1) + vlm_ec
        p_place_vlm = torch.stack([p_tgt[idx, 0], p_tgt[idx, 1], tz_b + place_z_lut[tids]], dim=-1) + vlm_eb

        d_s_xy_o = torch.norm(p_src[idx, :2] - p_ee[idx, :2], dim=-1)
        is_held = (g_st[idx] > 0.35) & (d_s_xy_o < (0.065 + 0.035 * is_hol))
        jit = torch.randn((B, 3), device=device) * 0.007
        p_ee_q = p_ee[idx] + jit
        p_src_q = torch.where(is_held[:, None], p_src[idx] + jit, p_src[idx])
        p_tgt_q = p_tgt[idx]

        a_b = act[idx].clone()
        a_b[:, :, :3] = a_b[:, :, :3] - jit[:, None, :]

        d_src = p_src_q - p_ee_q; d_tgt = p_tgt_q - p_ee_q; d_st = p_src_q - p_tgt_q
        d_vp = p_pick_vlm - p_ee_q; d_vb = p_place_vlm - p_ee_q
        z_ax = R_ee[idx, :, 2]; v_cur = vel[idx]

        th = torch.rand((B,), device=device) * (2.0 * math.pi)
        ct, st = torch.cos(th), torch.sin(th)
        Rz = torch.zeros((B, 3, 3), device=device)
        Rz[:, 0, 0] = ct; Rz[:, 0, 1] = -st
        Rz[:, 1, 0] = st; Rz[:, 1, 1] = ct
        Rz[:, 2, 2] = 1.0
        def rot3(v): return torch.bmm(Rz, v.unsqueeze(-1)).squeeze(-1)

        d_src = rot3(d_src); d_tgt = rot3(d_tgt); d_st = rot3(d_st)
        d_vp = rot3(d_vp); d_vb = rot3(d_vb); z_ax = rot3(z_ax); v_cur = rot3(v_cur)
        a_b[:, :, :3] = torch.bmm(a_b[:, :, :3], Rz.transpose(1, 2))
        a_b[:, :, 3:6] = torch.bmm(a_b[:, :, 3:6], Rz.transpose(1, 2))

        u0 = u_src[idx]
        ux = ct * u0[:, 0] - st * u0[:, 1]; uy = st * u0[:, 0] + ct * u0[:, 1]
        un = torch.sqrt(ux * ux + uy * uy).clamp(min=1e-5)
        ux = ux / un; uy = uy / un
        sgn = torch.where(torch.rand((B,), device=device) > 0.5, 1.0, -1.0)
        ori_feats = torch.stack([an * ux * sgn, an * uy * sgn, an * (ux * ux - uy * uy), an * (2.0 * ux * uy)], dim=-1)

        w_pl = torch.clamp((g_st[idx] - 0.20) / 0.40, 0.0, 1.0)[:, None]
        d_goal = (1.0 - w_pl) * d_vp + w_pl * d_vb
        dist_g = torch.norm(d_goal, dim=-1, keepdim=True).clamp(min=1e-5)
        u_g = d_goal / dist_g
        a_line = torch.zeros_like(a_b)
        a_line[:, :, :3] = torch.minimum(step_ladder, dist_g[:, :, None]) * u_g[:, None, :]
        a_line[:, :, 6] = g_st[idx][:, None]

        def dir_sat_t(v, scale):
            n = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-5)
            n_xy = torch.norm(v[:, :2], dim=-1, keepdim=True)
            return torch.cat([v / n, torch.tanh(n / scale), torch.tanh(n_xy / scale), v[:, 2:3].clamp(-0.25, 0.25)], dim=-1)

        t_vp = dir_sat_t(d_vp, 0.15)
        t_vb = dir_sat_t(d_vb, 0.15)
        t_ls = torch.cat([dir_sat_t(d_src, 0.12), (p_src_q[:, 2] - tz_b).unsqueeze(-1), ori_feats], dim=-1)
        t_lt = torch.cat([dir_sat_t(d_tgt, 0.12), torch.tanh(torch.norm(d_st[:, :2], dim=-1, keepdim=True) / 0.12), d_st[:, 2:3].clamp(-0.2, 0.2)], dim=-1)
        d_s_xy_n = torch.norm(d_src[:, :2], dim=-1)
        held_cue = (g_st[idx] * torch.exp(-((d_s_xy_n / (0.055 + 0.035 * is_hol)) ** 2))).unsqueeze(-1)
        t_ee = torch.cat([(p_ee_q[:, 2] - tz_b).unsqueeze(-1), z_ax, g_st[idx].unsqueeze(-1), v_cur, held_cue], dim=-1)
        obs_b = torch.cat([t_vp, t_vb, t_ls, t_lt, t_ee, g_b], dim=-1)
        return obs_b, a_line, a_b

    d_model, n_layers, n_heads = 384, 8, 8
    model = v4.build_v4_dit(d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    with torch.no_grad():
        _, _, a_stat = make_batch(sample_balanced_idx(6000))
        act_mean = a_stat.mean(dim=(0, 1))
        act_std = a_stat.std(dim=(0, 1)).clamp(min=0.03)

    ema_model = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v6_{mode}] {n_params/1e6:.2f}M params (d={d_model}, L={n_layers}, OBS=47, {len(active_tids)} tasks) | {n_steps} steps | Real={n_real} + EliteSim={n_sim}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2.8e-4, weight_decay=1e-4, fused=True)
    warmup = 300; bs = 544
    def lr_fn(s):
        if s < warmup: return float(s + 1) / float(warmup)
        p = float(s - warmup) / float(max(1, n_steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    beta_dist = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
    ema_params = list(ema_model.parameters()); cur_params = list(model.parameters())
    w_dim = torch.tensor([1.3, 1.3, 1.5, 0.85, 0.85, 1.0, 2.2], device=device)[None, None, :]

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        with torch.no_grad():
            obs_b, al_b, ae_b = make_batch(sample_balanced_idx(bs))
            al_n = ((al_b - act_mean) / act_std).clamp(-6.0, 6.0)
            a1 = ((ae_b - act_mean) / act_std).clamp(-6.0, 6.0)
            a0 = al_n + BRIDGE_SIGMA * torch.randn_like(a1)
            tau = beta_dist.sample((len(obs_b),)).clamp(1e-3, 1.0 - 1e-3)
            tv = tau[:, None, None]
            a_tau = (1.0 - tv) * a0 + tv * a1
            v_tgt = a1 - a0
        opt.zero_grad(set_to_none=True)
        v_pred = model(a_tau, al_n, tau, obs_b)
        loss = torch.mean(w_dim * ((v_pred - v_tgt) ** 2))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, cur_params, 0.005)
        if step % 2000 == 0 or step == 1 or step == n_steps:
            print(f"  [v6_{mode} step {step:5d}/{n_steps}] loss={loss.item():.4f} | {(time.perf_counter()-t0)/step*1000:.1f} ms/step ({time.perf_counter()-t0:.1f}s)", flush=True)

    train_dt = time.perf_counter() - t0
    with torch.no_grad():
        obs_v, al_v, ae_v = make_batch(sample_balanced_idx(544))
        pred = ema_model.render_chunk(((al_v - act_mean) / act_std).clamp(-6, 6), obs_v, n_steps=ODE_STEPS, seed=42) * act_std + act_mean
        mae_cm = torch.mean(torch.abs(pred[:, :6, :3] - ae_v[:, :6, :3])).item() * 100.0
        g_mae = torch.mean(torch.abs(pred[:, :6, 6] - ae_v[:, :6, 6])).item()
        print(f"[v6_{mode} COMPLETE] {train_dt:.1f}s | 4-step ODE PosMAE={mae_cm:.2f}cm | GripMAE={g_mae:.3f}", flush=True)

    torch.save({
        "ema_state_dict": ema_model.state_dict(),
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "obs_dim": OBS_DIM,
        "act_mean": act_mean.cpu().numpy(), "act_std": act_std.cpu().numpy(),
        "active_tids": active_tids, "n_real_chunks": n_real, "n_sim_chunks": n_sim,
        "train_sec": round(train_dt, 1), "val_pos_mae_cm": round(mae_cm, 2), "val_grip_mae": round(g_mae, 3),
    }, OUT_DIR / ckpt_name)


def stage_serve_v6(ckpt_name: str, port: int = PORT_V6) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / ckpt_name, map_location=device, weights_only=False)
    model = v4.build_v4_dit(int(ck["d_model"]), int(ck["n_layers"]), int(ck["n_heads"])).to(device).eval()
    model.load_state_dict(ck["ema_state_dict"])
    act_mean = torch.from_numpy(ck["act_mean"]).to(device)
    act_std = torch.from_numpy(ck["act_std"]).to(device)

    with torch.no_grad():
        _ = model.render_chunk(torch.zeros((1, HORIZON, ACT_DIM), device=device), torch.zeros((1, OBS_DIM), device=device), n_steps=ODE_STEPS)

    lock = threading.Lock(); stop_flag = threading.Event()
    req_bytes = 4 + (OBS_DIM + HORIZON * ACT_DIM) * 4

    def handle_client(conn: socket.socket):
        try:
            while not stop_flag.is_set():
                buf = b""
                while len(buf) < req_bytes:
                    c = conn.recv(req_bytes - len(buf))
                    if not c: return
                    buf += c
                seed = struct.unpack("<i", buf[:4])[0]
                if seed == -999999:
                    stop_flag.set(); return
                floats = np.frombuffer(buf[4:], dtype=np.float32).copy()
                obs_t = torch.from_numpy(floats[:OBS_DIM]).to(device)[None, :]
                al_t = torch.from_numpy(floats[OBS_DIM:].reshape(HORIZON, ACT_DIM)).to(device)[None, :, :]
                al_n = ((al_t - act_mean) / act_std).clamp(-6.0, 6.0)
                with lock, torch.no_grad():
                    a_n = model.render_chunk(al_n, obs_t, n_steps=ODE_STEPS, seed=(None if seed < 0 else int(seed)))
                    a_out = (a_n * act_std + act_mean)[0].cpu().numpy().astype(np.float32)
                conn.sendall(a_out.tobytes())
        finally:
            conn.close()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(0.5); srv.bind(("127.0.0.1", port)); srv.listen(64)
    print(f"[Serve v6 {ckpt_name}] READY on 127.0.0.1:{port} (d={ck['d_model']}, L={ck['n_layers']})", flush=True)
    try:
        while not stop_flag.is_set():
            try:
                conn, _ = srv.accept()
                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
    finally:
        srv.close()


def stage_eval_v6(tag: str, scope: str = "orig5", port: int = PORT_V6, random_yaw: bool = False) -> None:
    t0 = time.perf_counter()
    n_w = int(os.environ.get("R2S_WORKERS", "24"))
    all_specs = build_all_task_specs_v6()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((OUT_DIR / "task_meta_v6.json").read_text())

    if scope == "orig5":
        eval_tids = [0, 1, 2, 3, 4]; rng = np.random.default_rng(2026)
        meta_src = json.loads((uni.OUT_DIR / "task_meta.json").read_text())
    elif scope == "new6":
        eval_tids = [t for t in meta_obj["active_tids"] if 5 <= t <= 10]; rng = np.random.default_rng(2027)
        meta_src = meta_obj["tasks"]
    else:
        eval_tids = [t for t in meta_obj["active_tids"] if t >= 11]; rng = np.random.default_rng(2028)
        meta_src = meta_obj["tasks"]

    tasks = []
    for t in eval_tids:
        ts = spec_by_tid[t]; tm = meta_src[str(t)]
        x0, x1, y0, y1 = tm["xy_bounds"]; dmin, dmax = tm["dist_range"]
        tz = float(tm["table_z_med"]); qh = np.array(tm["q_home"], dtype=np.float32)
        for i in range(25):
            for _ in range(200):
                sxy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                txy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                if dmin <= math.hypot(sxy[0] - txy[0], sxy[1] - txy[1]) <= dmax: break
            ec = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
            eb = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
            syaw = float(rng.uniform(-math.pi, math.pi)) if (random_yaw and ts["aniso"] > 0.2) else 0.0
            tasks.append((ts, i, sxy, txy, syaw, tz, qh, ec, eb, 90000 + t * 1000 + i * 17, port))

    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
        for f in as_completed([pool.submit(v4._run_one_v4_ep, x) for x in tasks]):
            results.append(f.result())

    v4._shutdown_v4(port)

    by_task = {}
    for t in eval_tids:
        ts = spec_by_tid[t]
        rs = [r for r in results if r["tid"] == t]
        ok = sum(int(r["success"]) for r in rs)
        by_task[ts["name"]] = {"pass": ok, "n": len(rs), "rate": round(100.0 * ok / max(1, len(rs)), 1)}

    tot_ok = sum(int(r["success"]) for r in results); tot_n = len(results)
    full_tag = f"{tag}_{scope}" + ("_randyaw" if random_yaw else "")
    out = {"tag": full_tag, "pass": tot_ok, "n": tot_n, "rate": round(100.0 * tot_ok / max(1, tot_n), 1), "per_task": by_task, "elapsed_sec": round(time.perf_counter() - t0, 1)}
    (OUT_DIR / f"eval_{full_tag}.json").write_text(json.dumps(out, indent=2))
    print("=" * 92, flush=True)
    print(f"[EVAL {full_tag}] TOTAL = {tot_ok}/{tot_n} ({out['rate']:.1f}%) in {out['elapsed_sec']}s", flush=True)
    for k, v in by_task.items():
        print(f"  - {k:28s}: {v['pass']:2d}/{v['n']:2d} ({v['rate']:5.1f}%)", flush=True)
    print("=" * 92, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "harvest_v4", "train_real", "train_mix", "serve", "eval"])
    ap.add_argument("--ckpt", default="uni_bridge_v6_real.pt")
    ap.add_argument("--tag", default="v6_real")
    ap.add_argument("--scope", choices=["orig5", "new6", "brandnew6"], default="orig5")
    ap.add_argument("--port", type=int, default=PORT_V6)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--n-per-task", type=int, default=160)
    ap.add_argument("--sim-ratio", type=float, default=0.15)
    ap.add_argument("--elite-pct", type=float, default=50.0)
    ap.add_argument("--random-yaw", action="store_true")
    args = ap.parse_args()

    if args.cmd == "collect":
        stage_collect_v6()
    elif args.cmd == "harvest_v4":
        stage_harvest_v4(n_per_task=args.n_per_task, port=args.port)
    elif args.cmd == "train_real":
        stage_train_v6("real", n_steps=args.steps)
    elif args.cmd == "train_mix":
        stage_train_v6("mix", n_steps=args.steps, sim_ratio=args.sim_ratio, elite_pct=args.elite_pct)
    elif args.cmd == "serve":
        stage_serve_v6(args.ckpt, port=args.port)
    elif args.cmd == "eval":
        stage_eval_v6(args.tag, scope=args.scope, port=args.port, random_yaw=args.random_yaw)


if __name__ == "__main__":
    main()
