"""High-Throughput All-17-Task OOD Trajectory Harvester Powered by 75.2%/57.2% Uni-Goal-Bridge v8!

Runs `Uni-Goal-Bridge v8` (`OBS=48`, 8D physical geometry + 2-step DLS-IK) across 160 unseen full-table OOD
placements per task x ALL 17 tasks = 2,720 rollouts (disjoint seed=99999, 360-deg random yaw on every
anisotropic object), saving EVERY physically passing trajectory with exact `ep_len` per chunk into
`outputs/uni_goal_bridge_v8/v8_sim_harvest_all17.npz` for Uni-Goal-Bridge v9!
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

import scripts.remote_uni_goal_bridge_scaling as uni
import scripts.remote_uni_goal_bridge_v4 as v4
import scripts.remote_uni_goal_bridge_v6 as v6
import scripts.remote_uni_goal_bridge_v8 as v8

OUT_DIR = v8.OUT_DIR
PORT = 18984


def _harvest_one_v8_ep(args: tuple) -> dict:
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
    gv7 = uni.task_geom_vec(tspec)
    is_tc = np.array([float(tspec["is_tgt_container"])], dtype=np.float32)

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
            obs47 = v4.build_obs_v4_np(p_ee, R_ee, g_st, ee_vel, p_src, p_tgt, u_src, p_pick_vlm, p_place_vlm, tz, gv7)
            obs48 = np.concatenate([obs47, is_tc], axis=0)
            a_line = uni.make_straight_line_prior_np(p_ee, p_pick_vlm, p_place_vlm, g_st)
            act_chunk = client.query(obs48, a_line, seed=seed + tick * 7)
            chunk_p_targets = p_ee[None, :] + act_chunk[:, :3]
            chunk_R_targets = [Rotation.from_rotvec(act_chunk[k, 3:6]).as_matrix() @ R_ee for k in range(uni.HORIZON)]
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


def main() -> None:
    t0 = time.perf_counter()
    n_per_task = int(os.environ.get("HARVEST_PER_TASK", "160"))
    n_w = int(os.environ.get("R2S_WORKERS", "24"))
    all_specs = v8.build_all_task_specs_v8()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((v6.OUT_DIR / "task_meta_v6.json").read_text())
    active_tids = meta_obj["active_tids"]

    rng = np.random.default_rng(99999)
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
            tasks.append((ts, i, sxy, txy, syaw, tz, qh, ec, eb, 950000 + t * 10000 + i * 37, PORT))

    print(f"[v8 All-17 Harvest] Launching {len(tasks)} OOD rollouts ({n_per_task}/task x {len(active_tids)} tasks) on {n_w} workers...", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
        for k, f in enumerate(as_completed([pool.submit(_harvest_one_v8_ep, x) for x in tasks]), 1):
            results.append(f.result())
            if k % 400 == 0 or k == len(tasks):
                ok_n = sum(int(r["success"]) for r in results)
                print(f"  [v8 Harvest {k:4d}/{len(tasks)}] PASS={ok_n} ({100*ok_n/k:.1f}%) ({time.perf_counter()-t0:.1f}s)", flush=True)

    v8._shutdown_v8(PORT)

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
    np.savez_compressed(OUT_DIR / "v8_sim_harvest_all17.npz", **merged)
    summary = {"total_rollouts": len(tasks), "passing_trajs": sum(cnts.values()), "total_chunks": int(len(merged["act"])), "per_task": cnts, "elapsed_sec": round(time.perf_counter() - t0, 1)}
    (OUT_DIR / "v8_harvest_all17_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[v8 ALL-17 HARVEST COMPLETE] {sum(cnts.values())}/{len(tasks)} passing trajs -> {len(merged['act'])} chunks in {summary['elapsed_sec']}s | {cnts}", flush=True)


if __name__ == "__main__":
    main()
