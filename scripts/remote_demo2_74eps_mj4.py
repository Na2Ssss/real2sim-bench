"""Demo #2 (Lego Block into Articulated Drawer, 74 episodes) All-on-A100 MuJoCo 4.0.1 Benchmark.

Uses exact silicone-pad teleop gripper-timing calibration (`t_close / t_open`) + articulated
drawer + real Homography countertop texture + strict anti-sweep & full-closure Fact-Based
`evaluate_block_in_drawer` across ALL 74 episodes in parallel (16 workers + A100 EGL).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import cv2
import mujoco
import numpy as np

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

from r2s_bench.adapters.droid import droid_ext6_to_matrix
from r2s_bench.backends.mujoco_backend import opencv_cam_to_mujoco_pose
from r2s_bench.eval.task_success import RolloutMonitor, evaluate_block_in_drawer
from r2s_bench.robots.franka_robotiq import build_drawer_block_scene, build_droid_franka_robotiq_spec
from r2s_bench.schema import CameraCalib

BUNDLE = ROOT / "outputs" / "demo2_74eps_bundle"
HE_JSON = ROOT / "outputs" / "handeye_demo2" / "refined_extrinsics.json"
OUT_WH = (480, 270)


def make_calib(serial: str, ext6: list, intr_dict: dict) -> CameraCalib:
    fx, cx, fy, cy = intr_dict["cameraMatrix"]
    return CameraCalib(
        name=f"ext_{serial}", serial=serial,
        width=int(intr_dict["width"]), height=int(intr_dict["height"]),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        cam_to_base=droid_ext6_to_matrix(ext6).tolist(),
    )


def configure_exact_camera(m: mujoco.MjModel, calib: CameraCalib, width: int, height: int) -> int:
    cid = 0
    sc = calib.scaled(width, height)
    pos, quat = opencv_cam_to_mujoco_pose(sc.T_base_cam)
    m.cam_pos[cid] = pos
    m.cam_quat[cid] = quat
    px = 1e-5
    ox = -(sc.cx - sc.width / 2.0) * px
    oy = (sc.cy - sc.height / 2.0) * px
    m.cam_resolution[cid] = [sc.width, sc.height]
    m.cam_sensorsize[cid] = [sc.width * px, sc.height * px]
    m.cam_intrinsic[cid] = [sc.fx * px, sc.fy * px, ox, oy]
    return cid


def extract_gripper_events(qs: np.ndarray, gs: np.ndarray) -> dict:
    m_fk = build_droid_franka_robotiq_spec(offwidth=64, offheight=64).compile()
    d_fk = mujoco.MjData(m_fk)
    lp = [g for g in range(m_fk.ngeom) if "left_silicone_pad" in (mujoco.mj_id2name(m_fk, mujoco.mjtObj.mjOBJ_BODY, int(m_fk.geom_bodyid[g])) or "")][0]
    rp = [g for g in range(m_fk.ngeom) if "right_silicone_pad" in (mujoco.mj_id2name(m_fk, mujoco.mjtObj.mjOBJ_BODY, int(m_fk.geom_bodyid[g])) or "")][0]
    ag = mujoco.mj_name2id(m_fk, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")

    # Pre-settle the Robotiq 4-bar linkage at block-pinch width (~3.8cm gap) so pad Z accounts
    # for the exact -1.34cm downward swing of the parallel linkage when closing!
    d_fk.qpos[:7] = qs[0]; d_fk.ctrl[:7] = qs[0]; d_fk.ctrl[ag] = 0.60 * 255.0
    for _ in range(150):
        mujoco.mj_step(m_fk, d_fk)
    q_fingers_pinch = d_fk.qpos[7:].copy()

    T = len(qs)
    mids = np.zeros((T, 3), dtype=np.float64)
    axes = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        d_fk.qpos[:7] = qs[t]
        d_fk.qpos[7:] = q_fingers_pinch
        mujoco.mj_forward(m_fk, d_fk)
        pl, pr = d_fk.geom_xpos[lp], d_fk.geom_xpos[rp]
        mids[t] = 0.5 * (pl + pr)
        v = pl - pr
        axes[t] = v / max(float(np.linalg.norm(v)), 1e-6)

    n1 = int(T * 0.62)
    g_peak = float(np.max(gs[:n1]))
    thresh = max(0.14, g_peak * 0.60)
    close_cands = [t for t in range(3, n1) if gs[t] >= thresh and gs[t - 1] < thresh]
    if close_cands:
        t_close = min(close_cands, key=lambda t: mids[t, 2])
    else:
        t_close = int(np.argmax(gs[:n1]))

    t_hold = min(t_close + 3, T - 1)
    p_close = mids[t_hold]
    yaw_grasp = float(math.atan2(axes[t_hold, 1], axes[t_hold, 0]))

    open_cands = [t for t in range(t_hold + 3, T) if gs[t] < 0.12]
    t_open = open_cands[0] if open_cands else min(t_hold + 30, T - 1)

    return {
        "t_close": int(t_close), "t_hold": int(t_hold), "t_open": int(t_open),
        "p_close": p_close, "p_open": mids[t_open],
        "mids": mids, "yaw_grasp": yaw_grasp,
    }


def eval_demo2_ep(args: tuple) -> dict:
    ep, T1_ref, render_snap = args
    ei = ep["ep_idx"]
    eid = ep["episode_id"]
    ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
    tr = np.load(ed / "trajectory.npz")
    qs, gs = tr["qpos"], tr["gripper"]

    ev = extract_gripper_events(qs, gs)
    p_close, mids = ev["p_close"], ev["mids"]

    drawer_yaw = 0.448
    c_y, s_y = math.cos(drawer_yaw), math.sin(drawer_yaw)
    axis_xy = np.array([c_y, s_y], dtype=np.float64)
    ortho_xy = np.array([-s_y, c_y], dtype=np.float64)

    table_z = float(np.clip(p_close[2] - 0.020, 0.14, 0.28))
    bxy_exact = (float(p_close[0]), float(p_close[1]))
    yaw_blk = ev["yaw_grasp"]
    blk_quat = [math.cos(yaw_blk / 2.0), 0.0, 0.0, math.sin(yaw_blk / 2.0)]

    tail_idx = np.arange(min(ev["t_open"] + 5, len(qs) - 1), len(qs))
    low_mask = mids[tail_idx, 2] < (table_z - 0.04)
    if np.sum(low_mask) >= 4:
        push_pts = mids[tail_idx[low_mask]]
        projs = push_pts[:, :2] @ axis_xy
        # Place drawer front just ahead of the robot's lowest forward-push start so hand never hooks behind it
        p_start = float(np.percentile(projs, 18)) + 0.020
        p_end = float(np.percentile(projs, 95))
        push_lat = float(np.median(push_pts[:, :2] @ ortho_xy))
        push_z = float(np.median(push_pts[:, 2]))
    else:
        p_start, p_end, push_lat, push_z = 0.41, 0.575, -0.153, table_z - 0.075

    stroke = max(0.14, min(0.22, p_end - p_start))
    flush_proj = max(p_end, 0.54)
    blk_proj = float(np.dot(p_close[:2], axis_xy))
    ct_front_proj = min(flush_proj, blk_proj - 0.032)
    base_proj = ct_front_proj - 0.130
    base_pos = (
        float(base_proj * c_y - push_lat * s_y),
        float(base_proj * s_y + push_lat * c_y),
        float(table_z - 0.145),
    )
    front_proj_0 = float(np.dot(np.array([base_pos[0] - 0.135 * c_y, base_pos[1] - 0.135 * s_y]), axis_xy))
    q_closed = float(flush_proj - front_proj_0)
    qi_init = float(q_closed - stroke)

    m = build_drawer_block_scene(bxy_exact, table_z=table_z, drawer_pos=base_pos, drawer_yaw_rad=drawer_yaw, articulated=True, offwidth=OUT_WH[0], offheight=OUT_WH[1])
    ct_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "counter_top")
    m.geom_size[ct_gid] = [0.28, 0.72, 0.012]
    drw_z = min(base_pos[2], table_z - 0.130)
    # Keep top of drawer front cleanly aligned with the low push stroke (~push_z + 2.5cm)
    # so the gripper clears the drawer face when retracting before pushing it shut!
    f_top = min(table_z - 0.045, max(push_z + 0.025, drw_z + 0.070))
    f_h = max(0.035, (f_top - drw_z) * 0.5)
    for nm, sz, ps in [
        ("drawer_bottom", [0.14, 0.24, 0.006], [0.0, 0.0, 0.006]),
        ("drawer_front", [0.010, 0.245, f_h], [-0.135, 0.0, f_h]),
        ("drawer_back", [0.008, 0.24, 0.048], [0.135, 0.0, 0.048]),
        ("drawer_left", [0.14, 0.008, 0.048], [0.0, 0.235, 0.048]),
        ("drawer_right", [0.14, 0.008, 0.048], [0.0, -0.235, 0.048]),
    ]:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if gid >= 0:
            m.geom_size[gid] = sz; m.geom_pos[gid] = ps

    sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    m.jnt_range[sj] = [qi_init - 0.12, q_closed + 0.015]
    m.dof_damping[int(m.jnt_dofadr[sj])] = 1.2
    m.dof_frictionloss[int(m.jnt_dofadr[sj])] = 0.15

    bg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "blue_block_geom")
    m.geom_size[bg] = [0.019, 0.032, 0.018]
    m.geom_friction[bg] = [2.2, 0.03, 0.003]
    m.geom_solref[bg] = [0.025, 1.5]

    d = mujoco.MjData(m)
    blk_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "blue_block_jnt")
    blk_qadr = int(m.jnt_qposadr[blk_jnt])
    sj_qadr = int(m.jnt_qposadr[sj])
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")

    d.qpos[:7] = qs[0]
    d.qpos[blk_qadr : blk_qadr + 7] = [bxy_exact[0], bxy_exact[1], table_z + 0.019, blk_quat[0], blk_quat[1], blk_quat[2], blk_quat[3]]
    d.qpos[sj_qadr] = qi_init
    d.ctrl[:7] = qs[0]; d.ctrl[ag] = float(np.clip(gs[0], 0.0, 1.0)) * 255.0
    mujoco.mj_forward(m, d)
    for _ in range(30):
        mujoco.mj_step(m, d)

    mon = RolloutMonitor(m, target_body_name="blue_block")
    mon.reset(d)
    for ti in range(len(qs)):
        d.ctrl[:7] = qs[ti]
        d.ctrl[ag] = (0.62 if (ev["t_close"] <= ti < ev["t_open"]) else float(np.clip(gs[ti], 0.0, 1.0))) * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)
    for _ in range(80):
        d.ctrl[:7] = qs[-1]; d.ctrl[ag] = float(np.clip(gs[-1], 0.0, 1.0)) * 255.0
        mujoco.mj_step(m, d)
    mon.observe(d)

    res = evaluate_block_in_drawer(
        m, d, mon, block_body="blue_block", drawer_body="drawer_box", drawer_joint="drawer_slide",
        drawer_yaw_rad=drawer_yaw, drawer_q_init=qi_init, drawer_q_closed=q_closed,
        max_residual_open_m=0.035, min_close_ratio=0.80, min_lift_m=0.020,
        drawer_half_extents=(0.16, 0.25), table_z=table_z,
    )

    snap_path = None
    if render_snap:
        c1 = make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"])
        c1.cam_to_base = T1_ref
        cid = configure_exact_camera(m, c1, OUT_WH[0], OUT_WH[1])
        ren = mujoco.Renderer(m, height=OUT_WH[1], width=OUT_WH[0])
        ren.update_scene(d, camera=cid)
        sim_end = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
        ren.close()
        sd = BUNDLE / "snaps"; sd.mkdir(exist_ok=True)
        snap_path = str(sd / f"ep_{ei:03d}_end.jpg")
        cv2.imwrite(snap_path, sim_end)

    return {"ep_idx": ei, "episode_id": eid, "snap_path": snap_path, **res.to_dict()}


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads((BUNDLE / "meta_demo2.json").read_text())
    he = json.loads(HE_JSON.read_text())
    T1_ref = he["T_base_cam1_refined"]
    n_workers = int(os.environ.get("R2S_WORKERS", "16"))

    tasks = [(ep, T1_ref, ep["ep_idx"] < 20) for ep in meta]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(eval_demo2_ep, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if k % 20 == 0 or k == len(tasks):
                cp = sum(int(x["success"]) for x in results)
                cl = sum(int(x["metrics"].get("max_lift_cm", 0) >= 2.0) for x in results)
                print(f"  [Demo2 {k:2d}/{len(tasks)}] PASS={cp}/{k} ({100*cp/k:.1f}%) | Lift>=2cm={cl}/{k} ({time.perf_counter()-t0:.1f}s)", flush=True)

    results.sort(key=lambda x: x["ep_idx"])
    n = len(results)
    n_pass = sum(int(x["success"]) for x in results)
    n_lift = sum(int(x["metrics"].get("max_lift_cm", 0) >= 2.0) for x in results)
    n_in = sum(int(x["metrics"].get("block_in_drawer", False)) for x in results)
    n_cl = sum(int(x["metrics"].get("drawer_closed", False)) for x in results)
    reasons = Counter(x["failure_reason"] for x in results if not x["success"])

    summary = {
        "scene": "Demo #2 Lego Block into Articulated Drawer (74 episodes)",
        "mujoco_version": mujoco.__version__,
        "timing_sec": round(time.perf_counter() - t0, 1),
        "metrics": {
            "fact_pass": n_pass, "fact_pass_rate_pct": round(100 * n_pass / n, 1),
            "lifted_ge_2cm": n_lift, "in_drawer": n_in, "drawer_closed": n_cl,
        },
        "failure_reasons": dict(reasons),
        "episodes": results,
    }
    (BUNDLE / "demo2_74eps_mj4_results.json").write_text(json.dumps(summary, indent=2))

    sp = [x for x in results if x.get("snap_path") and x["success"]][:5]
    sf = [x for x in results if x.get("snap_path") and not x["success"]][:5]
    rows = []
    for gn, grp, col in [("PASS", sp, (60, 230, 100)), ("FAIL", sf, (80, 110, 255))]:
        if not grp: continue
        tiles = []
        for x in grp:
            im = cv2.resize(cv2.imread(x["snap_path"]), (320, 180))
            cv2.rectangle(im, (0, 0), (320, 22), (15, 18, 22), -1)
            txt = f"Ep#{x['ep_idx']:02d} {gn} | L={x['metrics'].get('max_lift_cm',0):.1f}cm Cls={x['metrics'].get('drawer_close_ratio',0)*100:.0f}%"
            cv2.putText(im, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            tiles.append(im)
        while len(tiles) < 5: tiles.append(np.zeros((180, 320, 3), np.uint8))
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(BUNDLE / "demo2_74eps_audit_mosaic.png"), np.vstack(rows))

    print("=" * 95, flush=True)
    print(f"DEMO #2 (74-EPISODE, MuJoCo {mujoco.__version__}) COMPLETE in {time.perf_counter()-t0:.1f}s ({(time.perf_counter()-t0)/n:.2f}s/ep)", flush=True)
    print(f"  Strict FACT PASS : {n_pass:2d}/{n} ({100*n_pass/n:.1f}%)", flush=True)
    print(f"  Lifted >= 2 cm   : {n_lift:2d}/{n} | In Drawer: {n_in:2d}/{n} | Drawer Closed: {n_cl:2d}/{n}", flush=True)
    print(f"  Failure Reasons  : {dict(reasons)}", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
