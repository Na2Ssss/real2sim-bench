"""Uni-Goal-Bridge DiT Action Render: Dual-Axis Scaling (5 Labs/Skills + Sim-in-the-Loop Flywheel).

ONE unified 4-step Bridge-Flow DiT Action Render (ZERO intermediate waypoints, ZERO v_base funnel,
ZERO phase state machine, ZERO hard-coded gripper rules!) scaled across TWO axes:

  Axis 1 — Multi-Lab, Multi-Skill Unification (4 Robotics Labs x 5 Diverse Object/Grasp Skills):
    Task 0 (AUTOLab):     Orange Cup -> Black Bowl         (tapered wrap-grasp, 50 real demos)
    Task 1 (GuptaLab):    Orange Block -> Green Block      (4.6cm precision stacking!, 42 real demos)
    Task 2 (AUTOLab):     Orange Rubber Duck -> Silver Pot (irregular geometry -> deep pot, 29 real demos)
    Task 3 (PennPAL):     Black Tape Roll -> White Plate   (hollow-ring rim_pinch!, 45 real demos)
    Task 4 (RPL Kitchen): Mini Soda Can -> Bowl            (tall slender 5.6x10.4cm can, 30 real demos)

  Axis 2 — Closed-Loop Sim-in-the-Loop Data Scaling Flywheel:
    Round 1 (v1, Real-Only):    Train on ~196 real passing DROID trajectories -> eval on 125 held-out
                                unseen full-table OOD placements (25/task, +-2.5cm VLM click error)
    Harvest (Sim Flywheel):     Rollout v1 across 250 brand-new random OOD placements (disjoint seed!)
                                in MuJoCo 4.0.1 -> keep EVERY physically passing trajectory!
    Round 2 (v2, Bootstrapped): Fine-tune same DiT on Real + Sim-Harvested OOD -> re-evaluate on the
                                EXACT SAME 125 held-out unseen OOD test placements!

Output:
  - Full v1 vs v2 vs Naive-VLM scaling report (`scaling_summary.json`)
  - 5-Row Synchronized Vertical Stack H.264 Showcase (`uni_goal_bridge_5tasks_scaling_showcase.mp4`)
  - Visual audit sheet (`inspect_uni_goal_bridge_5tasks.png`)
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
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "uni_goal_bridge_scaling"
DEMO1_DIT_DATA = ROOT / "outputs" / "dit_flow_demo1" / "dit_dataset.npz"
DEMO1_BUNDLE = ROOT / "outputs" / "demo1_100eps_bundle"
DEMO1_HY3D = ROOT / "outputs" / "hy3d_demo1"
DEMO1_HE = ROOT / "outputs" / "handeye_demo1" / "refined_extrinsics.json"

DT_CTRL = 1.0 / 15.0
OUT_WH = (480, 270)
HORIZON = 16
ACT_DIM = 7   # [dp_x, dp_y, dp_z, dr_x, dr_y, dr_z, g_target]
OBS_DIM = 43  # 36 (same SE(2) saturated tokens) + 7 (physical geometry descriptor: src_xyz, tgt_xyz, is_hollow)
PORT = 18958
BRIDGE_SIGMA = 0.35
ODE_STEPS = 4

TASK_SPECS = [
    {
        "tid": 0, "name": "demo1_cup_bowl", "label": "1. AUTOLab | Orange Cup -> Black Bowl",
        "kind": "demo1", "src_body": "orange_cup", "tgt_body": "black_bowl",
        "src_size": [0.086, 0.086, 0.088], "tgt_size": [0.170, 0.170, 0.048], "is_hollow": 0.0,
        "pick_z_off": 0.065, "place_z_off": 0.095, "table_z_default": -0.015,
        "outcome": {"type": "in_container", "max_xy_m": 0.095, "z_range": [-0.02, 0.12]},
    },
    {
        "tid": 1, "name": "demo7_gupta_stack", "label": "2. GuptaLab | Stack Orange Block on Green Block",
        "kind": "universal", "cfg": "configs/scenes/demo7_gupta_stack.json",
        "src_body": "orange_block", "tgt_body": "green_block",
        "src_size": [0.044, 0.044, 0.044], "tgt_size": [0.046, 0.046, 0.046], "is_hollow": 0.0,
        "pick_z_off": 0.024, "place_z_off": 0.068,
    },
    {
        "tid": 2, "name": "demo8_auto_duck", "label": "3. AUTOLab | Orange Rubber Duck -> Silver Pot",
        "kind": "universal", "cfg": "configs/scenes/demo8_auto_duck.json",
        "src_body": "rubber_duck", "tgt_body": "silver_pot",
        "src_size": [0.058, 0.048, 0.052], "tgt_size": [0.122, 0.122, 0.072], "is_hollow": 0.0,
        "pick_z_off": 0.028, "place_z_off": 0.088,
    },
    {
        "tid": 3, "name": "demo9_pennpal_tape", "label": "4. PennPAL | Hollow Black Tape -> White Plate",
        "kind": "universal", "cfg": "configs/scenes/demo9_pennpal_tape.json",
        "src_body": "black_tape", "tgt_body": "white_plate",
        "src_size": [0.094, 0.094, 0.048], "tgt_size": [0.205, 0.205, 0.016], "is_hollow": 1.0,
        "pick_z_off": 0.026, "place_z_off": 0.048,
    },
    {
        "tid": 4, "name": "demo10_rpl_can", "label": "5. RPL Kitchen | Mini Soda Can -> Bowl",
        "kind": "universal", "cfg": "configs/scenes/demo10_rpl_can.json",
        "src_body": "soda_can", "tgt_body": "target_bowl",
        "src_size": [0.056, 0.056, 0.104], "tgt_size": [0.145, 0.145, 0.068], "is_hollow": 0.0,
        "pick_z_off": 0.052, "place_z_off": 0.095,
    },
]


def pad_dist_to_g_state(pad_dist: float) -> float:
    return float(np.clip((0.0983 - pad_dist) / (0.0983 - 0.0149), 0.0, 1.0))


def task_geom_vec(tspec: dict) -> np.ndarray:
    s = tspec["src_size"]; t = tspec["tgt_size"]
    return np.array([s[0] * 10.0, s[1] * 10.0, s[2] * 10.0, t[0] * 10.0, t[1] * 10.0, t[2] * 10.0, float(tspec["is_hollow"])], dtype=np.float32)


def make_straight_line_prior_np(p_ee: np.ndarray, p_pick_vlm: np.ndarray, p_place_vlm: np.ndarray, g_st: float) -> np.ndarray:
    """ZERO intermediate waypoints: dumb straight-line chord from live p_ee toward active noisy VLM click!"""
    w = float(np.clip((g_st - 0.20) / 0.40, 0.0, 1.0))
    p_goal = (1.0 - w) * p_pick_vlm + w * p_place_vlm
    d = p_goal - p_ee
    dist = float(np.linalg.norm(d))
    u = d / max(dist, 1e-6)
    out = np.zeros((HORIZON, ACT_DIM), dtype=np.float32)
    for k in range(HORIZON):
        s = min((k + 1) * 0.018, dist)
        out[k, :3] = (s * u).astype(np.float32)
        out[k, 6] = float(g_st)
    return out


def build_obs_np(
    p_ee: np.ndarray, R_ee: np.ndarray, g_st: float, ee_vel: np.ndarray,
    p_src: np.ndarray, p_tgt: np.ndarray, p_pick_vlm: np.ndarray, p_place_vlm: np.ndarray,
    table_z: float, geom_vec: np.ndarray,
) -> np.ndarray:
    def dir_sat(vec: np.ndarray, scale: float) -> np.ndarray:
        n = float(np.linalg.norm(vec)); u = vec / max(n, 1e-5)
        n_xy = float(np.linalg.norm(vec[:2]))
        return np.array([u[0], u[1], u[2], math.tanh(n / scale), math.tanh(n_xy / scale), float(np.clip(vec[2], -0.25, 0.25))], dtype=np.float32)

    d_src = p_src - p_ee; d_tgt = p_tgt - p_ee; d_st = p_src - p_tgt
    t_vp = dir_sat(p_pick_vlm - p_ee, 0.15)                                                           # 6
    t_vb = dir_sat(p_place_vlm - p_ee, 0.15)                                                          # 6
    t_ls = np.concatenate([dir_sat(d_src, 0.12), [float(p_src[2] - table_z)]])                        # 7
    t_lt = np.concatenate([dir_sat(d_tgt, 0.12), [math.tanh(float(np.linalg.norm(d_st[:2])) / 0.12), float(np.clip(d_st[2], -0.2, 0.2))]])  # 8
    z_ax = R_ee[:, 2]
    is_hol = float(geom_vec[6])
    d_s_xy = float(np.linalg.norm(d_src[:2]))
    held_cue = float(g_st * math.exp(-((d_s_xy / (0.055 + 0.035 * is_hol)) ** 2)))
    t_ee = np.array([
        float(p_ee[2] - table_z), z_ax[0], z_ax[1], z_ax[2],
        float(g_st), ee_vel[0], ee_vel[1], ee_vel[2], held_cue,
    ], dtype=np.float32)                                                                              # 9
    return np.concatenate([t_vp, t_vb, t_ls, t_lt, t_ee, geom_vec], axis=0)                           # 43


# ---------------------------------------------------------------------------
# Stage 1: Collect Exact MuJoCo 4.0.1 Real State-Action Dataset Across All 5 Tasks (4 Labs)
# ---------------------------------------------------------------------------
def _replay_universal_ep(args: tuple) -> dict | None:
    import mujoco
    from scipy.optimize import minimize
    from r2s_bench.eval.task_success import RolloutMonitor, evaluate_object_in_container, evaluate_object_on_target
    from r2s_bench.grading.l1_visual import compute_mask_iou
    from scripts.remote_universal_stage2_mj4 import (
        build_universal_scene, configure_cam, extract_grasp, make_calib, ray_plane_z, OPT_WH,
    )

    tspec, ep, cfg, vis_paths, s1_colors, fallback_intr, ep_masks = args
    ei = ep["ep_idx"]; bundle = Path(cfg["bundle_dir"])
    tr = np.load(bundle / "episodes" / f"ep_{ei:03d}" / "trajectory.npz")
    qs, gs = tr["qpos"], tr["gripper"]
    c1 = make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"], fallback_intr["c1"])
    c2 = make_calib(ep["c2_serial"], ep["c2_extr"], ep["c2_intr"], fallback_intr["c2"])

    src_cfg, tgt_cfg = cfg["objects"][0], cfg["objects"][1]
    src_name, tgt_name = src_cfg["name"], tgt_cfg["name"]
    ev = extract_grasp(qs, gs, pinch_frac=float(cfg.get("pinch_frac", 0.62)))
    pc, po = ev["p_close"], ev["p_open"]
    grasp_h = float(src_cfg.get("grasp_height_m", 0.5 * src_cfg["size_xyz"][2]))
    tz = float(np.clip(pc[2] - grasp_h, -0.18, 0.35))

    obj_rgbs = {o["name"]: s1_colors["ep_rgbs"].get(o["name"], {}).get(str(ei), s1_colors["med_rgbs"].get(o["name"], o.get("default_rgb", [0.6, 0.6, 0.6]))) for o in cfg["objects"]}
    obj_xys = {}
    anchors = {src_name: pc[:2], tgt_name: po[:2]}
    for o in cfg["objects"]:
        nm = o["name"]; m1, m2 = ep_masks[f"c1_{nm}"], ep_masks[f"c2_{nm}"]
        has1, has2 = int(m1.sum()) >= 80, int(m2.sum()) >= 80
        anc = anchors[nm]; xy0 = anc.copy()
        if has1:
            y1, x1 = np.where(m1)
            cand = ray_plane_z(c1.scaled(OUT_WH[0], OUT_WH[1]), (float(np.median(x1)), float(np.median(y1))), tz + 0.5 * o["size_xyz"][2])[:2]
            if float(np.linalg.norm(cand - anc)) <= 0.16: xy0 = cand
        vis_xy = (float(xy0[0]), float(xy0[1]))
        if has1 or has2:
            tmp_cfg = dict(cfg, objects=[o])
            m_o = build_universal_scene(tmp_cfg, {nm: vis_xy}, {nm: 0.0}, {nm: obj_rgbs[nm]}, {nm: vis_paths.get(nm)}, tz, with_robot=False, offwidth=OPT_WH[0], offheight=OPT_WH[1])
            d_o = mujoco.MjData(m_o); bid = mujoco.mj_name2id(m_o, mujoco.mjtObj.mjOBJ_BODY, nm)
            gids = [g for g in range(m_o.ngeom) if m_o.geom_bodyid[g] == bid]
            ren_o = mujoco.Renderer(m_o, OPT_WH[1], OPT_WH[0]); ren_o.enable_segmentation_rendering()
            rm1 = cv2.resize(m1.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
            rm2 = cv2.resize(m2.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
            def ev_xy(x):
                if abs(x[0] - xy0[0]) > 0.12 or abs(x[1] - xy0[1]) > 0.12: return 0.0
                d_o.qpos[:3] = [float(x[0]), float(x[1]), tz + 0.001]; d_o.qpos[3:7] = [1, 0, 0, 0]
                s, ws = 0.0, 0.0
                if has1:
                    configure_cam(m_o, 0, c1, OPT_WH[0], OPT_WH[1]); mujoco.mj_forward(m_o, d_o); ren_o.update_scene(d_o, camera=0)
                    s += 0.75 * compute_mask_iou(rm1, np.isin(ren_o.render()[:, :, 0], gids)); ws += 0.75
                if has2:
                    configure_cam(m_o, 0, c2, OPT_WH[0], OPT_WH[1]); mujoco.mj_forward(m_o, d_o); ren_o.update_scene(d_o, camera=0)
                    s += 0.25 * compute_mask_iou(rm2, np.isin(ren_o.render()[:, :, 0], gids)); ws += 0.25
                return s / max(ws, 1e-6)
            r_opt = minimize(lambda x: 1.0 - ev_xy(x), xy0, method="Nelder-Mead", options={"maxiter": 34, "xatol": 2e-3, "fatol": 2e-3})
            vis_xy = (float(r_opt.x[0]), float(r_opt.x[1])); ren_o.close()
        min_xy_dim = float(min(o["size_xyz"][0], o["size_xyz"][1]))
        if o.get("grasp_mode") == "rim_pinch":
            r_out = 0.5 * max(o["size_xyz"][0], o["size_xyz"][1]); wall = float(o.get("wall_m", 0.024))
            r_mid = max(r_out - 0.5 * wall, 0.022)
            ax = ev["axis_grasp"][:2]; ax = ax / max(float(np.linalg.norm(ax)), 1e-6)
            sgn = 1.0 if float(np.dot(np.array(vis_xy) - pc[:2], ax)) >= 0.0 else -1.0
            obj_xys[nm] = (float(pc[0] + sgn * r_mid * ax[0]), float(pc[1] + sgn * r_mid * ax[1]))
        else:
            default_dev = min(0.28 * min_xy_dim, 0.026) if nm == src_name else min(0.32 * min_xy_dim, 0.055)
            max_dev = float(o.get("anchor_max_dev_m", default_dev))
            obj_xys[nm] = vis_xy if float(np.linalg.norm(np.array(vis_xy) - anc)) <= max_dev else (float(anc[0]), float(anc[1]))
    if float(np.linalg.norm(np.array(obj_xys[src_name]) - np.array(obj_xys[tgt_name]))) < 0.06:
        obj_xys[tgt_name] = (float(po[0]), float(po[1]))

    m = build_universal_scene(cfg, obj_xys, {src_name: ev["yaw_grasp"], tgt_name: 0.0}, obj_rgbs, vis_paths, tz, table_tex=None, with_robot=True, offwidth=64, offheight=64)
    d = mujoco.MjData(m)
    lp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_left_silicone_pad")
    rp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_right_silicone_pad")
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base_mount")
    s_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, src_name)
    t_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tgt_name)
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")

    d.qpos[:7] = qs[0]; d.ctrl[:7] = qs[0]; d.ctrl[ag] = float(np.clip(gs[0], 0, 1)) * 255.0
    mujoco.mj_forward(m, d)
    for _ in range(30): mujoco.mj_step(m, d)

    T = len(qs); hold_ctrl = float(cfg.get("hold_ctrl", 0.82))
    p_ees = np.zeros((T, 3), dtype=np.float32); R_ees = np.zeros((T, 3, 3), dtype=np.float32)
    g_states = np.zeros((T,), dtype=np.float32); g_cmds = np.zeros((T,), dtype=np.float32)
    p_srcs = np.zeros((T, 3), dtype=np.float32); p_tgts = np.zeros((T, 3), dtype=np.float32)

    mon = RolloutMonitor(m, target_body_name=src_name); mon.reset(d)
    for ti in range(T):
        gc = hold_ctrl if (max(0, ev["t_close"] - 1) <= ti < ev["t_open"]) else float(np.clip(gs[ti], 0, 1))
        p_ees[ti] = 0.5 * (d.xpos[lp] + d.xpos[rp]); R_ees[ti] = d.xmat[hb].reshape(3, 3)
        g_states[ti] = pad_dist_to_g_state(float(np.linalg.norm(d.xpos[lp] - d.xpos[rp])))
        g_cmds[ti] = gc; p_srcs[ti] = d.xpos[s_bid]; p_tgts[ti] = d.xpos[t_bid]
        d.ctrl[:7] = qs[min(ti + 1, T - 1)]; d.ctrl[ag] = gc * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0: mon.observe(d)
    for _ in range(50):
        d.ctrl[:7] = qs[-1]; d.ctrl[ag] = float(np.clip(gs[-1], 0, 1)) * 255.0; mujoco.mj_step(m, d)
    mon.observe(d)

    oc = cfg["outcome"]
    if oc["type"] == "in_container":
        res = evaluate_object_in_container(m, d, mon, src_name, tgt_name, max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [-0.02, 0.12]), table_z=tz)
    else:
        res = evaluate_object_on_target(m, d, mon, src_name, tgt_name, max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [0.005, 0.14]))
    if not res.success:
        return None
    return {
        "tid": tspec["tid"], "ep_idx": ei, "table_z": tz, "q_home": qs[0].astype(np.float32),
        "src_xy0": np.array(obj_xys[src_name], dtype=np.float32), "tgt_xy0": np.array(obj_xys[tgt_name], dtype=np.float32),
        "p_ee": p_ees, "R_ee": R_ees, "g_state": g_states, "g_cmd": g_cmds, "p_src": p_srcs, "p_tgt": p_tgts,
    }


def _demos_to_transitions(demos: List[dict], tspec: dict) -> dict:
    gv = task_geom_vec(tspec)
    p_ee_l, R_ee_l, g_st_l, vel_l, p_src_l, p_tgt_l, tz_l, tid_l, act_l, gv_l = [], [], [], [], [], [], [], [], [], []
    for dm in demos:
        p_ee, R_ee, g_st, g_cmd = dm["p_ee"], dm["R_ee"], dm["g_state"], dm["g_cmd"]
        p_src, p_tgt = dm["p_src"], dm["p_tgt"]
        T = len(p_ee); tz = float(dm["table_z"])
        spds = np.linalg.norm(np.diff(p_ee, axis=0, prepend=p_ee[:1]), axis=-1)
        t_start = max(0, int(np.argmax(spds > 0.002)) - 2)
        for t in range(t_start, T - 2):
            tp = max(0, t - 1)
            vel = (p_ee[t] - p_ee[tp]) / DT_CTRL
            act = np.zeros((HORIZON, ACT_DIM), dtype=np.float32)
            for k in range(HORIZON):
                tk = min(t + k + 1, T - 1)
                act[k, :3] = p_ee[tk] - p_ee[t]
                act[k, 3:6] = Rotation.from_matrix(R_ee[tk] @ R_ee[t].T).as_rotvec()
                act[k, 6] = g_cmd[tk]
            p_ee_l.append(p_ee[t]); R_ee_l.append(R_ee[t]); g_st_l.append(g_st[t]); vel_l.append(vel)
            p_src_l.append(p_src[t]); p_tgt_l.append(p_tgt[t]); tz_l.append(tz); tid_l.append(tspec["tid"])
            act_l.append(act); gv_l.append(gv)
    return {
        "p_ee": np.asarray(p_ee_l, dtype=np.float32), "R_ee": np.asarray(R_ee_l, dtype=np.float32),
        "g_st": np.asarray(g_st_l, dtype=np.float32), "vel": np.asarray(vel_l, dtype=np.float32),
        "p_src": np.asarray(p_src_l, dtype=np.float32), "p_tgt": np.asarray(p_tgt_l, dtype=np.float32),
        "table_z": np.asarray(tz_l, dtype=np.float32), "tid": np.asarray(tid_l, dtype=np.int64),
        "act": np.asarray(act_l, dtype=np.float32), "geom": np.asarray(gv_l, dtype=np.float32),
    }


def stage_collect() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    parts = []
    task_meta = {}

    # Task 0: Reuse exact cached Demo #1 dataset (50 passing real AUTOLab episodes -> 5103 samples!)
    d1 = np.load(DEMO1_DIT_DATA)
    n1 = len(d1["act"])
    gv0 = task_geom_vec(TASK_SPECS[0])
    parts.append({
        "p_ee": d1["p_ee"], "R_ee": d1["R_ee"], "g_st": d1["g_st"], "vel": d1["vel"],
        "p_src": d1["p_cup"], "p_tgt": d1["p_bowl"],
        "table_z": np.full((n1,), -0.015, dtype=np.float32),
        "tid": np.zeros((n1,), dtype=np.int64),
        "act": d1["act"], "geom": np.repeat(gv0[None, :], n1, axis=0),
    })
    task_meta["0"] = {
        "n_real_demos": int(d1["n_demos"]), "n_samples": n1,
        "q_home": d1["q_home"].tolist(), "table_z_med": -0.015,
        "xy_bounds": [0.41, 0.69, -0.25, 0.25], "dist_range": [0.17, 0.40],
    }
    print(f"[Collect Task 0 demo1_cup_bowl] Reused {int(d1['n_demos'])} real demos -> {n1} chunks", flush=True)

    # Tasks 1..4: Replay verified passing real episodes of Demo #7, #8, #9, #10 in MuJoCo 4.0.1!
    ctx = mp.get_context("spawn")
    n_w = int(os.environ.get("R2S_WORKERS", "12"))
    for tspec in TASK_SPECS[1:]:
        tid = tspec["tid"]
        cfg = json.loads((ROOT / tspec["cfg"]).read_text())
        sid = cfg["scene_id"]; bundle = Path(cfg["bundle_dir"]); asset_dir = Path(cfg["asset_dir"])
        meta = json.loads((bundle / "meta.json").read_text())
        masks = dict(np.load(bundle / "masks.npz"))
        s1_colors = json.loads((asset_dir / "stage1_colors.json").read_text())
        res_prev = json.loads((asset_dir / f"{sid}_results.json").read_text())
        pass_ids = [e["ep_idx"] for e in res_prev["episodes"] if e["success"]]
        fb_c1 = next(m["c1_intr"] for m in meta if m.get("c1_intr") and int(m["c1_intr"].get("width", 0)) > 0)
        fb_c2 = next(m["c2_intr"] for m in meta if m.get("c2_intr") and int(m["c2_intr"].get("width", 0)) > 0)
        fallback_intr = {"c1": fb_c1, "c2": fb_c2}
        vis_paths = {}
        for o in cfg["objects"]:
            mj = asset_dir / f"sim_ready_{o['name']}" / "asset_meta.json"
            if mj.exists():
                vis_paths[o["name"]] = Path(json.loads(mj.read_text())["visual_mesh_path"])

        meta_by_id = {m["ep_idx"]: m for m in meta}
        tasks = []
        for ei in pass_ids:
            ep = meta_by_id[ei]
            ep_m = {f"{cam}_{o['name']}": masks[f"ep{ei:03d}_{cam}_{o['name']}"] for o in cfg["objects"] for cam in ("c1", "c2")}
            tasks.append((tspec, ep, cfg, vis_paths, s1_colors, fallback_intr, ep_m))

        demos = []
        with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
            for f in as_completed([pool.submit(_replay_universal_ep, t) for t in tasks]):
                r = f.result()
                if r is not None: demos.append(r)

        demos.sort(key=lambda x: x["ep_idx"])
        part = _demos_to_transitions(demos, tspec)
        parts.append(part)
        all_xy = np.vstack([dm["src_xy0"] for dm in demos] + [dm["tgt_xy0"] for dm in demos])
        x_lo = float(np.clip(np.percentile(all_xy[:, 0], 5) - 0.06, 0.36, 0.50))
        x_hi = float(np.clip(np.percentile(all_xy[:, 0], 95) + 0.06, 0.58, 0.72))
        y_lo = float(np.clip(np.percentile(all_xy[:, 1], 5) - 0.08, -0.27, -0.08))
        y_hi = float(np.clip(np.percentile(all_xy[:, 1], 95) + 0.08, 0.08, 0.27))
        tz_med = float(np.median([dm["table_z"] for dm in demos]))
        q_home = np.median(np.stack([dm["q_home"] for dm in demos], axis=0), axis=0)
        # Ensure min separation clears both object footprints cleanly
        min_sep = max(0.14, 0.5 * (max(tspec["src_size"][:2]) + max(tspec["tgt_size"][:2])) + 0.045)
        task_meta[str(tid)] = {
            "n_real_demos": len(demos), "n_samples": len(part["act"]),
            "q_home": q_home.tolist(), "table_z_med": tz_med,
            "xy_bounds": [round(x_lo, 3), round(x_hi, 3), round(y_lo, 3), round(y_hi, 3)],
            "dist_range": [round(min_sep, 3), round(min(0.36, min_sep + 0.20), 3)],
            "ref_ep_idx": int(demos[0]["ep_idx"]),
        }
        print(f"[Collect Task {tid} {sid}] Replayed {len(demos)}/{len(pass_ids)} real demos -> {len(part['act'])} chunks | tz={tz_med*100:.1f}cm | xy={task_meta[str(tid)]['xy_bounds']}", flush=True)

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}
    np.savez_compressed(OUT_DIR / "uni_real_dataset.npz", **merged)
    (OUT_DIR / "task_meta.json").write_text(json.dumps(task_meta, indent=2))
    tot_demos = sum(v["n_real_demos"] for v in task_meta.values())
    print(f"[Collect ALL 5 TASKS COMPLETE] {tot_demos} real passing demos across 4 labs -> {len(merged['act'])} chunks in {time.perf_counter()-t0:.1f}s!", flush=True)


# ---------------------------------------------------------------------------
# Unified 6-Token QK-Norm TF32 Bridge-Flow DiT Action Render (~10M params)
# ---------------------------------------------------------------------------
def build_uni_bridge_dit():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SinEmb(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
            a = t[:, None] * freqs[None, :] * 100.0
            return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

    class QKNormBridgeBlock(nn.Module):
        def __init__(self, d_model: int = 256, n_heads: int = 8):
            super().__init__()
            self.n_heads = n_heads; self.d_head = d_model // n_heads
            self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.q_norm = nn.LayerNorm(self.d_head, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.d_head, eps=1e-6)
            self.sa_out = nn.Linear(d_model, d_model)
            self.norm_ca_q = nn.LayerNorm(d_model, eps=1e-6)
            self.norm_ca_kv = nn.LayerNorm(d_model, eps=1e-6)
            self.ca_q = nn.Linear(d_model, d_model)
            self.ca_kv = nn.Linear(d_model, 2 * d_model)
            self.ca_q_norm = nn.LayerNorm(self.d_head, eps=1e-6)
            self.ca_k_norm = nn.LayerNorm(self.d_head, eps=1e-6)
            self.ca_out = nn.Linear(d_model, d_model)
            self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model))
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            nn.init.zeros_(self.adaLN[-1].weight); nn.init.zeros_(self.adaLN[-1].bias)

        def forward(self, x: torch.Tensor, obs_toks: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            B, H, D = x.shape; Ns = obs_toks.shape[1]
            s1, sc1, g1, s2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)
            h = self.norm1(x) * (1.0 + sc1[:, None, :]) + s1[:, None, :]
            qkv = self.qkv(h).reshape(B, H, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
            out_sa = F.scaled_dot_product_attention(self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]).transpose(1, 2).reshape(B, H, D)
            x = x + g1[:, None, :] * self.sa_out(out_sa)

            q = self.ca_q_norm(self.ca_q(self.norm_ca_q(x)).reshape(B, H, self.n_heads, self.d_head).transpose(1, 2))
            kv = self.ca_kv(self.norm_ca_kv(obs_toks)).reshape(B, Ns, 2, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
            out_ca = F.scaled_dot_product_attention(q, self.ca_k_norm(kv[0]), kv[1]).transpose(1, 2).reshape(B, H, D)
            x = x + self.ca_out(out_ca)

            h2 = self.norm2(x) * (1.0 + sc2[:, None, :]) + s2[:, None, :]
            x = x + g2[:, None, :] * self.mlp(h2)
            return x

    class UniGoalBridgeDiT(nn.Module):
        def __init__(self, d_model: int = 256, n_layers: int = 6, n_heads: int = 8):
            super().__init__()
            # 6 structured physical tokens (ZERO task-ID hardcoding! Pure physical geometry + relative state!)
            self.enc_vp = nn.Sequential(nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_vb = nn.Sequential(nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_ls = nn.Sequential(nn.Linear(7, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_lt = nn.Sequential(nn.Linear(8, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_ee = nn.Sequential(nn.Linear(9, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_gm = nn.Sequential(nn.Linear(7, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.obs_pos = nn.Parameter(0.02 * torch.randn(1, 6, d_model))
            self.obs_pool = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.time_emb = nn.Sequential(SinEmb(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.act_in = nn.Linear(ACT_DIM * 2, d_model)
            self.act_pos = nn.Parameter(0.02 * torch.randn(1, HORIZON, d_model))
            self.blocks = nn.ModuleList([QKNormBridgeBlock(d_model, n_heads) for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
            self.act_out = nn.Linear(d_model, ACT_DIM)
            nn.init.zeros_(self.final_ada[-1].weight); nn.init.zeros_(self.final_ada[-1].bias)
            nn.init.zeros_(self.act_out.weight); nn.init.zeros_(self.act_out.bias)

        def encode_obs(self, obs: torch.Tensor):
            toks = torch.stack([
                self.enc_vp(obs[:, 0:6]),
                self.enc_vb(obs[:, 6:12]),
                self.enc_ls(obs[:, 12:19]),
                self.enc_lt(obs[:, 19:27]),
                self.enc_ee(obs[:, 27:36]),
                self.enc_gm(obs[:, 36:43]),
            ], dim=1) + self.obs_pos
            return toks, self.obs_pool(toks.mean(dim=1))

        def forward(self, a_tau: torch.Tensor, a_line: torch.Tensor, tau: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
            obs_toks, g_obs = self.encode_obs(obs)
            cond = self.time_emb(tau) + g_obs
            x = self.act_in(torch.cat([a_tau, a_line], dim=-1)) + self.act_pos
            for blk in self.blocks:
                x = blk(x, obs_toks, cond)
            sh, sc = self.final_ada(cond).chunk(2, dim=-1)
            x = self.final_norm(x) * (1.0 + sc[:, None, :]) + sh[:, None, :]
            return self.act_out(x)

        @torch.no_grad()
        def render_chunk(self, a_line_norm: torch.Tensor, obs: torch.Tensor, n_steps: int = ODE_STEPS, seed: int | None = None) -> torch.Tensor:
            B = obs.shape[0]
            if seed is not None:
                gen = torch.Generator(device=obs.device).manual_seed(seed)
                eps = torch.randn((B, HORIZON, ACT_DIM), device=obs.device, dtype=torch.float32, generator=gen)
            else:
                eps = torch.randn((B, HORIZON, ACT_DIM), device=obs.device, dtype=torch.float32)
            a = a_line_norm + BRIDGE_SIGMA * eps
            obs_toks, g_obs = self.encode_obs(obs)
            dt = 1.0 / n_steps
            for i in range(n_steps):
                tau = torch.full((B,), i * dt, device=obs.device, dtype=torch.float32)
                cond = self.time_emb(tau) + g_obs
                x = self.act_in(torch.cat([a, a_line_norm], dim=-1)) + self.act_pos
                for blk in self.blocks:
                    x = blk(x, obs_toks, cond)
                sh, sc = self.final_ada(cond).chunk(2, dim=-1)
                x = self.final_norm(x) * (1.0 + sc[:, None, :]) + sh[:, None, :]
                a = a + dt * self.act_out(x)
            return a

    return UniGoalBridgeDiT()


# ---------------------------------------------------------------------------
# Stage 2: Train Unified Uni-Goal-Bridge DiT (Round 1: Real-Only | Round 2: Sim-Bootstrapped!)
# ---------------------------------------------------------------------------
def stage_train(round_idx: int) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_ds = dict(np.load(OUT_DIR / "uni_real_dataset.npz"))
    if round_idx == 2 and (OUT_DIR / "uni_sim_harvest.npz").exists():
        sim_ds = dict(np.load(OUT_DIR / "uni_sim_harvest.npz"))
        print(f"[Uni-Train R2] Scaling with Real ({len(real_ds['act'])}) + Sim-Harvested OOD ({len(sim_ds['act'])}) chunks!", flush=True)
        ds = {k: np.concatenate([real_ds[k], sim_ds[k]], axis=0) for k in real_ds.keys()}
        n_real = len(real_ds["act"])
        is_sim_flag = np.concatenate([np.zeros(n_real, dtype=bool), np.ones(len(sim_ds["act"]), dtype=bool)])
    else:
        ds = real_ds
        is_sim_flag = np.zeros(len(ds["act"]), dtype=bool)
        print(f"[Uni-Train R1] Training on Real-Only 5-Task Multi-Lab Dataset ({len(ds['act'])} chunks)", flush=True)

    p_ee = torch.from_numpy(ds["p_ee"]).to(device)
    R_ee = torch.from_numpy(ds["R_ee"]).to(device)
    g_st = torch.from_numpy(ds["g_st"]).to(device)
    vel = torch.from_numpy(ds["vel"]).to(device)
    p_src = torch.from_numpy(ds["p_src"]).to(device)
    p_tgt = torch.from_numpy(ds["p_tgt"]).to(device)
    tz_t = torch.from_numpy(ds["table_z"]).to(device)
    geom_t = torch.from_numpy(ds["geom"]).to(device)
    act = torch.from_numpy(ds["act"]).to(device)
    tid_np = ds["tid"]

    # Pre-index per task (and real vs sim in Round 2) for strictly balanced multi-task batches!
    task_pools_real = [torch.from_numpy(np.where((tid_np == t) & (~is_sim_flag))[0]).to(device) for t in range(5)]
    task_pools_sim = [torch.from_numpy(np.where((tid_np == t) & is_sim_flag)[0]).to(device) for t in range(5)]
    pick_z_lut = torch.tensor([ts["pick_z_off"] for ts in TASK_SPECS], device=device, dtype=torch.float32)
    place_z_lut = torch.tensor([ts["place_z_off"] for ts in TASK_SPECS], device=device, dtype=torch.float32)
    tid_t = torch.from_numpy(tid_np).to(device)
    step_ladder = (torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32) * 0.018)[None, :, None]

    def sample_balanced_idx(bs: int) -> torch.Tensor:
        per_t = bs // 5
        chunks = []
        for t in range(5):
            pr = task_pools_real[t]
            ps = task_pools_sim[t]
            if round_idx == 2 and len(ps) >= 32:
                nr = per_t // 2; ns = per_t - nr
                chunks.append(pr[torch.randint(0, len(pr), (nr,), device=device)])
                chunks.append(ps[torch.randint(0, len(ps), (ns,), device=device)])
            else:
                chunks.append(pr[torch.randint(0, len(pr), (per_t,), device=device)])
        return torch.cat(chunks, dim=0)

    def make_batch(idx: torch.Tensor):
        B = len(idx)
        tz_b = tz_t[idx]
        tids = tid_t[idx]
        g_b = geom_t[idx]
        is_hol = g_b[:, 6]

        # Simulate realistic +-2.5cm VLM click errors on source & target anchors!
        vlm_ec = torch.cat([torch.randn((B, 2), device=device) * 0.022, torch.randn((B, 1), device=device) * 0.010], dim=-1).clamp(-0.042, 0.042)
        vlm_eb = torch.cat([torch.randn((B, 2), device=device) * 0.022, torch.randn((B, 1), device=device) * 0.010], dim=-1).clamp(-0.042, 0.042)
        p_pick_vlm = torch.stack([p_src[idx, 0], p_src[idx, 1], tz_b + pick_z_lut[tids]], dim=-1) + vlm_ec
        p_place_vlm = torch.stack([p_tgt[idx, 0], p_tgt[idx, 1], tz_b + place_z_lut[tids]], dim=-1) + vlm_eb

        # Mild closed-loop position jitter (+-7mm) for active recovery!
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

        # Exact SO(2) 360-degree planar yaw augmentation across all 5 labs!
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

        # Trivial Straight-Line Bridge Prior A_line (ZERO intermediate waypoints!)
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
        t_ls = torch.cat([dir_sat_t(d_src, 0.12), (p_src_q[:, 2] - tz_b).unsqueeze(-1)], dim=-1)
        t_lt = torch.cat([dir_sat_t(d_tgt, 0.12), torch.tanh(torch.norm(d_st[:, :2], dim=-1, keepdim=True) / 0.12), d_st[:, 2:3].clamp(-0.2, 0.2)], dim=-1)
        d_s_xy_n = torch.norm(d_src[:, :2], dim=-1)
        held_cue = (g_st[idx] * torch.exp(-((d_s_xy_n / (0.055 + 0.035 * is_hol)) ** 2))).unsqueeze(-1)
        t_ee = torch.cat([(p_ee_q[:, 2] - tz_b).unsqueeze(-1), z_ax, g_st[idx].unsqueeze(-1), v_cur, held_cue], dim=-1)
        obs_b = torch.cat([t_vp, t_vb, t_ls, t_lt, t_ee, g_b], dim=-1)
        return obs_b, a_line, a_b

    model = build_uni_bridge_dit().to(device)
    if round_idx == 2 and (OUT_DIR / "uni_bridge_r1.pt").exists():
        ck1 = torch.load(OUT_DIR / "uni_bridge_r1.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck1["ema_state_dict"])
        act_mean = torch.from_numpy(ck1["act_mean"]).to(device)
        act_std = torch.from_numpy(ck1["act_std"]).to(device)
        n_steps = 3000; lr_max = 1.8e-4
    else:
        with torch.no_grad():
            _, _, a_stat = make_batch(sample_balanced_idx(5000))
            act_mean = a_stat.mean(dim=(0, 1))
            act_std = a_stat.std(dim=(0, 1)).clamp(min=0.03)
        n_steps = 5500; lr_max = 3.0e-4

    ema_model = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Uni-Goal-Bridge R{round_idx}] {n_params/1e6:.2f}M params | {n_steps} steps | TF32+QK-Norm+SDPA+FusedAdamW", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-4, fused=True)
    warmup = 200; bs = 500
    def lr_fn(s):
        if s < warmup: return float(s + 1) / float(warmup)
        p = float(s - warmup) / float(max(1, n_steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    beta_dist = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
    ema_params = list(ema_model.parameters()); cur_params = list(model.parameters())
    w_dim = torch.tensor([1.3, 1.3, 1.5, 0.7, 0.7, 0.7, 2.2], device=device)[None, None, :]

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        with torch.no_grad():
            obs_b, al_b, ae_b = make_batch(sample_balanced_idx(bs))
            al_n = ((al_b - act_mean) / act_std).clamp(-6.0, 6.0)
            a1 = ((ae_b - act_mean) / act_std).clamp(-6.0, 6.0)
            a0 = al_n + BRIDGE_SIGMA * torch.randn_like(a1)
            tau = beta_dist.sample((bs,)).clamp(1e-3, 1.0 - 1e-3)
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
        if step % 1000 == 0 or step == 1 or step == n_steps:
            print(f"  [Uni R{round_idx} step {step:4d}/{n_steps}] loss={loss.item():.4f} | {(time.perf_counter()-t0)/step*1000:.1f} ms/step ({time.perf_counter()-t0:.1f}s)", flush=True)

    train_dt = time.perf_counter() - t0
    with torch.no_grad():
        obs_v, al_v, ae_v = make_batch(sample_balanced_idx(500))
        pred = ema_model.render_chunk(((al_v - act_mean) / act_std).clamp(-6, 6), obs_v, n_steps=ODE_STEPS, seed=42) * act_std + act_mean
        mae_cm = torch.mean(torch.abs(pred[:, :6, :3] - ae_v[:, :6, :3])).item() * 100.0
        g_mae = torch.mean(torch.abs(pred[:, :6, 6] - ae_v[:, :6, 6])).item()
        print(f"[Uni R{round_idx} COMPLETE] {train_dt:.1f}s | 4-step ODE PosMAE={mae_cm:.2f}cm | GripMAE={g_mae:.3f}", flush=True)

    torch.save({
        "ema_state_dict": ema_model.state_dict(),
        "act_mean": act_mean.cpu().numpy(), "act_std": act_std.cpu().numpy(),
        "train_sec": train_dt, "round": round_idx,
    }, OUT_DIR / f"uni_bridge_r{round_idx}.pt")


# ---------------------------------------------------------------------------
# Stage 3A: Persistent-Connection GPU Policy Server (4-Step Uni-Goal-Bridge DiT)
# ---------------------------------------------------------------------------
def stage_serve(round_idx: int) -> None:
    import threading
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(OUT_DIR / f"uni_bridge_r{round_idx}.pt", map_location=device, weights_only=False)
    model = build_uni_bridge_dit().to(device).eval()
    model.load_state_dict(ckpt["ema_state_dict"])
    act_mean = torch.from_numpy(ckpt["act_mean"]).to(device)
    act_std = torch.from_numpy(ckpt["act_std"]).to(device)

    with torch.no_grad():
        _ = model.render_chunk(torch.zeros((1, HORIZON, ACT_DIM), device=device), torch.zeros((1, OBS_DIM), device=device), n_steps=ODE_STEPS)

    lock = threading.Lock()
    stop_flag = threading.Event()
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
                    stop_flag.set()
                    return
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
    srv.settimeout(0.5)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(32)
    print(f"[Uni-Bridge R{round_idx} Server] READY on 127.0.0.1:{PORT} (Persistent TCP, 4-step ODE)", flush=True)
    try:
        while not stop_flag.is_set():
            try:
                conn, _ = srv.accept()
                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
    finally:
        srv.close()
        print(f"[Uni-Bridge R{round_idx} Server] Shutdown cleanly.", flush=True)


class PersistentBridgeClient:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("127.0.0.1", PORT))

    def query(self, obs: np.ndarray, a_line: np.ndarray, seed: int) -> np.ndarray:
        payload = struct.pack("<i", int(seed)) + np.asarray(obs, dtype=np.float32).tobytes() + np.asarray(a_line, dtype=np.float32).tobytes()
        self.s.sendall(payload)
        need = HORIZON * ACT_DIM * 4
        buf = b""
        while len(buf) < need:
            c = self.s.recv(need - len(buf))
            if not c: break
            buf += c
        return np.frombuffer(buf, dtype=np.float32).reshape(HORIZON, ACT_DIM).copy()

    def close(self):
        self.s.close()


# ---------------------------------------------------------------------------
# Stage 3B: Unified Multi-Task Rollout Worker (Supports Eval, Naive Baseline, & Sim Harvesting!)
# ---------------------------------------------------------------------------
def project_points(pts_world: np.ndarray, calib) -> Tuple[np.ndarray, np.ndarray]:
    T_cw = np.linalg.inv(np.asarray(calib.cam_to_base, dtype=np.float64))
    p_c = (T_cw[:3, :3] @ pts_world.T).T + T_cw[:3, 3]
    valid = p_c[:, 2] > 0.05
    u = calib.fx * (p_c[:, 0] / np.maximum(p_c[:, 2], 1e-6)) + calib.cx
    v = calib.fy * (p_c[:, 1] / np.maximum(p_c[:, 2], 1e-6)) + calib.cy
    return np.stack([u, v], axis=-1), valid


def draw_uni_vis(img_bgr, calib_sc, p_ee, p_pick_vlm, p_place_vlm, g_st, pred_3d, pred_g, ee_trail):
    vis = img_bgr.copy()
    w = float(np.clip((g_st - 0.20) / 0.40, 0.0, 1.0))
    p_goal_vlm = (1.0 - w) * p_pick_vlm + w * p_place_vlm
    seg = np.linspace(p_ee, p_goal_vlm, 10)
    uv_s, ok_s = project_points(seg, calib_sc)
    px_s = np.round(uv_s).astype(np.int32)
    for k in range(0, len(px_s) - 1, 2):
        if ok_s[k] and ok_s[k + 1]:
            cv2.line(vis, tuple(px_s[k]), tuple(px_s[k + 1]), (60, 70, 255), 2, cv2.LINE_AA)
    if ok_s[-1]:
        cv2.drawMarker(vis, (int(px_s[-1, 0]), int(px_s[-1, 1])), (60, 70, 255), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
    if len(ee_trail) >= 2:
        t_uv, t_ok = project_points(np.asarray(ee_trail[-75:], dtype=np.float64), calib_sc)
        t_px = np.round(t_uv).astype(np.int32)
        for k in range(len(t_px) - 1):
            if t_ok[k] and t_ok[k + 1]:
                cv2.line(vis, tuple(t_px[k]), tuple(t_px[k + 1]), (90, 255, 130), 2, cv2.LINE_AA)
    if pred_3d is not None and len(pred_3d) >= 2:
        uvs, oks = project_points(pred_3d, calib_sc); pxs = np.round(uvs).astype(np.int32)
        for k in range(len(pxs) - 1):
            if oks[k] and oks[k + 1]:
                g = float(np.clip(pred_g[k], 0.0, 1.0)) if pred_g is not None else 0.0
                col = (int(255 * (1 - g) + 80 * g), int(235 * (1 - g) + 60 * g), int(60 * (1 - g) + 255 * g))
                cv2.line(vis, tuple(pxs[k]), tuple(pxs[k + 1]), col, 2, cv2.LINE_AA)
        if oks[0]:
            cv2.circle(vis, tuple(pxs[0]), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return vis


def _build_task_scene_and_cam(tspec: dict, src_xy, tgt_xy, tz: float, render_frames: bool):
    import mujoco
    if tspec["kind"] == "demo1":
        from scripts.remote_run_demo1_100eps_a100 import build_scene, configure_exact_camera, load_cached_simready, make_calib
        cup = load_cached_simready(DEMO1_HY3D / "simready_cup_mv_v2")
        bowl = load_cached_simready(DEMO1_HY3D / "simready_bowl_mv")
        m = build_scene(cup, bowl, DEMO1_HY3D / "real_wood_table.png", tuple(src_xy), tuple(tgt_xy), offwidth=OUT_WH[0], offheight=OUT_WH[1])
        for aid in range(min(7, m.nu)):
            m.actuator_gainprm[aid, 0] *= 2.2; m.actuator_biasprm[aid, 1] *= 2.2; m.actuator_biasprm[aid, 2] *= 1.4832
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "orange_cup" in nm or "pad" in nm: m.geom_condim[g] = 4
        c1_sc = None
        if render_frames:
            ep0 = json.loads((DEMO1_BUNDLE / "meta_100eps.json").read_text())[0]
            he = json.loads(DEMO1_HE.read_text())
            c1 = make_calib("22008760", ep0["c1_extr"], ep0["c1_intr"]); c1.cam_to_base = np.asarray(he["T_base_cam1_refined"], dtype=np.float64)
            configure_exact_camera(m, c1, OUT_WH[0], OUT_WH[1]); c1_sc = c1.scaled(OUT_WH[0], OUT_WH[1])
        return m, c1_sc, tspec["outcome"]
    else:
        from scripts.remote_universal_stage2_mj4 import build_universal_scene, configure_cam, make_calib
        cfg = json.loads((ROOT / tspec["cfg"]).read_text())
        asset_dir = Path(cfg["asset_dir"]); bundle = Path(cfg["bundle_dir"])
        s1_colors = json.loads((asset_dir / "stage1_colors.json").read_text())
        vis_paths = {}
        obj_rgbs = {}
        for o in cfg["objects"]:
            nm = o["name"]
            mj = asset_dir / f"sim_ready_{nm}" / "asset_meta.json"
            if mj.exists(): vis_paths[nm] = Path(json.loads(mj.read_text())["visual_mesh_path"])
            obj_rgbs[nm] = s1_colors["med_rgbs"].get(nm, o.get("default_rgb", [0.6, 0.6, 0.6]))
        tbl_tex = None
        if render_frames:
            tbl_cands = sorted(asset_dir.glob("tbl_ep*.png"))
            if tbl_cands: tbl_tex = tbl_cands[0]
        m = build_universal_scene(
            cfg, {tspec["src_body"]: tuple(src_xy), tspec["tgt_body"]: tuple(tgt_xy)},
            {tspec["src_body"]: 0.0, tspec["tgt_body"]: 0.0}, obj_rgbs, vis_paths, tz,
            table_tex=tbl_tex, with_robot=True, offwidth=OUT_WH[0], offheight=OUT_WH[1],
        )
        c1_sc = None
        if render_frames:
            meta = json.loads((bundle / "meta.json").read_text())
            fb_c1 = next(e["c1_intr"] for e in meta if e.get("c1_intr") and int(e["c1_intr"].get("width", 0)) > 0)
            ep0 = meta[0]
            c1 = make_calib(ep0["c1_serial"], ep0["c1_extr"], ep0["c1_intr"], fb_c1)
            configure_cam(m, 0, c1, OUT_WH[0], OUT_WH[1]); c1_sc = c1.scaled(OUT_WH[0], OUT_WH[1])
        return m, c1_sc, cfg["outcome"]


def _run_one_uni_ep(args: tuple) -> dict:
    import mujoco
    from r2s_bench.eval.task_success import RolloutMonitor, evaluate_object_in_container, evaluate_object_on_target

    mode, tspec, ep_idx, src_xy, tgt_xy, tz, q_home, vlm_ec, vlm_eb, base_seed, render_frames, record_traj = args
    m, c1_sc, oc = _build_task_scene_and_cam(tspec, src_xy, tgt_xy, tz, render_frames)
    d = mujoco.MjData(m)
    lp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_left_silicone_pad")
    rp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_right_silicone_pad")
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base_mount")
    s_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tspec["src_body"])
    t_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tspec["tgt_body"])
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")

    q_home_arr = np.asarray(q_home, dtype=np.float64)
    q_cmd = q_home_arr.copy()
    d.qpos[:7] = q_cmd; d.ctrl[:7] = q_cmd; d.ctrl[ag] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(35): mujoco.mj_step(m, d)

    p_pick_vlm = np.array([src_xy[0] + vlm_ec[0], src_xy[1] + vlm_ec[1], tz + tspec["pick_z_off"] + vlm_ec[2]], dtype=np.float64)
    p_place_vlm = np.array([tgt_xy[0] + vlm_eb[0], tgt_xy[1] + vlm_eb[1], tz + tspec["place_z_off"] + vlm_eb[2]], dtype=np.float64)
    gv = task_geom_vec(tspec)

    ren = mujoco.Renderer(m, OUT_WH[1], OUT_WH[0]) if render_frames else None
    client = PersistentBridgeClient() if mode == "uni_bridge" else None
    mon = RolloutMonitor(m, target_body_name=tspec["src_body"]); mon.reset(d)

    q_lo = m.jnt_range[:7, 0].copy() + 0.05; q_hi = m.jnt_range[:7, 1].copy() - 0.05
    jacp_l = np.zeros((3, m.nv)); jacp_r = np.zeros((3, m.nv)); jacr_h = np.zeros((3, m.nv)); jacp_tmp = np.zeros((3, m.nv))

    p_ee_prev = 0.5 * (d.xpos[lp] + d.xpos[rp])
    R_init = d.xmat[hb].reshape(3, 3).copy()
    exec_k = 5; chunk = None; chunk_p0 = None; chunk_R0 = None; chunk_step = exec_k; replan_cnt = 0
    naive_ph = 0; naive_tmr = 0; g_cmd = 0.0
    ee_trail = []; frames_out = []
    rec_p_ee, rec_R_ee, rec_g_st, rec_g_cmd, rec_p_src, rec_p_tgt = [], [], [], [], [], []

    try:
        for ti in range(210):
            p_ee = 0.5 * (d.xpos[lp] + d.xpos[rp])
            R_ee = d.xmat[hb].reshape(3, 3)
            g_st = pad_dist_to_g_state(float(np.linalg.norm(d.xpos[lp] - d.xpos[rp])))
            p_src = d.xpos[s_bid].copy(); p_tgt = d.xpos[t_bid].copy()
            ee_vel = (p_ee - p_ee_prev) / DT_CTRL
            ee_trail.append(p_ee.copy())

            if mode == "naive_vlm_line":
                if naive_ph == 0:
                    diff = p_pick_vlm - p_ee; v_lin = np.clip(2.8 * diff, -0.32, 0.32); g_cmd = 0.0
                    if float(np.linalg.norm(diff)) < 0.018 or ti > 70: naive_ph = 1; naive_tmr = 0
                elif naive_ph == 1:
                    v_lin = np.zeros(3); g_cmd = 0.90; naive_tmr += 1
                    if naive_tmr >= 12: naive_ph = 2; naive_tmr = 0
                elif naive_ph == 2:
                    diff = p_place_vlm - p_ee; v_lin = np.clip(2.8 * diff, -0.32, 0.32); g_cmd = 0.90; naive_tmr += 1
                    if float(np.linalg.norm(diff)) < 0.022 or naive_tmr > 85: naive_ph = 3; naive_tmr = 0
                else:
                    v_lin = np.array([0.0, 0.0, 0.22]); g_cmd = 0.0; naive_tmr += 1
                    if naive_tmr >= 16: break
                w_ang = np.clip(2.5 * Rotation.from_matrix(R_init @ R_ee.T).as_rotvec(), -1.5, 1.5)
            else:
                if chunk is None or chunk_step >= exec_k:
                    obs = build_obs_np(p_ee, R_ee, g_st, ee_vel, p_src, p_tgt, p_pick_vlm, p_place_vlm, tz, gv)
                    a_line = make_straight_line_prior_np(p_ee, p_pick_vlm, p_place_vlm, g_st)
                    chunk = client.query(obs, a_line, seed=(base_seed + replan_cnt * 13))
                    chunk_p0 = p_ee.copy(); chunk_R0 = R_ee.copy(); chunk_step = 0; replan_cnt += 1
                a_step = chunk[min(chunk_step, HORIZON - 1)]
                p_des = chunk_p0 + a_step[:3].astype(np.float64)
                R_des = Rotation.from_rotvec(a_step[3:6].astype(np.float64)).as_matrix() @ chunk_R0
                g_cmd = float(np.clip(a_step[6], 0.0, 1.0))
                chunk_step += 1
                v_lin = np.clip((p_des - p_ee) / DT_CTRL, -0.45, 0.45)
                w_ang = np.clip(Rotation.from_matrix(R_des @ R_ee.T).as_rotvec() / DT_CTRL, -1.8, 1.8)

            if record_traj:
                rec_p_ee.append(p_ee.astype(np.float32)); rec_R_ee.append(R_ee.astype(np.float32))
                rec_g_st.append(np.float32(g_st)); rec_g_cmd.append(np.float32(g_cmd))
                rec_p_src.append(p_src.astype(np.float32)); rec_p_tgt.append(p_tgt.astype(np.float32))

            mujoco.mj_jacBody(m, d, jacp_l, jacp_tmp, lp)
            mujoco.mj_jacBody(m, d, jacp_r, jacp_tmp, rp)
            mujoco.mj_jacBody(m, d, jacp_tmp, jacr_h, hb)
            Jp = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7]); Jr = jacr_h[:, :7]
            J = np.vstack([Jp, 0.55 * Jr]); xi = np.concatenate([v_lin, 0.55 * w_ang])
            J_pinv = J.T @ np.linalg.inv(J @ J.T + (0.06 ** 2) * np.eye(6))
            q_cur = d.qpos[:7].copy()
            q_dot = np.clip(J_pinv @ xi + (np.eye(7) - J_pinv @ J) @ (0.7 * (q_home_arr - q_cur)), -1.8, 1.8)
            q_cmd = np.clip(q_cur + DT_CTRL * q_dot, q_lo, q_hi)
            p_ee_prev = p_ee.copy()

            d.ctrl[:7] = q_cmd; d.ctrl[ag] = g_cmd * 255.0
            for s in range(33):
                mujoco.mj_step(m, d)
                if (s & 3) == 0: mon.observe(d)

            if ren is not None and (ti % 2 == 0):
                ren.update_scene(d, camera=0)
                sb = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
                p3d = (chunk_p0[None, :] + chunk[:, :3].astype(np.float64)) if chunk is not None else None
                pg = chunk[:, 6] if chunk is not None else None
                vb = draw_uni_vis(sb, c1_sc, p_ee, p_pick_vlm, p_place_vlm, g_st, p3d, pg, ee_trail)
                frames_out.append((ti, g_cmd, vb, sb))

            if mode == "uni_bridge" and ti > 95 and g_cmd < 0.15 and float(np.linalg.norm(p_src[:2] - p_tgt[:2])) < 0.06 and (p_ee[2] - p_src[2]) > 0.12:
                break
    finally:
        if client is not None: client.close()
        if ren is not None: ren.close()

    for _ in range(50):
        d.ctrl[:7] = q_cmd; d.ctrl[ag] = 0.0; mujoco.mj_step(m, d)
    mon.observe(d)

    if oc["type"] == "in_container":
        res = evaluate_object_in_container(m, d, mon, tspec["src_body"], tspec["tgt_body"], max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [-0.02, 0.12]), table_z=tz)
    else:
        res = evaluate_object_on_target(m, d, mon, tspec["src_body"], tspec["tgt_body"], max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [0.005, 0.14]))

    out = {"tid": tspec["tid"], "ep_idx": ep_idx, "success": bool(res.success), "failure_reason": res.failure_reason, "vlm_err_cm": round(float(np.linalg.norm(vlm_ec[:2])) * 100.0, 2)}
    if record_traj and res.success and len(rec_p_ee) >= 24:
        out["traj"] = {
            "tid": tspec["tid"], "table_z": tz,
            "p_ee": np.stack(rec_p_ee), "R_ee": np.stack(rec_R_ee),
            "g_state": np.asarray(rec_g_st), "g_cmd": np.asarray(rec_g_cmd),
            "p_src": np.stack(rec_p_src), "p_tgt": np.stack(rec_p_tgt),
        }
    if render_frames and frames_out and res.success:
        fd = OUT_DIR / "vid_frames" / f"t{tspec['tid']}_ep{ep_idx:03d}"
        fd.mkdir(parents=True, exist_ok=True)
        for k, (ti, gc, vb, sb) in enumerate(frames_out):
            cv2.imwrite(str(fd / f"br_{k:04d}.jpg"), vb, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            cv2.imwrite(str(fd / f"sim_{k:04d}.jpg"), sb, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        (fd / "meta.json").write_text(json.dumps({"tid": tspec["tid"], "ep_idx": ep_idx, "vlm_err_cm": out["vlm_err_cm"], "n_frames": len(frames_out), "g_cmds": [round(float(x[1]), 2) for x in frames_out]}))
    return out


def _sample_task_ood_specs(task_meta: dict, n_per_task: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    specs = []
    for tspec in TASK_SPECS:
        tid = tspec["tid"]
        tm = task_meta[str(tid)]
        x0, x1, y0, y1 = tm["xy_bounds"]
        d_lo, d_hi = tm["dist_range"]
        tz = float(tm["table_z_med"]); qh = tm["q_home"]
        for i in range(n_per_task):
            for _ in range(500):
                sx = float(rng.uniform(x0, x1)); sy = float(rng.uniform(y0, y1))
                tx = float(rng.uniform(x0, x1)); ty = float(rng.uniform(y0, y1))
                dist = math.hypot(sx - tx, sy - ty)
                if d_lo <= dist <= d_hi: break
            a1 = float(rng.uniform(0, 2 * math.pi)); r1 = float(rng.uniform(0.018, 0.032))
            a2 = float(rng.uniform(0, 2 * math.pi)); r2 = float(rng.uniform(0.018, 0.032))
            ec = (r1 * math.cos(a1), r1 * math.sin(a1), float(rng.uniform(-0.008, 0.008)))
            eb = (r2 * math.cos(a2), r2 * math.sin(a2), float(rng.uniform(-0.008, 0.008)))
            specs.append((tspec, i, (sx, sy), (tx, ty), tz, qh, ec, eb))
    return specs


def _shutdown_server():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", PORT))
        s.sendall(struct.pack("<i", -999999) + b"\x00" * ((OBS_DIM + HORIZON * ACT_DIM) * 4))
        s.close()
    except Exception:
        pass


def stage_eval_or_harvest(round_idx: int, do_harvest: bool = False, run_naive: bool = False, render_vids: bool = False) -> None:
    t0 = time.perf_counter()
    task_meta = json.loads((OUT_DIR / "task_meta.json").read_text())
    ctx = mp.get_context("spawn")
    n_w = int(os.environ.get("R2S_WORKERS", "12"))

    if run_naive:
        test_specs = _sample_task_ood_specs(task_meta, n_per_task=25, seed=2026)
        tasks_n = [("naive_vlm_line", ts, i, sxy, txy, tz, qh, ec, eb, 0, False, False) for ts, i, sxy, txy, tz, qh, ec, eb in test_specs]
        with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
            n_res = [f.result() for f in as_completed([pool.submit(_run_one_uni_ep, t) for t in tasks_n])]
        per_t = {ts["name"]: f"{sum(int(r['success']) for r in n_res if r['tid']==ts['tid'])}/25" for ts in TASK_SPECS}
        tot = sum(int(r["success"]) for r in n_res)
        (OUT_DIR / "naive_125_results.json").write_text(json.dumps({"total": tot, "out_of": len(n_res), "per_task": per_t}, indent=2))
        print(f"[Naive Straight-Line +-2.5cm VLM Click (125 OOD)] {tot}/125 ({100*tot/125:.1f}%) | {per_t}", flush=True)

    if do_harvest:
        # Disjoint seed=9999 -> 50 brand-new OOD placements per task = 250 rollouts!
        harv_specs = _sample_task_ood_specs(task_meta, n_per_task=50, seed=9999)
        tasks_h = [("uni_bridge", ts, i, sxy, txy, tz, qh, ec, eb, 80000 + ts["tid"] * 1000 + i * 17, False, True) for ts, i, sxy, txy, tz, qh, ec, eb in harv_specs]
        with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
            h_res = [f.result() for f in as_completed([pool.submit(_run_one_uni_ep, t) for t in tasks_h])]
        parts = []
        counts = {}
        for ts in TASK_SPECS:
            tid = ts["tid"]
            ok_trajs = [r["traj"] for r in h_res if r["tid"] == tid and r["success"] and "traj" in r]
            counts[ts["name"]] = len(ok_trajs)
            if ok_trajs:
                parts.append(_demos_to_transitions(ok_trajs, ts))
        if parts:
            merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}
            np.savez_compressed(OUT_DIR / "uni_sim_harvest.npz", **merged)
            tot_h = sum(counts.values())
            print(f"[Sim-in-the-Loop HARVEST] Collected {tot_h}/250 ({100*tot_h/250:.1f}%) passing OOD trajectories -> {len(merged['act'])} new chunks in {time.perf_counter()-t0:.1f}s | {counts}", flush=True)
            (OUT_DIR / "harvest_summary.json").write_text(json.dumps({"passing_trajs": tot_h, "total_rollouts": 250, "new_chunks": len(merged["act"]), "per_task": counts}, indent=2))
        _shutdown_server()
        return

    # Evaluate on the fixed held-out 125 unseen full-table OOD placements (25 per task, seed=2026)
    test_specs = _sample_task_ood_specs(task_meta, n_per_task=25, seed=2026)
    tasks_e = [("uni_bridge", ts, i, sxy, txy, tz, qh, ec, eb, 20000 + ts["tid"] * 1000 + i * 31, (render_vids and i < 5), False) for ts, i, sxy, txy, tz, qh, ec, eb in test_specs]
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
        e_res = [f.result() for f in as_completed([pool.submit(_run_one_uni_ep, t) for t in tasks_e])]
    _shutdown_server()

    per_task = {}
    tot_ok = 0
    for ts in TASK_SPECS:
        tid = ts["tid"]
        sub = [r for r in e_res if r["tid"] == tid]
        ok = sum(int(r["success"]) for r in sub)
        tot_ok += ok
        per_task[ts["name"]] = {"pass": ok, "total": len(sub), "rate_pct": round(100 * ok / len(sub), 1)}
    summary = {"round": round_idx, "total_pass": tot_ok, "total_eps": len(e_res), "rate_pct": round(100 * tot_ok / len(e_res), 1), "eval_sec": round(time.perf_counter() - t0, 1), "per_task": per_task}
    (OUT_DIR / f"uni_r{round_idx}_eval_125.json").write_text(json.dumps(summary, indent=2))

    print("=" * 96, flush=True)
    print(f"[UNI-GOAL-BRIDGE DiT ROUND {round_idx} — 125 HELD-OUT UNSEEN OOD POSES (5 TASKS x 4 LABS)] {tot_ok}/125 ({100*tot_ok/125:.1f}%) in {time.perf_counter()-t0:.1f}s", flush=True)
    for ts in TASK_SPECS:
        pt = per_task[ts["name"]]
        print(f"  Task {ts['tid']} {ts['label']:48s}: {pt['pass']:2d}/{pt['total']} ({pt['rate_pct']:5.1f}%)", flush=True)
    print("=" * 96, flush=True)

    # Render 5-Row Synchronized Vertical Stack Showcase Video (Round 2!)
    if render_vids:
        row_clips = []
        row_w, row_h = 960, 270
        for ts in TASK_SPECS:
            tid = ts["tid"]
            cands = sorted((OUT_DIR / "vid_frames").glob(f"t{tid}_ep*"))
            if not cands: continue
            vd = cands[0]; vm = json.loads((vd / "meta.json").read_text())
            frs = []
            for k in range(vm["n_frames"]):
                vb = cv2.imread(str(vd / f"br_{k:04d}.jpg")); sb = cv2.imread(str(vd / f"sim_{k:04d}.jpg"))
                if vb is not None and sb is not None:
                    frs.append((np.hstack([vb, sb]), vm["g_cmds"][k], vm["vlm_err_cm"]))
            if frs:
                pt = per_task[ts["name"]]
                row_clips.append((f"{ts['label']} | Unseen OOD {pt['pass']}/{pt['total']} ({pt['rate_pct']:.0f}%) | VLM err {vm['vlm_err_cm']:.1f}cm", frs))

        if row_clips:
            hdr_h, bar_h = 40, 22
            n_rows = len(row_clips)
            total_h = hdr_h + n_rows * (row_h + bar_h)
            if total_h % 2 != 0: total_h += 1
            n_out = max(len(c[1]) for c in row_clips)
            raw_mp4 = OUT_DIR / "raw_uni5.mp4"
            out_mp4 = OUT_DIR / "uni_goal_bridge_5tasks_scaling_showcase.mp4"
            vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (row_w, total_h))
            snaps = []
            for fi in range(n_out):
                hdr = np.full((hdr_h, row_w, 3), 18, dtype=np.uint8)
                cv2.putText(hdr, f"ONE MODEL x 5 LABS/SKILLS: UNI-GOAL-BRIDGE DiT R{round_idx} (125 UNSEEN OOD: {tot_ok}/125 = {100*tot_ok/125:.1f}%, ZERO WAYPOINTS!)", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (110, 248, 160), 1, cv2.LINE_AA)
                parts_img = [hdr]
                for lbl, frs in row_clips:
                    img2, gc, _ = frs[min(fi, len(frs) - 1)]
                    bar = np.full((bar_h, row_w, 3), 26, dtype=np.uint8)
                    cv2.putText(bar, f"{lbl} | g={gc:.2f}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (105, 245, 165), 1, cv2.LINE_AA)
                    parts_img.extend([bar, img2])
                canvas = np.vstack(parts_img)
                if canvas.shape[0] < total_h:
                    canvas = np.vstack([canvas, np.zeros((total_h - canvas.shape[0], row_w, 3), dtype=np.uint8)])
                vw.write(canvas)
                if fi in (18, n_out // 2, n_out - 5):
                    snaps.append(canvas)
            vw.release()
            subprocess.run(["ffmpeg", "-y", "-i", str(raw_mp4), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", str(out_mp4)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raw_mp4.unlink(missing_ok=True)
            if snaps:
                cv2.imwrite(str(OUT_DIR / "inspect_uni_goal_bridge_5tasks.png"), np.hstack(snaps[:2]))
            print(f"[UNI-5-TASK VIDEO] Saved {out_mp4} ({out_mp4.stat().st_size/1e6:.2f} MB) & inspect_uni_goal_bridge_5tasks.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["collect", "train", "serve", "eval", "harvest"], required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.stage == "collect":
        stage_collect()
    elif args.stage == "train":
        stage_train(args.round)
    elif args.stage == "serve":
        stage_serve(args.round)
    elif args.stage == "harvest":
        stage_eval_or_harvest(args.round, do_harvest=True)
    else:
        stage_eval_or_harvest(args.round, do_harvest=False, run_naive=args.naive, render_vids=args.render)
