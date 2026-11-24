"""Uni-Goal-Bridge v4: Clean-Reconstructed 11-Task x 7-Lab + SE(2) Object-Orientation-Aware Action Render.

Upgrades over v3 (see `docs/REAL2SIM_PIPELINE_ISSUES.md`):
  1. Single Source of Truth: reads `size_xyz`, `grasp_height_m`, `is_hollow` LIVE from `configs/scenes/*.json`
     (eliminates Demo #8 config-policy geometry desync).
  2. SE(2) Nematic Object-Orientation Token (`OBS_DIM: 43 -> 47`):
     - Records live horizontal principal axis `u_src_xy` of every object from MuJoCo `d.xmat[s_bid][:, 1]`
     - Computes anisotropy weight `aniso = clip((max(sx,sy) - min(sx,sy)) / max(sx,sy), 0, 1)`
     - Appends 4 exact rotation-equivariant features to `t_ls`:
         `[aniso*u_x, aniso*u_y, aniso*(u_x^2 - u_y^2), aniso*(2*u_x*u_y)]`
       rotated consistently under 360-deg SO(2) augmentation!
     - Vanishes cleanly to 0 for rotationally symmetric objects (cups/blocks/cans/tape) and gives exact
       180-deg-symmetric barrel heading `(cos 2phi, sin 2phi)` for elongated objects (markers/pens/ducks)!
  3. Re-collects all 11 tasks from the upgraded P1/P2/P3 Real2Sim scenes (0 explosions, capsule markers,
     caliper-exact duck/can/block/cup geometry), trains 19.3M `v4_grand_real`, and evaluates on the exact
     same `orig5` (125 poses, seed=2026) + `new6` (150 poses, seed=2027) OOD benchmarks!
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

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

import scripts.remote_uni_goal_bridge_scaling as uni

OUT_DIR = ROOT / "outputs" / "uni_goal_bridge_v4"
DEMO1_DIT_DATA = uni.DEMO1_DIT_DATA
DEMO1_BUNDLE = uni.DEMO1_BUNDLE
DEMO1_HY3D = uni.DEMO1_HY3D
DEMO1_HE = uni.DEMO1_HE

DT_CTRL = uni.DT_CTRL
OUT_WH = uni.OUT_WH
HORIZON = uni.HORIZON
ACT_DIM = uni.ACT_DIM
OBS_DIM = 47  # 43 + 4 SE(2) nematic object orientation features!
BRIDGE_SIGMA = uni.BRIDGE_SIGMA
ODE_STEPS = uni.ODE_STEPS
PORT_V4 = 18964

UNIVERSAL_SCENES = [
    (1, "demo7_gupta_stack", "2. GuptaLab | Stack Orange Block on Green Block", "configs/scenes/demo7_gupta_stack.json"),
    (2, "demo8_auto_duck", "3. AUTOLab | Orange Rubber Duck -> Silver Pot", "configs/scenes/demo8_auto_duck.json"),
    (3, "demo9_pennpal_tape", "4. PennPAL | Hollow Black Tape -> White Plate", "configs/scenes/demo9_pennpal_tape.json"),
    (4, "demo10_rpl_can", "5. RPL Kitchen | Mini Soda Can -> Bowl", "configs/scenes/demo10_rpl_can.json"),
    (5, "demo11_auto_green_block", "6. AUTOLab | Green Block -> Black Bowl", "configs/scenes/demo11_auto_green_block.json"),
    (6, "demo12_auto_cup_in_cup", "7. AUTOLab | Blue Cup -> Orange Cup", "configs/scenes/demo12_auto_cup_in_cup.json"),
    (7, "demo13_iris_marker_bowl", "8. IRIS | Marker -> Target Bowl", "configs/scenes/demo13_iris_marker_bowl.json"),
    (8, "demo14_clvr_pen_mug", "9. CLVR | Pen -> Target Mug", "configs/scenes/demo14_clvr_pen_mug.json"),
    (9, "demo15_rail_block_plate", "10. RAIL | Yellow Block -> Orange Plate", "configs/scenes/demo15_rail_block_plate.json"),
    (10, "demo16_rail_marker_pot", "11. RAIL | Marker -> Target Pot", "configs/scenes/demo16_rail_marker_pot.json"),
]


def build_all_task_specs_v4() -> List[dict]:
    specs = [copy.deepcopy(uni.TASK_SPECS[0])]
    specs[0]["aniso"] = 0.0
    for tid, name, label, cfg_rel in UNIVERSAL_SCENES:
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


def geom_aniso_from_vec(gv: np.ndarray) -> float:
    sx, sy = float(gv[0]), float(gv[1])
    return float(np.clip((max(sx, sy) - min(sx, sy)) / max(sx, sy, 1e-4), 0.0, 1.0))


def build_obs_v4_np(
    p_ee: np.ndarray, R_ee: np.ndarray, g_st: float, ee_vel: np.ndarray,
    p_src: np.ndarray, p_tgt: np.ndarray, u_src_xy: np.ndarray,
    p_pick_vlm: np.ndarray, p_place_vlm: np.ndarray,
    table_z: float, geom_vec: np.ndarray,
) -> np.ndarray:
    def dir_sat(vec: np.ndarray, scale: float) -> np.ndarray:
        n = float(np.linalg.norm(vec)); u = vec / max(n, 1e-5)
        n_xy = float(np.linalg.norm(vec[:2]))
        return np.array([u[0], u[1], u[2], math.tanh(n / scale), math.tanh(n_xy / scale), float(np.clip(vec[2], -0.25, 0.25))], dtype=np.float32)

    d_src = p_src - p_ee; d_tgt = p_tgt - p_ee; d_st = p_src - p_tgt
    t_vp = dir_sat(p_pick_vlm - p_ee, 0.15)                                                           # 6
    t_vb = dir_sat(p_place_vlm - p_ee, 0.15)                                                          # 6

    an = geom_aniso_from_vec(geom_vec)
    un = float(np.linalg.norm(u_src_xy))
    ux, uy = (float(u_src_xy[0] / un), float(u_src_xy[1] / un)) if un > 1e-5 else (0.0, 1.0)
    ori_feats = np.array([an * ux, an * uy, an * (ux * ux - uy * uy), an * (2.0 * ux * uy)], dtype=np.float32)  # 4
    t_ls = np.concatenate([dir_sat(d_src, 0.12), [float(p_src[2] - table_z)], ori_feats])             # 11

    t_lt = np.concatenate([dir_sat(d_tgt, 0.12), [math.tanh(float(np.linalg.norm(d_st[:2])) / 0.12), float(np.clip(d_st[2], -0.2, 0.2))]])  # 8
    z_ax = R_ee[:, 2]
    is_hol = float(geom_vec[6])
    d_s_xy = float(np.linalg.norm(d_src[:2]))
    held_cue = float(g_st * math.exp(-((d_s_xy / (0.055 + 0.035 * is_hol)) ** 2)))
    t_ee = np.array([
        float(p_ee[2] - table_z), z_ax[0], z_ax[1], z_ax[2],
        float(g_st), ee_vel[0], ee_vel[1], ee_vel[2], held_cue,
    ], dtype=np.float32)                                                                              # 9
    return np.concatenate([t_vp, t_vb, t_ls, t_lt, t_ee, geom_vec], axis=0)                           # 47


# ---------------------------------------------------------------------------
# Stage 1: Re-collect Clean 11-Task x 7-Lab Real Dataset with Live Object Axis `u_src`
# ---------------------------------------------------------------------------
def _replay_universal_ep_v4(args: tuple) -> dict | None:
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
            if float(np.linalg.norm(cand - anc)) <= 0.16:
                xy0 = cand
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

    # Exact P1 Non-Overlap Radius Guard (matching Stage 2 v3!)
    r_src = 0.5 * max(float(src_cfg["size_xyz"][0]), float(src_cfg["size_xyz"][1]))
    r_tgt = 0.5 * max(float(tgt_cfg["size_xyz"][0]), float(tgt_cfg["size_xyz"][1]))
    min_sep = r_src + r_tgt + 0.012
    if float(np.linalg.norm(np.array(obj_xys[src_name]) - np.array(obj_xys[tgt_name]))) < min_sep:
        if float(np.linalg.norm(po[:2] - np.array(obj_xys[src_name]))) >= min_sep:
            obj_xys[tgt_name] = (float(po[0]), float(po[1]))
        else:
            u = po[:2] - pc[:2]
            un = float(np.linalg.norm(u))
            u = (u / un) if un > 1e-4 else np.array([0.0, 1.0])
            obj_xys[tgt_name] = (float(obj_xys[src_name][0] + min_sep * u[0]), float(obj_xys[src_name][1] + min_sep * u[1]))

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
    for _ in range(30):
        mujoco.mj_step(m, d)

    T = len(qs); hold_ctrl = float(cfg.get("hold_ctrl", 0.82))
    p_ees = np.zeros((T, 3), dtype=np.float32); R_ees = np.zeros((T, 3, 3), dtype=np.float32)
    g_states = np.zeros((T,), dtype=np.float32); g_cmds = np.zeros((T,), dtype=np.float32)
    p_srcs = np.zeros((T, 3), dtype=np.float32); p_tgts = np.zeros((T, 3), dtype=np.float32)
    u_srcs = np.zeros((T, 2), dtype=np.float32)

    mon = RolloutMonitor(m, target_body_name=src_name); mon.reset(d)
    for ti in range(T):
        gc = hold_ctrl if (max(0, ev["t_close"] - 1) <= ti < ev["t_open"]) else float(np.clip(gs[ti], 0, 1))
        p_ees[ti] = 0.5 * (d.xpos[lp] + d.xpos[rp]); R_ees[ti] = d.xmat[hb].reshape(3, 3)
        g_states[ti] = uni.pad_dist_to_g_state(float(np.linalg.norm(d.xpos[lp] - d.xpos[rp])))
        g_cmds[ti] = gc; p_srcs[ti] = d.xpos[s_bid]; p_tgts[ti] = d.xpos[t_bid]
        # Principal horizontal axis of source body (local Y-axis of capsule/box!)
        y_ax = d.xmat[s_bid].reshape(3, 3)[:2, 1]
        yn = float(np.linalg.norm(y_ax))
        u_srcs[ti] = (y_ax / yn).astype(np.float32) if yn > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)

        d.ctrl[:7] = qs[min(ti + 1, T - 1)]; d.ctrl[ag] = gc * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)
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
        "p_ee": p_ees, "R_ee": R_ees, "g_state": g_states, "g_cmd": g_cmds,
        "p_src": p_srcs, "p_tgt": p_tgts, "u_src": u_srcs,
    }


def _demos_to_transitions_v4(demos: List[dict], tspec: dict) -> dict:
    gv = uni.task_geom_vec(tspec)
    p_ee_l, R_ee_l, g_st_l, vel_l, p_src_l, p_tgt_l, u_src_l, tz_l, tid_l, act_l, gv_l = [], [], [], [], [], [], [], [], [], [], []
    for dm in demos:
        p_ee, R_ee, g_st, g_cmd = dm["p_ee"], dm["R_ee"], dm["g_state"], dm["g_cmd"]
        p_src, p_tgt, u_src = dm["p_src"], dm["p_tgt"], dm["u_src"]
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
            p_src_l.append(p_src[t]); p_tgt_l.append(p_tgt[t]); u_src_l.append(u_src[t])
            tz_l.append(tz); tid_l.append(tspec["tid"]); act_l.append(act); gv_l.append(gv)
    return {
        "p_ee": np.asarray(p_ee_l, dtype=np.float32), "R_ee": np.asarray(R_ee_l, dtype=np.float32),
        "g_st": np.asarray(g_st_l, dtype=np.float32), "vel": np.asarray(vel_l, dtype=np.float32),
        "p_src": np.asarray(p_src_l, dtype=np.float32), "p_tgt": np.asarray(p_tgt_l, dtype=np.float32),
        "u_src": np.asarray(u_src_l, dtype=np.float32), "table_z": np.asarray(tz_l, dtype=np.float32),
        "tid": np.asarray(tid_l, dtype=np.int64), "act": np.asarray(act_l, dtype=np.float32),
        "geom": np.asarray(gv_l, dtype=np.float32),
    }


def stage_collect_v4() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    all_specs = build_all_task_specs_v4()
    parts = []
    task_meta = {}
    active_tids = [0]

    # Task 0: Reuse verified Demo #1 (round cup, aniso=0.0)
    d1 = np.load(DEMO1_DIT_DATA)
    n1 = len(d1["act"])
    gv0 = uni.task_geom_vec(all_specs[0])
    parts.append({
        "p_ee": d1["p_ee"], "R_ee": d1["R_ee"], "g_st": d1["g_st"], "vel": d1["vel"],
        "p_src": d1["p_cup"], "p_tgt": d1["p_bowl"],
        "u_src": np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (n1, 1)),
        "table_z": np.full((n1,), -0.015, dtype=np.float32),
        "tid": np.zeros((n1,), dtype=np.int64),
        "act": d1["act"], "geom": np.repeat(gv0[None, :], n1, axis=0),
    })
    task_meta["0"] = {
        "name": "demo1_cup_bowl", "n_real_demos": int(d1["n_demos"]), "n_samples": n1,
        "q_home": d1["q_home"].tolist(), "table_z_med": -0.015,
        "xy_bounds": [0.41, 0.69, -0.25, 0.25], "dist_range": [0.17, 0.40],
    }
    print(f"[Collect v4 Task  0 demo1_cup_bowl] Reused {int(d1['n_demos'])} real demos -> {n1} chunks", flush=True)

    ctx = mp.get_context("spawn")
    n_w = int(os.environ.get("R2S_WORKERS", "24"))
    for tspec in all_specs[1:]:
        tid = tspec["tid"]
        cfg = json.loads((ROOT / tspec["cfg"]).read_text())
        sid = cfg["scene_id"]; bundle = Path(cfg["bundle_dir"]); asset_dir = Path(cfg["asset_dir"])
        res_prev = json.loads((asset_dir / f"{sid}_results.json").read_text())
        pass_ids = [e["ep_idx"] for e in res_prev["episodes"] if e["success"]]

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
            for f in as_completed([pool.submit(_replay_universal_ep_v4, t) for t in tasks]):
                r = f.result()
                if r is not None:
                    demos.append(r)

        demos.sort(key=lambda x: x["ep_idx"])
        part = _demos_to_transitions_v4(demos, tspec)
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
        print(f"[Collect v4 Task {tid:2d} {sid}] {len(demos)}/{len(pass_ids)} demos -> {len(part['act'])} chunks | aniso={tspec['aniso']:.2f} | tz={tz_med*100:.1f}cm", flush=True)

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}
    np.savez_compressed(OUT_DIR / "uni_real_v4_11tasks.npz", **merged)
    (OUT_DIR / "task_meta_v4.json").write_text(json.dumps({"active_tids": active_tids, "tasks": task_meta}, indent=2))
    tot = sum(v["n_real_demos"] for v in task_meta.values())
    print(f"[Collect v4 COMPLETE] {len(active_tids)} tasks | {tot} clean real demos -> {len(merged['act'])} chunks in {time.perf_counter()-t0:.1f}s!", flush=True)


# ---------------------------------------------------------------------------
# Stage 2: 19.3M SE(2)-Orientation-Aware Uni-Goal-Bridge v4 DiT
# ---------------------------------------------------------------------------
def build_v4_dit(d_model: int = 320, n_layers: int = 8, n_heads: int = 8):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SinEmb(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
            ang = t[:, None] * freqs[None, :] * 1000.0
            return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    class QKNormBridgeBlock(nn.Module):
        def __init__(self, d: int, nh: int):
            super().__init__()
            self.nh = nh; self.hd = d // nh
            self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
            self.qkv = nn.Linear(d, 3 * d)
            self.q_norm = nn.LayerNorm(self.hd, eps=1e-6); self.k_norm = nn.LayerNorm(self.hd, eps=1e-6)
            self.proj1 = nn.Linear(d, d)
            self.norm_ca = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
            self.norm_ctx = nn.LayerNorm(d, eps=1e-6)
            self.ca_q = nn.Linear(d, d); self.ca_kv = nn.Linear(d, 2 * d)
            self.ca_qn = nn.LayerNorm(self.hd, eps=1e-6); self.ca_kn = nn.LayerNorm(self.hd, eps=1e-6)
            self.proj_ca = nn.Linear(d, d)
            self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
            nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

        def forward(self, x: torch.Tensor, ctx: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            B, H, D = x.shape
            sh1, sc1, g1, sh2, sc2, g2 = self.ada(cond).chunk(6, dim=-1)
            h = self.norm1(x) * (1.0 + sc1[:, None, :]) + sh1[:, None, :]
            q, k, v = self.qkv(h).view(B, H, 3, self.nh, self.hd).unbind(dim=2)
            q = self.q_norm(q).transpose(1, 2); k = self.k_norm(k).transpose(1, 2); v = v.transpose(1, 2)
            sa = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, H, D)
            x = x + g1[:, None, :] * self.proj1(sa)
            hc = self.norm_ca(x); c = self.norm_ctx(ctx)
            cq = self.ca_qn(self.ca_q(hc).view(B, H, self.nh, self.hd)).transpose(1, 2)
            ck, cv = self.ca_kv(c).view(B, c.shape[1], 2, self.nh, self.hd).unbind(dim=2)
            ck = self.ca_kn(ck).transpose(1, 2); cv = cv.transpose(1, 2)
            ca = F.scaled_dot_product_attention(cq, ck, cv).transpose(1, 2).reshape(B, H, D)
            x = x + self.proj_ca(ca)
            hm = self.norm2(x) * (1.0 + sc2[:, None, :]) + sh2[:, None, :]
            return x + g2[:, None, :] * self.mlp(hm)

    class UniGoalBridgeDiTV4(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_vp = nn.Sequential(nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_vb = nn.Sequential(nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_ls = nn.Sequential(nn.Linear(11, d_model), nn.SiLU(), nn.Linear(d_model, d_model))  # 7 + 4 SE(2) ori!
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
                self.enc_vp(obs[:, 0:6]), self.enc_vb(obs[:, 6:12]),
                self.enc_ls(obs[:, 12:23]), self.enc_lt(obs[:, 23:31]),
                self.enc_ee(obs[:, 31:40]), self.enc_gm(obs[:, 40:47]),
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

    return UniGoalBridgeDiTV4()


def stage_train_v4(n_steps: int = 8500) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_specs = build_all_task_specs_v4()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((OUT_DIR / "task_meta_v4.json").read_text())
    active_tids = meta_obj["active_tids"]

    ds = dict(np.load(OUT_DIR / "uni_real_v4_11tasks.npz"))
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

    pools = {t: torch.from_numpy(np.where(tid_np == t)[0]).to(device) for t in active_tids}
    step_ladder = (torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32) * 0.018)[None, :, None]

    def sample_balanced_idx(batch_size: int) -> torch.Tensor:
        per_t = max(8, batch_size // len(active_tids))
        return torch.cat([pools[t][torch.randint(0, len(pools[t]), (per_t,), device=device)] for t in active_tids], dim=0)

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

        # Exact 360-deg SO(2) planar yaw augmentation (including object 2D principal axis u_src!)
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
        ux = ct * u0[:, 0] - st * u0[:, 1]
        uy = st * u0[:, 0] + ct * u0[:, 1]
        un = torch.sqrt(ux * ux + uy * uy).clamp(min=1e-5)
        ux = ux / un; uy = uy / un
        # Random 180-deg sign flip on symmetric bar vector to reinforce nematic invariance!
        sgn = torch.where(torch.rand((B,), device=device) > 0.5, 1.0, -1.0)
        ux_s = ux * sgn; uy_s = uy * sgn
        ori_feats = torch.stack([an * ux_s, an * uy_s, an * (ux * ux - uy * uy), an * (2.0 * ux * uy)], dim=-1)

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

    model = build_v4_dit(d_model=320, n_layers=8, n_heads=8).to(device)
    with torch.no_grad():
        _, _, a_stat = make_batch(sample_balanced_idx(6000))
        act_mean = a_stat.mean(dim=(0, 1))
        act_std = a_stat.std(dim=(0, 1)).clamp(min=0.03)

    ema_model = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v4_grand_real] {n_params/1e6:.2f}M params (d=320, L=8, OBS=47 SE2-Ori) | {n_steps} steps | Real={len(ds['act'])} chunks", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1e-4, fused=True)
    warmup = 250; bs = 528
    def lr_fn(s):
        if s < warmup: return float(s + 1) / float(warmup)
        p = float(s - warmup) / float(max(1, n_steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    beta_dist = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
    ema_params = list(ema_model.parameters()); cur_params = list(model.parameters())
    w_dim = torch.tensor([1.3, 1.3, 1.5, 0.85, 0.85, 1.0, 2.2], device=device)[None, None, :]  # boosted yaw weight 1.0!

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
        if step % 1000 == 0 or step == 1 or step == n_steps:
            print(f"  [v4 step {step:4d}/{n_steps}] loss={loss.item():.4f} | {(time.perf_counter()-t0)/step*1000:.1f} ms/step ({time.perf_counter()-t0:.1f}s)", flush=True)

    train_dt = time.perf_counter() - t0
    with torch.no_grad():
        obs_v, al_v, ae_v = make_batch(sample_balanced_idx(550))
        pred = ema_model.render_chunk(((al_v - act_mean) / act_std).clamp(-6, 6), obs_v, n_steps=ODE_STEPS, seed=42) * act_std + act_mean
        mae_cm = torch.mean(torch.abs(pred[:, :6, :3] - ae_v[:, :6, :3])).item() * 100.0
        g_mae = torch.mean(torch.abs(pred[:, :6, 6] - ae_v[:, :6, 6])).item()
        print(f"[v4 COMPLETE] {train_dt:.1f}s | 4-step ODE PosMAE={mae_cm:.2f}cm | GripMAE={g_mae:.3f}", flush=True)

    torch.save({
        "ema_state_dict": ema_model.state_dict(),
        "d_model": 320, "n_layers": 8, "n_heads": 8, "obs_dim": OBS_DIM,
        "act_mean": act_mean.cpu().numpy(), "act_std": act_std.cpu().numpy(),
        "active_tids": active_tids, "n_real_chunks": len(ds["act"]),
        "train_sec": round(train_dt, 1), "val_pos_mae_cm": round(mae_cm, 2), "val_grip_mae": round(g_mae, 3),
    }, OUT_DIR / "uni_bridge_v4_grand.pt")


# ---------------------------------------------------------------------------
# Stage 3: Serve & Evaluate v4 (with live SE(2) object axis + optional random OOD yaw!)
# ---------------------------------------------------------------------------
def stage_serve_v4(port: int = PORT_V4) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / "uni_bridge_v4_grand.pt", map_location=device, weights_only=False)
    model = build_v4_dit(int(ck["d_model"]), int(ck["n_layers"]), int(ck["n_heads"])).to(device).eval()
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
    print(f"[Serve v4] READY on 127.0.0.1:{port} (OBS=47 SE2-Ori, d=320, L=8)", flush=True)
    try:
        while not stop_flag.is_set():
            try:
                conn, _ = srv.accept()
                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
    finally:
        srv.close()


class V4BridgeClient:
    def __init__(self, port: int = PORT_V4):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("127.0.0.1", port))

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


def _shutdown_v4(port: int = PORT_V4) -> None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s.sendall(struct.pack("<i", -999999) + b"\x00" * ((OBS_DIM + HORIZON * ACT_DIM) * 4))
        s.close()
    except Exception:
        pass


def _run_one_v4_ep(args: tuple) -> dict:
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
    for _ in range(35):
        mujoco.mj_step(m, d)

    p_src0 = d.xpos[s_bid].copy(); p_tgt0 = d.xpos[t_bid].copy()
    p_pick_vlm = np.array([p_src0[0], p_src0[1], tz + tspec["pick_z_off"]], dtype=np.float32) + vlm_ec
    p_place_vlm = np.array([p_tgt0[0], p_tgt0[1], tz + tspec["place_z_off"]], dtype=np.float32) + vlm_eb

    mon = RolloutMonitor(m, target_body_name=tspec["src_body"]); mon.reset(d)
    client = V4BridgeClient(port=port)

    max_ticks = 210; exec_k = 6
    q_cmd = q_ref.copy(); p_ee_prev = 0.5 * (d.xpos[lp] + d.xpos[rp]).copy()
    chunk_p_targets = None; chunk_R_targets = None; chunk_g_targets = None; chunk_idx = exec_k
    success_hold = 0

    for tick in range(max_ticks):
        p_ee = 0.5 * (d.xpos[lp] + d.xpos[rp]).copy()
        R_ee = d.xmat[hb].reshape(3, 3).copy()
        g_st = uni.pad_dist_to_g_state(float(np.linalg.norm(d.xpos[lp] - d.xpos[rp])))
        p_src = d.xpos[s_bid].copy(); p_tgt = d.xpos[t_bid].copy()
        ee_vel = (p_ee - p_ee_prev) / DT_CTRL; p_ee_prev = p_ee.copy()
        y_ax = d.xmat[s_bid].reshape(3, 3)[:2, 1]
        yn = float(np.linalg.norm(y_ax))
        u_src = (y_ax / yn).astype(np.float32) if yn > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)

        if chunk_idx >= exec_k or chunk_p_targets is None:
            obs = build_obs_v4_np(p_ee, R_ee, g_st, ee_vel, p_src, p_tgt, u_src, p_pick_vlm, p_place_vlm, tz, gv)
            a_line = uni.make_straight_line_prior_np(p_ee, p_pick_vlm, p_place_vlm, g_st)
            act_chunk = client.query(obs, a_line, seed=seed + tick * 7)
            chunk_p_targets = p_ee[None, :] + act_chunk[:, :3]
            chunk_R_targets = [Rotation.from_rotvec(act_chunk[k, 3:6]).as_matrix() @ R_ee for k in range(HORIZON)]
            chunk_g_targets = np.clip(act_chunk[:, 6], 0.0, 1.0)
            chunk_idx = 0

        p_des = chunk_p_targets[chunk_idx]; R_des = chunk_R_targets[chunk_idx]; g_des = float(chunk_g_targets[chunk_idx])
        chunk_idx += 1

        # 2-iteration damped-LS IK
        for _ in range(2):
            p_cur = 0.5 * (d.xpos[lp] + d.xpos[rp])
            R_cur = d.xmat[hb].reshape(3, 3)
            ep_err = p_des - p_cur
            en = float(np.linalg.norm(ep_err))
            if en > 0.045: ep_err = ep_err * (0.045 / en)
            er = Rotation.from_matrix(R_des @ R_cur.T).as_rotvec()
            rn = float(np.linalg.norm(er))
            if rn > 0.35: er = er * (0.35 / rn)
            jp_l = np.zeros((3, m.nv)); jp_r = np.zeros((3, m.nv)); jr_h = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp_l, None, lp)
            mujoco.mj_jacBody(m, d, jp_r, None, rp)
            mujoco.mj_jacBody(m, d, None, jr_h, hb)
            J = np.vstack([0.5 * (jp_l[:, :7] + jp_r[:, :7]), jr_h[:, :7]])
            err = np.concatenate([ep_err, er])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.0025 * np.eye(6), err)
            dq_null = 0.02 * (q_ref - d.qpos[:7])
            dq += (np.eye(7) - J.T @ np.linalg.solve(J @ J.T + 0.0025 * np.eye(6), J)) @ dq_null
            q_cmd = np.clip(d.qpos[:7] + np.clip(dq, -0.18, 0.18), q_lo + 0.02, q_hi - 0.02)

        d.ctrl[:7] = q_cmd; d.ctrl[ag] = g_des * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)

        if g_st < 0.25 and tick > 45:
            dxy = float(np.linalg.norm(d.xpos[s_bid][:2] - d.xpos[t_bid][:2]))
            if dxy <= float(oc.get("max_xy_m", 0.095)) and float(p_ee[2] - tz) > 0.14:
                success_hold += 1
                if success_hold >= 12:
                    break
            else:
                success_hold = 0

    for _ in range(45):
        d.ctrl[:7] = q_cmd; d.ctrl[ag] = 0.0; mujoco.mj_step(m, d)
    mon.observe(d)
    client.close()

    if oc["type"] == "in_container":
        res = evaluate_object_in_container(m, d, mon, tspec["src_body"], tspec["tgt_body"], max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [-0.02, 0.12]), table_z=tz)
    else:
        res = evaluate_object_on_target(m, d, mon, tspec["src_body"], tspec["tgt_body"], max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [0.005, 0.14]))
    return {"tid": tspec["tid"], "ep_id": ep_id, "success": bool(res.success)}


def stage_eval_v4(scope: str = "orig5", port: int = PORT_V4, random_yaw: bool = False) -> None:
    t0 = time.perf_counter()
    n_w = int(os.environ.get("R2S_WORKERS", "24"))
    all_specs = build_all_task_specs_v4()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((OUT_DIR / "task_meta_v4.json").read_text())

    if scope == "orig5":
        # EXACT SAME 125 unseen OOD poses (seed=2026) using original task_meta bounds for 100% apples-to-apples comparison!
        orig_meta = json.loads((uni.OUT_DIR / "task_meta.json").read_text())
        eval_tids = [0, 1, 2, 3, 4]
        rng = np.random.default_rng(2026)
        tasks = []
        for t in eval_tids:
            ts = spec_by_tid[t]
            tm = orig_meta[str(t)]
            x0, x1, y0, y1 = tm["xy_bounds"]; dmin, dmax = tm["dist_range"]
            tz = float(tm["table_z_med"]); qh = np.array(tm["q_home"], dtype=np.float32)
            for i in range(25):
                for _ in range(200):
                    sxy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                    txy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                    if dmin <= math.hypot(sxy[0] - txy[0], sxy[1] - txy[1]) <= dmax:
                        break
                ec = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
                eb = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
                syaw = float(rng.uniform(-math.pi, math.pi)) if (random_yaw and ts["aniso"] > 0.2) else 0.0
                tasks.append((ts, i, sxy, txy, syaw, tz, qh, ec, eb, 90000 + t * 1000 + i * 17, port))
    else:
        eval_tids = [t for t in meta_obj["active_tids"] if t >= 5]
        rng = np.random.default_rng(2027)
        tasks = []
        for t in eval_tids:
            ts = spec_by_tid[t]
            tm = meta_obj["tasks"][str(t)]
            x0, x1, y0, y1 = tm["xy_bounds"]; dmin, dmax = tm["dist_range"]
            tz = float(tm["table_z_med"]); qh = np.array(tm["q_home"], dtype=np.float32)
            for i in range(25):
                for _ in range(200):
                    sxy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                    txy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                    if dmin <= math.hypot(sxy[0] - txy[0], sxy[1] - txy[1]) <= dmax:
                        break
                ec = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
                eb = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
                syaw = float(rng.uniform(-math.pi, math.pi)) if (random_yaw and ts["aniso"] > 0.2) else 0.0
                tasks.append((ts, i, sxy, txy, syaw, tz, qh, ec, eb, 90000 + t * 1000 + i * 17, port))

    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
        for f in as_completed([pool.submit(_run_one_v4_ep, t) for t in tasks]):
            results.append(f.result())

    _shutdown_v4(port)

    by_task = {}
    for t in eval_tids:
        ts = spec_by_tid[t]
        rs = [r for r in results if r["tid"] == t]
        ok = sum(int(r["success"]) for r in rs)
        by_task[ts["name"]] = {"pass": ok, "n": len(rs), "rate": round(100.0 * ok / max(1, len(rs)), 1)}

    tot_ok = sum(int(r["success"]) for r in results)
    tot_n = len(results)
    tag = f"v4_grand_{scope}" + ("_randyaw" if random_yaw else "")
    out = {"tag": tag, "pass": tot_ok, "n": tot_n, "rate": round(100.0 * tot_ok / max(1, tot_n), 1), "per_task": by_task, "elapsed_sec": round(time.perf_counter() - t0, 1)}
    (OUT_DIR / f"eval_{tag}.json").write_text(json.dumps(out, indent=2))
    print("=" * 92, flush=True)
    print(f"[EVAL {tag}] TOTAL = {tot_ok}/{tot_n} ({out['rate']:.1f}%) in {out['elapsed_sec']}s", flush=True)
    for k, v in by_task.items():
        print(f"  - {k:28s}: {v['pass']:2d}/{v['n']:2d} ({v['rate']:5.1f}%)", flush=True)
    print("=" * 92, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "train", "serve", "eval"])
    ap.add_argument("--scope", choices=["orig5", "new6"], default="orig5")
    ap.add_argument("--port", type=int, default=PORT_V4)
    ap.add_argument("--steps", type=int, default=8500)
    ap.add_argument("--random-yaw", action="store_true")
    args = ap.parse_args()
    if args.cmd == "collect":
        stage_collect_v4()
    elif args.cmd == "train":
        stage_train_v4(n_steps=args.steps)
    elif args.cmd == "serve":
        stage_serve_v4(port=args.port)
    elif args.cmd == "eval":
        stage_eval_v4(scope=args.scope, port=args.port, random_yaw=args.random_yaw)


if __name__ == "__main__":
    main()
