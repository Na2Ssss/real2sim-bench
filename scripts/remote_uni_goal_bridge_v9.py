"""Uni-Goal-Bridge v9: 42.3M Full-Scale 23-Task x 11-Lab Action Render (~739 Real Demos + v8 All-17 Sim Flywheel!).

Scales BOTH Data & Model cleanly on top of our record-breaking `v8` (`OBS=48`, 8D physical geometry + SE(2) nematic):
  - Real Dataset: 23 active tasks across 11 calibrated DROID labs (~739 verified passing real demos, ~100,000+ real chunks!)
      - Tasks 0..16  : Cached clean 17-task dataset (`561` real demos, `75,907` chunks)
      - Task 17 (IRIS):     Marker -> Red Mug           (`demo23`, +34 real demos)
      - Task 18 (TRI):      Spoon -> Cup (NEW OBJECT!)  (`demo24`, +26 real demos)
      - Task 19 (TRI):      Red Block -> Box (NEW!)     (`demo25`, +21 real demos)
      - Task 20 (IRIS):     Pen -> Glass Cup            (`demo26`, +37 real demos)
      - Task 21 (AUTOLab):  Block -> Black Bowl B2      (`demo27`, +37 real demos)
      - Task 22 (GuptaLab): 4.6cm Precision Stack B2    (`demo28`, +23 real demos -> +55% more stacking data!)
  - Sim Flywheel: `v8_sim_harvest_all17.npz` (harvested by 75.2%/57.2% `v8` across ALL 17 base tasks!),
    filtered per-task to fastest Top-55% crisp trajectories (`ep_len >= 35`).
  - Architecture: `d_model=448, n_layers=9, n_heads=8` (~42.3M params, TF32 + QK-Norm + SDPA + FusedAdamW).
  - Schedule: Temperature Sampling (`alpha=0.5`, `bs=640`) + Stage A (14,000 steps, 80% Real : 20% Elite v8-Sim)
    -> Stage B (5,000 steps, 100% Pure Real Precision Annealing)!
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
from typing import List

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import numpy as np

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

import scripts.remote_uni_goal_bridge_scaling as uni
import scripts.remote_uni_goal_bridge_v4 as v4
import scripts.remote_uni_goal_bridge_v6 as v6
import scripts.remote_uni_goal_bridge_v8 as v8

V6_DIR = v6.OUT_DIR
V8_DIR = v8.OUT_DIR
OUT_DIR = ROOT / "outputs" / "uni_goal_bridge_v9"
HORIZON = uni.HORIZON
ACT_DIM = uni.ACT_DIM
OBS_DIM = 48
BRIDGE_SIGMA = uni.BRIDGE_SIGMA
ODE_STEPS = uni.ODE_STEPS
PORT_V9 = 18986

WAVE4_SCENES = [
    (17, "demo23_iris_marker_mug",  "18. IRIS | Marker -> Red Mug",       "configs/scenes/demo23_iris_marker_mug.json"),
    (18, "demo24_tri_spoon_cup",    "19. TRI | Spoon -> Cup",             "configs/scenes/demo24_tri_spoon_cup.json"),
    (19, "demo25_tri_block_box",    "20. TRI | Red Block -> Box",         "configs/scenes/demo25_tri_block_box.json"),
    (20, "demo26_iris_pen_cup",     "21. IRIS | Pen -> Glass Cup",        "configs/scenes/demo26_iris_pen_cup.json"),
    (21, "demo27_auto_block_bowl2", "22. AUTOLab | Block -> Bowl B2",     "configs/scenes/demo27_auto_block_bowl2.json"),
    (22, "demo28_gupta_stack_more", "23. GuptaLab | 4.6cm Stack B2",      "configs/scenes/demo28_gupta_stack_more.json"),
]


def build_all_task_specs_v9() -> List[dict]:
    specs = v8.build_all_task_specs_v8()
    for tid, name, label, cfg_rel in WAVE4_SCENES:
        cfg_p = ROOT / cfg_rel
        if not cfg_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text())
        src_o, tgt_o = cfg["objects"][0], cfg["objects"][1]
        s_xyz = [float(v) for v in src_o["size_xyz"]]
        t_xyz = [float(v) for v in tgt_o["size_xyz"]]
        is_hol = 1.0 if src_o.get("grasp_mode") == "rim_pinch" else 0.0
        is_tc = 1.0 if cfg["outcome"]["type"] == "in_container" else 0.0
        pick_z = float(src_o.get("grasp_height_m", 0.5 * s_xyz[2]))
        place_z = float(t_xyz[2] + 0.032) if is_tc > 0.5 else float(t_xyz[2] + 0.025)
        aniso = float(np.clip((max(s_xyz[0], s_xyz[1]) - min(s_xyz[0], s_xyz[1])) / max(s_xyz[0], s_xyz[1], 1e-4), 0.0, 1.0))
        specs.append({
            "tid": tid, "name": name, "label": label, "kind": "universal", "cfg": cfg_rel,
            "src_body": src_o["name"], "tgt_body": tgt_o["name"],
            "src_size": s_xyz, "tgt_size": t_xyz, "is_hollow": is_hol, "is_tgt_container": is_tc,
            "aniso": round(aniso, 4), "pick_z_off": round(pick_z, 4), "place_z_off": round(place_z, 4),
        })
    return specs


def stage_collect_v9(min_demos: int = 5) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    base_ds = dict(np.load(V6_DIR / "uni_real_v6_17tasks.npz"))
    base_meta = json.loads((V6_DIR / "task_meta_v6.json").read_text())
    parts = [base_ds]
    task_meta = dict(base_meta["tasks"])
    active_tids = list(base_meta["active_tids"])

    all_specs = build_all_task_specs_v9()
    ctx = mp.get_context("spawn")
    n_w = int(os.environ.get("R2S_WORKERS", "24"))

    for tspec in all_specs[17:]:
        tid = tspec["tid"]
        cfg = json.loads((ROOT / tspec["cfg"]).read_text())
        sid = cfg["scene_id"]; bundle = Path(cfg["bundle_dir"]); asset_dir = Path(cfg["asset_dir"])
        res_p = asset_dir / f"{sid}_results.json"
        if not res_p.exists(): continue
        res_prev = json.loads(res_p.read_text())
        pass_ids = [e["ep_idx"] for e in res_prev["episodes"] if e["success"]]
        if len(pass_ids) < min_demos: continue

        meta = json.loads((bundle / "meta.json").read_text())
        masks = dict(np.load(bundle / "masks.npz"))
        s1_colors = json.loads((asset_dir / "stage1_colors.json").read_text())
        fb_c1 = next(m["c1_intr"] for m in meta if m.get("c1_intr") and int(m["c1_intr"].get("width", 0)) > 0)
        fb_c2 = next(m["c2_intr"] for m in meta if m.get("c2_intr") and int(m["c2_intr"].get("width", 0)) > 0)
        fallback_intr = {"c1": fb_c1, "c2": fb_c2}
        vis_paths = {}
        for o in cfg["objects"]:
            mj = asset_dir / f"sim_ready_{o['name']}" / "asset_meta.json"
            if mj.exists(): vis_paths[o["name"]] = Path(json.loads(mj.read_text())["visual_mesh_path"])

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
                if r is not None: demos.append(r)

        if len(demos) < min_demos: continue
        demos.sort(key=lambda x: x["ep_idx"])
        part = v4._demos_to_transitions_v4(demos, tspec)
        parts.append(part); active_tids.append(tid)

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
        print(f"[Collect v9 Task {tid:2d} {sid}] {len(demos)}/{len(pass_ids)} demos -> {len(part['act'])} chunks", flush=True)

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}
    np.savez_compressed(OUT_DIR / "uni_real_v9_23tasks.npz", **merged)
    (OUT_DIR / "task_meta_v9.json").write_text(json.dumps({"active_tids": active_tids, "tasks": task_meta}, indent=2))
    tot = sum(task_meta[str(t)]["n_real_demos"] for t in active_tids)
    print(f"[Collect v9 COMPLETE] {len(active_tids)} tasks | {tot} verified real demos -> {len(merged['act'])} real chunks in {time.perf_counter()-t0:.1f}s!", flush=True)


def stage_train_v9(
    pretrain_steps: int = 14000,
    anneal_steps: int = 5000,
    temp_alpha: float = 0.5,
    sim_ratio: float = 0.20,
    elite_pct: float = 55.0,
    pure_real: bool = False,
    ckpt_name: str = "uni_bridge_v9_42m.pt",
) -> None:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_specs = build_all_task_specs_v9()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((OUT_DIR / "task_meta_v9.json").read_text())
    active_tids = meta_obj["active_tids"]

    real_ds = dict(np.load(OUT_DIR / "uni_real_v9_23tasks.npz"))
    n_real = len(real_ds["act"])

    if pure_real:
        sim_ratio = 0.0
        pretrain_steps = 0
        anneal_steps = 16000
        n_sim = 0
        ds = real_ds
        is_sim_flag = np.zeros(n_real, dtype=bool)
        print(f"[v9-RealOnly Data] Pure Real={n_real} chunks ({len(active_tids)} tasks)", flush=True)
    else:
        sim_p = V8_DIR / "v8_sim_harvest_all17.npz"
        if not sim_p.exists():
            sim_p = V6_DIR / "v4_sim_harvest_11tasks.npz"
        sim = dict(np.load(sim_p))
        ep_len = sim["ep_len"]; stid = sim["tid"]
        keep = np.zeros(len(stid), dtype=bool)
        for t in sorted(np.unique(stid).tolist()):
            m_t = (stid == t) & (ep_len >= 35)
            if not m_t.any(): continue
            cut = float(np.percentile(ep_len[m_t], elite_pct))
            keep |= (m_t & (ep_len <= cut))
        sim_ds = {k: v[keep] for k, v in sim.items() if k != "ep_len"}
        n_sim = len(sim_ds["act"])
        print(f"[v9 Data] Real={n_real} chunks ({len(active_tids)} tasks) + EliteSim={n_sim} chunks (from {sim_p.name})", flush=True)
        ds = {k: np.concatenate([real_ds[k], sim_ds[k]], axis=0) for k in real_ds.keys()}
        is_sim_flag = np.concatenate([np.zeros(n_real, dtype=bool), np.ones(n_sim, dtype=bool)])

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
    tgt_cont_lut = torch.zeros((max_tid,), device=device, dtype=torch.float32)
    for t in active_tids:
        pick_z_lut[t] = float(spec_by_tid[t]["pick_z_off"])
        place_z_lut[t] = float(spec_by_tid[t]["place_z_off"])
        aniso_lut[t] = float(spec_by_tid[t]["aniso"])
        tgt_cont_lut[t] = float(spec_by_tid[t]["is_tgt_container"])

    pools_r = {t: torch.from_numpy(np.where((tid_np == t) & (~is_sim_flag))[0]).to(device) for t in active_tids}
    pools_s = {t: torch.from_numpy(np.where((tid_np == t) & is_sim_flag)[0]).to(device) for t in active_tids}

    bs = 640
    weights = np.array([float(meta_obj["tasks"][str(t)]["n_real_demos"]) ** temp_alpha for t in active_tids], dtype=np.float64)
    probs = weights / weights.sum()
    quotas = np.maximum(12, np.round(probs * bs).astype(int))
    diff = bs - int(quotas.sum())
    if diff != 0:
        quotas[np.argmax(quotas)] += diff
    quota_map = {t: int(q) for t, q in zip(active_tids, quotas)}
    step_ladder = (torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32) * 0.018)[None, :, None]

    def sample_batch_idx(use_sim: bool) -> torch.Tensor:
        chunks = []
        for t in active_tids:
            q = quota_map[t]
            pr, ps = pools_r[t], pools_s[t]
            if use_sim and len(ps) >= 32 and sim_ratio > 0.0:
                ns = max(1, int(round(q * sim_ratio))); nr = q - ns
                chunks.append(pr[torch.randint(0, len(pr), (nr,), device=device)])
                chunks.append(ps[torch.randint(0, len(ps), (ns,), device=device)])
            else:
                chunks.append(pr[torch.randint(0, len(pr), (q,), device=device)])
        return torch.cat(chunks, dim=0)

    def make_batch(idx: torch.Tensor):
        B = len(idx)
        tz_b = tz_t[idx]; tids = tid_t[idx]
        g_b = torch.cat([geom_t[idx], tgt_cont_lut[tids].unsqueeze(-1)], dim=-1)
        is_hol = g_b[:, 6]; an = aniso_lut[tids]

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

    d_model, n_layers, n_heads = 448, 9, 8
    model = v8.build_v8_dit(d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    with torch.no_grad():
        stat_idx = torch.cat([sample_batch_idx(False) for _ in range(10)], dim=0)
        _, _, a_stat = make_batch(stat_idx)
        act_mean = a_stat.mean(dim=(0, 1))
        act_std = a_stat.std(dim=(0, 1)).clamp(min=0.03)

    ema_model = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    total_steps = pretrain_steps + anneal_steps
    print(f"[v9 42M 23-Task Grand] {n_params/1e6:.2f}M params (d={d_model}, L={n_layers}) | StageA={pretrain_steps} -> StageB={anneal_steps}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2.6e-4, weight_decay=1e-4, fused=True)
    warmup = 350
    def lr_fn(s):
        if s < warmup: return float(s + 1) / float(warmup)
        if pretrain_steps == 0:
            p = float(s - warmup) / float(max(1, anneal_steps - warmup))
            return 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * p))
        if s < pretrain_steps:
            p = float(s - warmup) / float(max(1, pretrain_steps - warmup))
            return 0.22 + 0.78 * 0.5 * (1.0 + math.cos(math.pi * p))
        p = float(s - pretrain_steps) / float(max(1, anneal_steps))
        return 0.025 + 0.195 * 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    beta_dist = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
    ema_params = list(ema_model.parameters()); cur_params = list(model.parameters())
    w_dim = torch.tensor([1.3, 1.3, 1.5, 0.85, 0.85, 1.0, 2.2], device=device)[None, None, :]

    t0 = time.perf_counter()
    for step in range(1, total_steps + 1):
        use_sim = (step <= pretrain_steps)
        with torch.no_grad():
            obs_b, al_b, ae_b = make_batch(sample_batch_idx(use_sim))
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
            torch._foreach_lerp_(ema_params, cur_params, 0.005 if use_sim else 0.010)
        if step % 2500 == 0 or step == 1 or step == total_steps:
            ph = "A-Mix " if use_sim else "B-Real"
            print(f"  [v9 {ph} step {step:5d}/{total_steps}] loss={loss.item():.4f} | {(time.perf_counter()-t0)/step*1000:.1f} ms/step ({time.perf_counter()-t0:.1f}s)", flush=True)

    train_dt = time.perf_counter() - t0
    with torch.no_grad():
        obs_v, al_v, ae_v = make_batch(sample_batch_idx(False))
        pred = ema_model.render_chunk(((al_v - act_mean) / act_std).clamp(-6, 6), obs_v, n_steps=ODE_STEPS, seed=42) * act_std + act_mean
        mae_cm = torch.mean(torch.abs(pred[:, :6, :3] - ae_v[:, :6, :3])).item() * 100.0
        g_mae = torch.mean(torch.abs(pred[:, :6, 6] - ae_v[:, :6, 6])).item()
        print(f"[v9 COMPLETE] {train_dt:.1f}s | 4-step ODE PosMAE={mae_cm:.2f}cm | GripMAE={g_mae:.3f}", flush=True)

    torch.save({
        "ema_state_dict": ema_model.state_dict(),
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "obs_dim": OBS_DIM,
        "act_mean": act_mean.cpu().numpy(), "act_std": act_std.cpu().numpy(),
        "active_tids": active_tids, "n_real_chunks": n_real, "n_sim_chunks": n_sim,
        "train_sec": round(train_dt, 1), "val_pos_mae_cm": round(mae_cm, 2), "val_grip_mae": round(g_mae, 3),
    }, OUT_DIR / ckpt_name)


def stage_serve_v9(port: int = PORT_V9, ckpt_name: str = "uni_bridge_v9_42m.pt") -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / ckpt_name, map_location=device, weights_only=False)
    model = v8.build_v8_dit(int(ck["d_model"]), int(ck["n_layers"]), int(ck["n_heads"])).to(device).eval()
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
    print(f"[Serve v9] READY on 127.0.0.1:{port} (d={ck['d_model']}, L={ck['n_layers']}, OBS=48)", flush=True)
    try:
        while not stop_flag.is_set():
            try:
                conn, _ = srv.accept()
                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
    finally:
        srv.close()


def stage_eval_v9(scope: str = "orig5", port: int = PORT_V9, random_yaw: bool = False, tag_prefix: str = "v9_42m") -> None:
    t0 = time.perf_counter()
    n_w = int(os.environ.get("R2S_WORKERS", "24"))
    all_specs = build_all_task_specs_v9()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((OUT_DIR / "task_meta_v9.json").read_text())

    if scope == "orig5":
        eval_tids = [0, 1, 2, 3, 4]; rng = np.random.default_rng(2026)
        meta_src = json.loads((uni.OUT_DIR / "task_meta.json").read_text())
    elif scope == "new6":
        eval_tids = [t for t in meta_obj["active_tids"] if 5 <= t <= 10]; rng = np.random.default_rng(2027)
        meta_src = meta_obj["tasks"]
    elif scope == "brandnew6":
        eval_tids = [t for t in meta_obj["active_tids"] if 11 <= t <= 16]; rng = np.random.default_rng(2028)
        meta_src = meta_obj["tasks"]
    else:
        eval_tids = [t for t in meta_obj["active_tids"] if t >= 17]; rng = np.random.default_rng(2029)
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
        for f in as_completed([pool.submit(v8._run_one_v8_ep, x) for x in tasks]):
            results.append(f.result())

    v8._shutdown_v8(port)

    by_task = {}
    for t in eval_tids:
        ts = spec_by_tid[t]
        rs = [r for r in results if r["tid"] == t]
        ok = sum(int(r["success"]) for r in rs)
        by_task[ts["name"]] = {"pass": ok, "n": len(rs), "rate": round(100.0 * ok / max(1, len(rs)), 1)}

    tot_ok = sum(int(r["success"]) for r in results); tot_n = len(results)
    full_tag = f"{tag_prefix}_{scope}" + ("_randyaw" if random_yaw else "")
    out = {"tag": full_tag, "pass": tot_ok, "n": tot_n, "rate": round(100.0 * tot_ok / max(1, tot_n), 1), "per_task": by_task, "elapsed_sec": round(time.perf_counter() - t0, 1)}
    (OUT_DIR / f"eval_{full_tag}.json").write_text(json.dumps(out, indent=2))
    print("=" * 92, flush=True)
    print(f"[EVAL {full_tag}] TOTAL = {tot_ok}/{tot_n} ({out['rate']:.1f}%) in {out['elapsed_sec']}s", flush=True)
    for k, v in by_task.items():
        print(f"  - {k:28s}: {v['pass']:2d}/{v['n']:2d} ({v['rate']:5.1f}%)", flush=True)
    print("=" * 92, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "train", "serve", "eval"])
    ap.add_argument("--scope", choices=["orig5", "new6", "brandnew6", "wave4"], default="orig5")
    ap.add_argument("--port", type=int, default=PORT_V9)
    ap.add_argument("--random-yaw", action="store_true")
    ap.add_argument("--pure-real", action="store_true")
    ap.add_argument("--ckpt", default="uni_bridge_v9_42m.pt")
    ap.add_argument("--tag-prefix", default="v9_42m")
    args = ap.parse_args()
    if args.cmd == "collect":
        stage_collect_v9()
    elif args.cmd == "train":
        ckpt = "uni_bridge_v9_real_42m.pt" if (args.pure_real and args.ckpt == "uni_bridge_v9_42m.pt") else args.ckpt
        stage_train_v9(pure_real=args.pure_real, ckpt_name=ckpt)
    elif args.cmd == "serve":
        stage_serve_v9(port=args.port, ckpt_name=args.ckpt)
    elif args.cmd == "eval":
        stage_eval_v9(scope=args.scope, port=args.port, random_yaw=args.random_yaw, tag_prefix=args.tag_prefix)


if __name__ == "__main__":
    main()
