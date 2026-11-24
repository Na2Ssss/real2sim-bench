"""Uni-Goal-Bridge v7: 17-Task x 11-Lab 27.7M DiT with Temperature Sampling + Sim-Pretrain -> Real-Anneal!

Solves R2S-13 discovered in v6:
  1. Temperature-Balanced Task Sampling (`p_t proportional to N_real_t ** 0.5`):
     Prevents tiny 5-7 demo scenes (`demo18`, `demo19`) from drowning out precision high-N scenes (`demo7`, `demo1`, `demo20`)
     while still learning all 17 tasks x 11 robotics labs in a single zero-task-ID model!
  2. Two-Stage Training Recipe (Mid-Training -> Precision Annealing):
     - Stage A (12,000 steps, lr=3e-4 -> 6e-5, bs=576): 80% Real (561 demos, 75,907 chunks) + 20% Top-50% Ultra-Crisp v4-Sim
       (1,057 harvested OOD trajs across Tasks 0..10) -> broad workspace/yaw generalization & zero-shot cross-task transfer!
     - Stage B (4,000 steps, lr=7e-5 -> 8e-6, bs=576): 100% PURE REAL (561 demos across all 17 tasks x 11 labs)
       -> eliminates sim-distillation softening and sharpens contact/gripper timing back to `GripMAE ~ 0.035`!
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import numpy as np

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

import scripts.remote_uni_goal_bridge_scaling as uni
import scripts.remote_uni_goal_bridge_v4 as v4
import scripts.remote_uni_goal_bridge_v6 as v6

V6_DIR = v6.OUT_DIR
OUT_DIR = ROOT / "outputs" / "uni_goal_bridge_v7"
HORIZON = uni.HORIZON
ACT_DIM = uni.ACT_DIM
OBS_DIM = v4.OBS_DIM  # 47
BRIDGE_SIGMA = uni.BRIDGE_SIGMA
ODE_STEPS = uni.ODE_STEPS
PORT_V7 = 18976


def stage_train_v7(pretrain_steps: int = 12000, anneal_steps: int = 4000, temp_alpha: float = 0.5, sim_ratio: float = 0.20, elite_pct: float = 50.0) -> None:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_specs = v6.build_all_task_specs_v6()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((V6_DIR / "task_meta_v6.json").read_text())
    active_tids = meta_obj["active_tids"]

    real_ds = dict(np.load(V6_DIR / "uni_real_v6_17tasks.npz"))
    n_real = len(real_ds["act"])

    sim = dict(np.load(V6_DIR / "v4_sim_harvest_11tasks.npz"))
    ep_len = sim["ep_len"]; stid = sim["tid"]
    keep = np.zeros(len(stid), dtype=bool)
    for t in sorted(np.unique(stid).tolist()):
        m_t = (stid == t) & (ep_len >= 35)
        if not m_t.any(): continue
        cut = float(np.percentile(ep_len[m_t], elite_pct))
        keep |= (m_t & (ep_len <= cut))
    sim_ds = {k: v[keep] for k, v in sim.items() if k != "ep_len"}
    n_sim = len(sim_ds["act"])

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
    for t in active_tids:
        pick_z_lut[t] = float(spec_by_tid[t]["pick_z_off"])
        place_z_lut[t] = float(spec_by_tid[t]["place_z_off"])
        aniso_lut[t] = float(spec_by_tid[t]["aniso"])

    pools_r = {t: torch.from_numpy(np.where((tid_np == t) & (~is_sim_flag))[0]).to(device) for t in active_tids}
    pools_s = {t: torch.from_numpy(np.where((tid_np == t) & is_sim_flag)[0]).to(device) for t in active_tids}

    # Temperature-balanced per-task quotas: proportional to (n_real_demos ** temp_alpha)!
    bs = 576
    weights = np.array([float(meta_obj["tasks"][str(t)]["n_real_demos"]) ** temp_alpha for t in active_tids], dtype=np.float64)
    probs = weights / weights.sum()
    quotas = np.maximum(12, np.round(probs * bs).astype(int))
    diff = bs - int(quotas.sum())
    if diff != 0:
        quotas[np.argmax(quotas)] += diff
    quota_map = {t: int(q) for t, q in zip(active_tids, quotas)}
    print(f"[v7 Temperature Quotas (alpha={temp_alpha}, bs={sum(quota_map.values())})]: {quota_map}", flush=True)

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
        stat_idx = torch.cat([sample_batch_idx(False) for _ in range(10)], dim=0)
        _, _, a_stat = make_batch(stat_idx)
        act_mean = a_stat.mean(dim=(0, 1))
        act_std = a_stat.std(dim=(0, 1)).clamp(min=0.03)

    ema_model = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    total_steps = pretrain_steps + anneal_steps
    print(f"[v7 Sim->Real Anneal] {n_params/1e6:.2f}M params (d={d_model}, L={n_layers}) | StageA={pretrain_steps} (80:20) -> StageB={anneal_steps} (100% Real)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1e-4, fused=True)
    warmup = 300
    def lr_fn(s):
        if s < warmup:
            return float(s + 1) / float(warmup)
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
        if step == pretrain_steps + 1:
            print(f"  >>> [v7 ENTERING STAGE B: 100% PURE REAL PRECISION ANNEALING ({anneal_steps} steps)] <<<", flush=True)
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
        ema_alpha = 0.005 if use_sim else 0.010  # slightly faster EMA tracking during real-only annealing!
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, cur_params, ema_alpha)
        if step % 2000 == 0 or step == 1 or step == total_steps:
            ph = "A-Mix " if use_sim else "B-Real"
            print(f"  [v7 {ph} step {step:5d}/{total_steps}] loss={loss.item():.4f} | {(time.perf_counter()-t0)/step*1000:.1f} ms/step ({time.perf_counter()-t0:.1f}s)", flush=True)

    train_dt = time.perf_counter() - t0
    with torch.no_grad():
        obs_v, al_v, ae_v = make_batch(sample_batch_idx(False))
        pred = ema_model.render_chunk(((al_v - act_mean) / act_std).clamp(-6, 6), obs_v, n_steps=ODE_STEPS, seed=42) * act_std + act_mean
        mae_cm = torch.mean(torch.abs(pred[:, :6, :3] - ae_v[:, :6, :3])).item() * 100.0
        g_mae = torch.mean(torch.abs(pred[:, :6, 6] - ae_v[:, :6, 6])).item()
        print(f"[v7 COMPLETE] {train_dt:.1f}s | 4-step ODE PosMAE={mae_cm:.2f}cm | GripMAE={g_mae:.3f}", flush=True)

    torch.save({
        "ema_state_dict": ema_model.state_dict(),
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "obs_dim": OBS_DIM,
        "act_mean": act_mean.cpu().numpy(), "act_std": act_std.cpu().numpy(),
        "active_tids": active_tids, "n_real_chunks": n_real, "n_sim_chunks": n_sim,
        "quota_map": quota_map, "train_sec": round(train_dt, 1),
        "val_pos_mae_cm": round(mae_cm, 2), "val_grip_mae": round(g_mae, 3),
    }, OUT_DIR / "uni_bridge_v7_anneal.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "serve", "eval"])
    ap.add_argument("--scope", choices=["orig5", "new6", "brandnew6"], default="orig5")
    ap.add_argument("--port", type=int, default=PORT_V7)
    ap.add_argument("--random-yaw", action="store_true")
    args = ap.parse_args()

    if args.cmd == "train":
        stage_train_v7()
    elif args.cmd == "serve":
        v6.OUT_DIR = OUT_DIR
        v6.stage_serve_v6("uni_bridge_v7_anneal.pt", port=args.port)
    elif args.cmd == "eval":
        # Load task_meta_v6 from V6_DIR, write eval outputs to OUT_DIR
        v6.OUT_DIR = V6_DIR
        v6.stage_eval_v6("v7_anneal", scope=args.scope, port=args.port, random_yaw=args.random_yaw)


if __name__ == "__main__":
    main()
