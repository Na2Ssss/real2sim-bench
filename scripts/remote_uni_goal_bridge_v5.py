"""Uni-Goal-Bridge v5: Full-Wrist SE(2) Observation (OBS=51) + All-11-Task Crisp Sim Flywheel!

Key Breakthrough over v4 (see `docs/REAL2SIM_PIPELINE_ISSUES.md` R2S-11):
  - In v4 (`OBS=47`), `t_ee` only included `R_ee[:, 2]` (downward Z-axis) -> the wrist was BLIND to its own
    current finger-closing yaw, causing 16-28% on markers at `yaw=0` vs 56-60% when `q_home` happened to align!
  - In v5 (`OBS=51`), `t_ee` (`9D -> 13D`) includes:
      1. `ee_x_ax = R_ee[:, 0]` (3D gripper finger-closing axis, rotated identically by `Rz(th)` in 360° SO(2) aug!)
      2. `align_cue = aniso * abs(dot(ee_x_xy, u_src_xy))` (direct scalar alignment cue: 0 = fingers perpendicular
         to marker barrel [IDEAL GRASP!], 1 = fingers parallel to barrel [MUST ROTATE 90°!])
  - Reuses our clean 448-demo / 55,648-chunk `uni_real_v4_11tasks.npz` (which already stores full `R_ee` & `u_src`!).
  - Full closed-loop Flywheel across ALL 11 TASKS x 7 LABS:
      1. `train_real`: Train 19.34M `v5_real` (8,500 steps, pure real 448 demos)
      2. `eval`: Evaluate `v5_real` on `orig5` (125), `new6` (150), and `new6_randyaw` (150)
      3. `harvest_11`: Rollout `v5_real` under 2-step DLS-IK across 2,200 brand-new OOD poses (200/task x 11 tasks,
         disjoint seed=88888, random 360° yaw on anisotropic objects!) -> keep exact trajectory lengths!
      4. `train_mix`: Per-task top-55% fastest elite filtering + 80% Real : 20% Elite-Sim -> train `v5_mix` & re-eval!
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
OUT_DIR = ROOT / "outputs" / "uni_goal_bridge_v5"
DT_CTRL = uni.DT_CTRL
HORIZON = uni.HORIZON
ACT_DIM = uni.ACT_DIM
OBS_DIM = 51  # 47 + 4 (ee_x_ax [3] + aniso*abs(dot(ee_x_xy, u_src_xy)) [1])!
BRIDGE_SIGMA = uni.BRIDGE_SIGMA
ODE_STEPS = uni.ODE_STEPS
PORT_V5 = 18968


def build_obs_v5_np(
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

    an = v4.geom_aniso_from_vec(geom_vec)
    un = float(np.linalg.norm(u_src_xy))
    ux, uy = (float(u_src_xy[0] / un), float(u_src_xy[1] / un)) if un > 1e-5 else (0.0, 1.0)
    ori_feats = np.array([an * ux, an * uy, an * (ux * ux - uy * uy), an * (2.0 * ux * uy)], dtype=np.float32)  # 4
    t_ls = np.concatenate([dir_sat(d_src, 0.12), [float(p_src[2] - table_z)], ori_feats])             # 11

    t_lt = np.concatenate([dir_sat(d_tgt, 0.12), [math.tanh(float(np.linalg.norm(d_st[:2])) / 0.12), float(np.clip(d_st[2], -0.2, 0.2))]])  # 8
    z_ax = R_ee[:, 2]
    x_ax = R_ee[:, 0]
    xn_xy = float(np.linalg.norm(x_ax[:2]))
    ex_x, ex_y = (float(x_ax[0] / xn_xy), float(x_ax[1] / xn_xy)) if xn_xy > 1e-5 else (1.0, 0.0)
    align_cue = float(an * abs(ex_x * ux + ex_y * uy))  # 0 = perpendicular (good!), 1 = parallel (rotate 90!)

    is_hol = float(geom_vec[6])
    d_s_xy = float(np.linalg.norm(d_src[:2]))
    held_cue = float(g_st * math.exp(-((d_s_xy / (0.055 + 0.035 * is_hol)) ** 2)))
    t_ee = np.array([
        float(p_ee[2] - table_z), z_ax[0], z_ax[1], z_ax[2],
        x_ax[0], x_ax[1], x_ax[2], align_cue,
        float(g_st), ee_vel[0], ee_vel[1], ee_vel[2], held_cue,
    ], dtype=np.float32)                                                                              # 13
    return np.concatenate([t_vp, t_vb, t_ls, t_lt, t_ee, geom_vec], axis=0)                           # 51


def build_v5_dit(d_model: int = 320, n_layers: int = 8, n_heads: int = 8):
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

    class UniGoalBridgeDiTV5(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_vp = nn.Sequential(nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_vb = nn.Sequential(nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_ls = nn.Sequential(nn.Linear(11, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_lt = nn.Sequential(nn.Linear(8, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.enc_ee = nn.Sequential(nn.Linear(13, d_model), nn.SiLU(), nn.Linear(d_model, d_model))  # 9 + 4 wrist axis!
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
                self.enc_ee(obs[:, 31:44]), self.enc_gm(obs[:, 44:51]),
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

    return UniGoalBridgeDiTV5()


def filter_sim_elite_per_task(sim_path: Path, elite_pct: float = 55.0) -> Tuple[dict, dict]:
    sim = dict(np.load(sim_path))
    ep_len = sim["ep_len"]
    tid = sim["tid"]
    keep_mask = np.zeros(len(tid), dtype=bool)
    stats = {}
    for t in sorted(np.unique(tid).tolist()):
        m_t = (tid == t) & (ep_len >= 35)
        lens_t = ep_len[m_t]
        if len(lens_t) == 0:
            continue
        cut = float(np.percentile(lens_t, elite_pct))
        sel_t = m_t & (ep_len <= cut)
        keep_mask |= sel_t
        stats[str(t)] = {"kept_chunks": int(sel_t.sum()), "total_chunks": int(m_t.sum()), "cutoff_ticks": round(cut, 1)}
        print(f"  [v5 Elite Filter tid={t:2d}] kept {int(sel_t.sum()):5d}/{int(m_t.sum()):5d} chunks (cutoff<={cut:.1f}t)", flush=True)
    out = {k: v[keep_mask] for k, v in sim.items() if k != "ep_len"}
    return out, stats


def stage_train_v5(mode: str = "real", n_steps: int = 8500, sim_ratio: float = 0.20, elite_pct: float = 55.0) -> None:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_specs = v4.build_all_task_specs_v4()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((V4_DIR / "task_meta_v4.json").read_text())
    active_tids = meta_obj["active_tids"]

    real_ds = dict(np.load(V4_DIR / "uni_real_v4_11tasks.npz"))
    n_real = len(real_ds["act"])
    if mode == "mix":
        sim_ds, elite_stats = filter_sim_elite_per_task(OUT_DIR / "v5_sim_harvest_2200.npz", elite_pct=elite_pct)
        ds = {k: np.concatenate([real_ds[k], sim_ds[k]], axis=0) for k in real_ds.keys()}
        n_sim = len(sim_ds["act"])
        is_sim_flag = np.concatenate([np.zeros(n_real, dtype=bool), np.ones(n_sim, dtype=bool)])
        ckpt_name = "uni_bridge_v5_mix.pt"
    else:
        ds = real_ds; n_sim = 0; elite_stats = {}; is_sim_flag = np.zeros(n_real, dtype=bool)
        ckpt_name = "uni_bridge_v5_real.pt"

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
        z_ax = R_ee[idx, :, 2]; x_ax = R_ee[idx, :, 0]; v_cur = vel[idx]

        # Exact 360-deg SO(2) planar rotation of ALL vectors including gripper finger-closing axis x_ax AND object axis u_src!
        th = torch.rand((B,), device=device) * (2.0 * math.pi)
        ct, st = torch.cos(th), torch.sin(th)
        Rz = torch.zeros((B, 3, 3), device=device)
        Rz[:, 0, 0] = ct; Rz[:, 0, 1] = -st
        Rz[:, 1, 0] = st; Rz[:, 1, 1] = ct
        Rz[:, 2, 2] = 1.0
        def rot3(v): return torch.bmm(Rz, v.unsqueeze(-1)).squeeze(-1)

        d_src = rot3(d_src); d_tgt = rot3(d_tgt); d_st = rot3(d_st)
        d_vp = rot3(d_vp); d_vb = rot3(d_vb); z_ax = rot3(z_ax); x_ax = rot3(x_ax); v_cur = rot3(v_cur)
        a_b[:, :, :3] = torch.bmm(a_b[:, :, :3], Rz.transpose(1, 2))
        a_b[:, :, 3:6] = torch.bmm(a_b[:, :, 3:6], Rz.transpose(1, 2))

        u0 = u_src[idx]
        ux = ct * u0[:, 0] - st * u0[:, 1]; uy = st * u0[:, 0] + ct * u0[:, 1]
        un = torch.sqrt(ux * ux + uy * uy).clamp(min=1e-5)
        ux = ux / un; uy = uy / un
        sgn = torch.where(torch.rand((B,), device=device) > 0.5, 1.0, -1.0)
        ori_feats = torch.stack([an * ux * sgn, an * uy * sgn, an * (ux * ux - uy * uy), an * (2.0 * ux * uy)], dim=-1)

        xn = torch.sqrt(x_ax[:, 0] ** 2 + x_ax[:, 1] ** 2).clamp(min=1e-5)
        align_cue = (an * torch.abs((x_ax[:, 0] / xn) * ux + (x_ax[:, 1] / xn) * uy)).unsqueeze(-1)

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
        t_ee = torch.cat([(p_ee_q[:, 2] - tz_b).unsqueeze(-1), z_ax, x_ax, align_cue, g_st[idx].unsqueeze(-1), v_cur, held_cue], dim=-1)
        obs_b = torch.cat([t_vp, t_vb, t_ls, t_lt, t_ee, g_b], dim=-1)
        return obs_b, a_line, a_b

    model = build_v5_dit(d_model=320, n_layers=8, n_heads=8).to(device)
    with torch.no_grad():
        _, _, a_stat = make_batch(sample_balanced_idx(6000))
        act_mean = a_stat.mean(dim=(0, 1))
        act_std = a_stat.std(dim=(0, 1)).clamp(min=0.03)

    ema_model = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v5_{mode}] {n_params/1e6:.2f}M params (d=320, L=8, OBS=51 FullWrist-SE2) | {n_steps} steps | Real={n_real} + EliteSim={n_sim}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1e-4, fused=True)
    warmup = 250; bs = 528
    def lr_fn(s):
        if s < warmup: return float(s + 1) / float(warmup)
        p = float(s - warmup) / float(max(1, n_steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    beta_dist = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
    ema_params = list(ema_model.parameters()); cur_params = list(model.parameters())
    w_dim = torch.tensor([1.3, 1.3, 1.5, 0.9, 0.9, 1.2, 2.2], device=device)[None, None, :]  # strong yaw weight 1.2!

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
        if step % 1500 == 0 or step == 1 or step == n_steps:
            print(f"  [v5_{mode} step {step:4d}/{n_steps}] loss={loss.item():.4f} | {(time.perf_counter()-t0)/step*1000:.1f} ms/step ({time.perf_counter()-t0:.1f}s)", flush=True)

    train_dt = time.perf_counter() - t0
    with torch.no_grad():
        obs_v, al_v, ae_v = make_batch(sample_balanced_idx(550))
        pred = ema_model.render_chunk(((al_v - act_mean) / act_std).clamp(-6, 6), obs_v, n_steps=ODE_STEPS, seed=42) * act_std + act_mean
        mae_cm = torch.mean(torch.abs(pred[:, :6, :3] - ae_v[:, :6, :3])).item() * 100.0
        g_mae = torch.mean(torch.abs(pred[:, :6, 6] - ae_v[:, :6, 6])).item()
        print(f"[v5_{mode} COMPLETE] {train_dt:.1f}s | 4-step ODE PosMAE={mae_cm:.2f}cm | GripMAE={g_mae:.3f}", flush=True)

    torch.save({
        "ema_state_dict": ema_model.state_dict(),
        "d_model": 320, "n_layers": 8, "n_heads": 8, "obs_dim": OBS_DIM,
        "act_mean": act_mean.cpu().numpy(), "act_std": act_std.cpu().numpy(),
        "active_tids": active_tids, "n_real_chunks": n_real, "n_sim_chunks": n_sim,
        "elite_stats": elite_stats, "train_sec": round(train_dt, 1),
        "val_pos_mae_cm": round(mae_cm, 2), "val_grip_mae": round(g_mae, 3),
    }, OUT_DIR / ckpt_name)


def stage_serve_v5(ckpt_name: str, port: int = PORT_V5) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / ckpt_name, map_location=device, weights_only=False)
    model = build_v5_dit(int(ck["d_model"]), int(ck["n_layers"]), int(ck["n_heads"])).to(device).eval()
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
    print(f"[Serve v5 {ckpt_name}] READY on 127.0.0.1:{port} (OBS=51 FullWrist-SE2)", flush=True)
    try:
        while not stop_flag.is_set():
            try:
                conn, _ = srv.accept()
                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
    finally:
        srv.close()


class V5BridgeClient:
    def __init__(self, port: int = PORT_V5):
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


def _shutdown_v5(port: int = PORT_V5) -> None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s.sendall(struct.pack("<i", -999999) + b"\x00" * ((OBS_DIM + HORIZON * ACT_DIM) * 4))
        s.close()
    except Exception:
        pass


def _run_one_v5_ep(args: tuple) -> dict:
    import mujoco
    from r2s_bench.eval.task_success import RolloutMonitor, evaluate_object_in_container, evaluate_object_on_target

    tspec, ep_id, src_xy, tgt_xy, src_yaw, tz, q_home, vlm_ec, vlm_eb, seed, port, record_traj = args
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
    client = V5BridgeClient(port=port)

    max_ticks = 210; exec_k = 6
    q_cmd = q_ref.copy(); p_ee_prev = 0.5 * (d.xpos[lp] + d.xpos[rp]).copy()
    chunk_p_targets = None; chunk_R_targets = None; chunk_g_targets = None; chunk_idx = exec_k
    success_hold = 0
    rec_p_ee, rec_R_ee, rec_g_st, rec_g_cmd, rec_p_src, rec_p_tgt, rec_u_src = [], [], [], [], [], [], []

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
            obs = build_obs_v5_np(p_ee, R_ee, g_st, ee_vel, p_src, p_tgt, u_src, p_pick_vlm, p_place_vlm, tz, gv)
            a_line = uni.make_straight_line_prior_np(p_ee, p_pick_vlm, p_place_vlm, g_st)
            act_chunk = client.query(obs, a_line, seed=seed + tick * 7)
            chunk_p_targets = p_ee[None, :] + act_chunk[:, :3]
            chunk_R_targets = [Rotation.from_rotvec(act_chunk[k, 3:6]).as_matrix() @ R_ee for k in range(HORIZON)]
            chunk_g_targets = np.clip(act_chunk[:, 6], 0.0, 1.0)
            chunk_idx = 0

        p_des = chunk_p_targets[chunk_idx]; R_des = chunk_R_targets[chunk_idx]; g_des = float(chunk_g_targets[chunk_idx])
        chunk_idx += 1

        if record_traj:
            rec_p_ee.append(p_ee.astype(np.float32)); rec_R_ee.append(R_ee.astype(np.float32))
            rec_g_st.append(np.float32(g_st)); rec_g_cmd.append(np.float32(g_des))
            rec_p_src.append(p_src.astype(np.float32)); rec_p_tgt.append(p_tgt.astype(np.float32))
            rec_u_src.append(u_src)

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

    for _ in range(45):
        d.ctrl[:7] = q_cmd; d.ctrl[ag] = 0.0; mujoco.mj_step(m, d)
    mon.observe(d)
    client.close()

    if oc["type"] == "in_container":
        res = evaluate_object_in_container(m, d, mon, tspec["src_body"], tspec["tgt_body"], max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [-0.02, 0.12]), table_z=tz)
    else:
        res = evaluate_object_on_target(m, d, mon, tspec["src_body"], tspec["tgt_body"], max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [0.005, 0.14]))

    out = {"tid": tspec["tid"], "ep_id": ep_id, "success": bool(res.success), "ticks": len(rec_p_ee)}
    if record_traj and res.success and len(rec_p_ee) >= 35:
        out["traj"] = {
            "tid": tspec["tid"], "table_z": tz,
            "p_ee": np.stack(rec_p_ee), "R_ee": np.stack(rec_R_ee),
            "g_state": np.asarray(rec_g_st), "g_cmd": np.asarray(rec_g_cmd),
            "p_src": np.stack(rec_p_src), "p_tgt": np.stack(rec_p_tgt),
            "u_src": np.stack(rec_u_src),
        }
    return out


def stage_eval_v5(tag: str, scope: str = "orig5", port: int = PORT_V5, random_yaw: bool = False) -> None:
    t0 = time.perf_counter()
    n_w = int(os.environ.get("R2S_WORKERS", "24"))
    all_specs = v4.build_all_task_specs_v4()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((V4_DIR / "task_meta_v4.json").read_text())

    if scope == "orig5":
        orig_meta = json.loads((uni.OUT_DIR / "task_meta.json").read_text())
        eval_tids = [0, 1, 2, 3, 4]; rng = np.random.default_rng(2026)
        meta_src = orig_meta
    else:
        eval_tids = [t for t in meta_obj["active_tids"] if t >= 5]; rng = np.random.default_rng(2027)
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
            tasks.append((ts, i, sxy, txy, syaw, tz, qh, ec, eb, 90000 + t * 1000 + i * 17, port, False))

    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
        for f in as_completed([pool.submit(_run_one_v5_ep, x) for x in tasks]):
            results.append(f.result())

    _shutdown_v5(port)

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


def stage_harvest_11(n_per_task: int = 200, port: int = PORT_V5) -> None:
    t0 = time.perf_counter()
    n_w = int(os.environ.get("R2S_WORKERS", "28"))
    all_specs = v4.build_all_task_specs_v4()
    spec_by_tid = {s["tid"]: s for s in all_specs}
    meta_obj = json.loads((V4_DIR / "task_meta_v4.json").read_text())
    active_tids = meta_obj["active_tids"]

    rng = np.random.default_rng(88888)  # Disjoint harvest seed!
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
            tasks.append((ts, i, sxy, txy, syaw, tz, qh, ec, eb, 700000 + t * 10000 + i * 31, port, True))

    print(f"[v5 Harvest 11-Tasks] Launching {len(tasks)} OOD rollouts ({n_per_task}/task x {len(active_tids)} tasks)...", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
        futs = [pool.submit(_run_one_v5_ep, x) for x in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if k % 400 == 0 or k == len(tasks):
                ok_n = sum(int(r["success"]) for r in results)
                print(f"  [Harvest {k:4d}/{len(tasks)}] PASS={ok_n} ({100*ok_n/k:.1f}%) ({time.perf_counter()-t0:.1f}s)", flush=True)

    _shutdown_v5(port)

    parts, cnts = [], {}
    for t in active_tids:
        ts = spec_by_tid[t]
        ok_rs = [r for r in results if r["tid"] == t and r["success"] and "traj" in r]
        cnts[ts["name"]] = len(ok_rs)
        if not ok_rs: continue
        for r in ok_rs:
            p = v4._demos_to_transitions_v4([r["traj"]], ts)
            p["ep_len"] = np.full((len(p["act"]),), int(r["ticks"]), dtype=np.int32)
            parts.append(p)

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}
    np.savez_compressed(OUT_DIR / "v5_sim_harvest_2200.npz", **merged)
    summary = {"total_rollouts": len(tasks), "passing_trajs": sum(cnts.values()), "total_chunks": int(len(merged["act"])), "per_task": cnts, "elapsed_sec": round(time.perf_counter() - t0, 1)}
    (OUT_DIR / "v5_harvest_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[v5 HARVEST COMPLETE] {sum(cnts.values())}/{len(tasks)} passing trajs -> {len(merged['act'])} chunks in {summary['elapsed_sec']}s | {cnts}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train_real", "train_mix", "serve", "eval", "harvest"])
    ap.add_argument("--ckpt", default="uni_bridge_v5_real.pt")
    ap.add_argument("--tag", default="v5_real")
    ap.add_argument("--scope", choices=["orig5", "new6"], default="orig5")
    ap.add_argument("--port", type=int, default=PORT_V5)
    ap.add_argument("--steps", type=int, default=8500)
    ap.add_argument("--n-per-task", type=int, default=200)
    ap.add_argument("--sim-ratio", type=float, default=0.20)
    ap.add_argument("--elite-pct", type=float, default=55.0)
    ap.add_argument("--random-yaw", action="store_true")
    args = ap.parse_args()

    if args.cmd == "train_real":
        stage_train_v5("real", n_steps=args.steps)
    elif args.cmd == "train_mix":
        stage_train_v5("mix", n_steps=args.steps, sim_ratio=args.sim_ratio, elite_pct=args.elite_pct)
    elif args.cmd == "serve":
        stage_serve_v5(args.ckpt, port=args.port)
    elif args.cmd == "eval":
        stage_eval_v5(args.tag, scope=args.scope, port=args.port, random_yaw=args.random_yaw)
    elif args.cmd == "harvest":
        stage_harvest_11(n_per_task=args.n_per_task, port=args.port)


if __name__ == "__main__":
    main()
