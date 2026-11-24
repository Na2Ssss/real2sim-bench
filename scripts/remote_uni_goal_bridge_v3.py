"""Uni-Goal-Bridge v3: Grand Multi-Lab Scaling (11 Tasks x 7 Labs + Per-Task Elite Sim Flywheel).

Extends `scripts/remote_uni_goal_bridge_scaling.py` (Tasks 0..4) with:
  1. Tasks 5..10 from brand-new real DROID scenes Demo #11-#16 (+3 new labs: IRIS, CLVR, RAIL + AUTOLab):
       - Task 5  (AUTOLab): Green Block -> Black Bowl          (demo11)
       - Task 6  (AUTOLab): Blue Cup -> Orange Cup             (demo12)
       - Task 7  (IRIS #8): Marker -> Target Bowl              (demo13)
       - Task 8  (CLVR):    Pen/Marker -> Target Mug           (demo14)
       - Task 9  (RAIL #9): Yellow Block -> Orange Plate       (demo15)
       - Task 10 (RAIL #9): Marker -> Target Pot               (demo16)
  2. Per-Task Top-P% Elite Filtering on `uni_sim_harvest_1500_all.npz` (704 OOD trajs, 115,704 chunks):
       - Reconstructs exact trajectory slices & tick lengths per task
       - Keeps fastest `elite_pct` (default 60%) per task so slower-by-nature tasks (demo7/demo10) are preserved
       - Mixes at Golden Ratio (75% Real : 25% Elite-Sim) to eliminate self-distillation sharpness dilution!
  3. Trains & Evaluates:
       - `v3_5t`    (10M DiT, controlled 5-task Real + Per-Task Elite Sim 75:25) on exact 125 held-out poses (seed=2026)
       - `v3_grand` (19M DiT, 11-Task x 7-Lab Real + Per-Task Elite Sim 75:25) on 125 original + 150 new-task OOD poses!
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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import numpy as np

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

import scripts.remote_uni_goal_bridge_scaling as uni

OUT_DIR = uni.OUT_DIR
HORIZON = uni.HORIZON
ACT_DIM = uni.ACT_DIM
BRIDGE_SIGMA = uni.BRIDGE_SIGMA
ODE_STEPS = uni.ODE_STEPS

NEW_SCENE_CFGS = [
    (5, "demo11_auto_green_block", "6. AUTOLab | Green Block -> Black Bowl", "configs/scenes/demo11_auto_green_block.json"),
    (6, "demo12_auto_cup_in_cup", "7. AUTOLab | Blue Cup -> Orange Cup", "configs/scenes/demo12_auto_cup_in_cup.json"),
    (7, "demo13_iris_marker_bowl", "8. IRIS | Marker -> Target Bowl", "configs/scenes/demo13_iris_marker_bowl.json"),
    (8, "demo14_clvr_pen_mug", "9. CLVR | Pen -> Target Mug", "configs/scenes/demo14_clvr_pen_mug.json"),
    (9, "demo15_rail_block_plate", "10. RAIL | Yellow Block -> Orange Plate", "configs/scenes/demo15_rail_block_plate.json"),
    (10, "demo16_rail_marker_pot", "11. RAIL | Marker -> Target Pot", "configs/scenes/demo16_rail_marker_pot.json"),
]


def build_all_task_specs() -> List[dict]:
    specs = list(uni.TASK_SPECS)
    for tid, name, label, cfg_rel in NEW_SCENE_CFGS:
        cfg_p = ROOT / cfg_rel
        if not cfg_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text())
        src_o, tgt_o = cfg["objects"][0], cfg["objects"][1]
        s_xyz = list(src_o["size_xyz"])
        t_xyz = list(tgt_o["size_xyz"])
        is_hol = 1.0 if src_o.get("grasp_mode") == "rim_pinch" else 0.0
        pick_z = float(src_o.get("grasp_height_m", 0.5 * s_xyz[2]))
        place_z = float(t_xyz[2] + 0.035)
        specs.append({
            "tid": tid,
            "name": name,
            "label": label,
            "kind": "universal",
            "cfg": cfg_rel,
            "src_body": src_o["name"],
            "tgt_body": tgt_o["name"],
            "src_size": s_xyz,
            "tgt_size": t_xyz,
            "is_hollow": is_hol,
            "pick_z_off": round(pick_z, 4),
            "place_z_off": round(place_z, 4),
        })
    return specs


# ---------------------------------------------------------------------------
# Per-Task Top-P% Elite Trajectory Filtering on Harvested Sim NPZ
# ---------------------------------------------------------------------------
def extract_per_task_elite_sim(sim_path: Path, elite_pct: float = 60.0) -> Tuple[dict, dict]:
    sim = dict(np.load(sim_path))
    N = len(sim["act"])
    tid = sim["tid"]
    p_ee = sim["p_ee"]
    p_tgt = sim["p_tgt"]

    # Detect exact trajectory starts: tid change OR p_ee teleport back to q_home (>8cm)
    starts = [0]
    for i in range(1, N):
        if tid[i] != tid[i - 1] or float(np.linalg.norm(p_ee[i] - p_ee[i - 1])) > 0.08:
            starts.append(i)
    starts.append(N)

    trajs_by_tid: Dict[int, List[Tuple[int, int, int]]] = {}
    for s, e in zip(starts[:-1], starts[1:]):
        if (e - s) < 35:
            continue
        t = int(tid[s])
        trajs_by_tid.setdefault(t, []).append((s, e, e - s))

    keep_idx = []
    stats = {}
    for t in sorted(trajs_by_tid.keys()):
        trs = trajs_by_tid[t]
        lens = np.array([ln for _, _, ln in trs], dtype=np.float32)
        cutoff = float(np.percentile(lens, elite_pct))
        kept = [(s, e, ln) for s, e, ln in trs if ln <= cutoff]
        for s, e, _ in kept:
            keep_idx.append(np.arange(s, e, dtype=np.int64))
        stats[str(t)] = {
            "total_trajs": len(trs),
            "kept_trajs": len(kept),
            "kept_chunks": int(sum(ln for _, _, ln in kept)),
            "len_min": int(lens.min()),
            "len_med": int(np.median(lens)),
            "len_cutoff": round(cutoff, 1),
            "len_max": int(lens.max()),
        }
        print(
            f"  [Elite Filter tid={t}] {len(kept):3d}/{len(trs):3d} trajs kept "
            f"(cutoff<={cutoff:.1f}t | min={int(lens.min())}, med={int(np.median(lens))}, max={int(lens.max())}) "
            f"-> {stats[str(t)]['kept_chunks']} chunks",
            flush=True,
        )

    sel = np.concatenate(keep_idx, axis=0)
    filtered = {k: v[sel] for k, v in sim.items()}
    return filtered, stats


# ---------------------------------------------------------------------------
# Stage 1: Collect All Passing Real Demos Across 11 Tasks x 7 Labs!
# ---------------------------------------------------------------------------
def stage_collect_11tasks(min_demos: int = 8) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Reuse Tasks 0..4 directly from existing verified `uni_real_dataset.npz`
    base_ds = dict(np.load(OUT_DIR / "uni_real_dataset.npz"))
    base_meta = json.loads((OUT_DIR / "task_meta.json").read_text())
    parts = [base_ds]
    task_meta = dict(base_meta)

    all_specs = build_all_task_specs()
    ctx = mp.get_context("spawn")
    n_w = int(os.environ.get("R2S_WORKERS", "16"))

    active_tids = [0, 1, 2, 3, 4]
    for tspec in all_specs[5:]:
        tid = tspec["tid"]
        cfg = json.loads((ROOT / tspec["cfg"]).read_text())
        sid = cfg["scene_id"]
        bundle = Path(cfg["bundle_dir"])
        asset_dir = Path(cfg["asset_dir"])
        res_p = asset_dir / f"{sid}_results.json"
        if not res_p.exists():
            print(f"[Collect Task {tid} {sid}] SKIP (no {res_p.name})", flush=True)
            continue

        res_prev = json.loads(res_p.read_text())
        pass_ids = [e["ep_idx"] for e in res_prev["episodes"] if e["success"]]
        if len(pass_ids) < min_demos:
            print(f"[Collect Task {tid} {sid}] SKIP (only {len(pass_ids)} passing demos < {min_demos})", flush=True)
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
        tasks = []
        for ei in pass_ids:
            ep = meta_by_id[ei]
            ep_m = {f"{cam}_{o['name']}": masks[f"ep{ei:03d}_{cam}_{o['name']}"] for o in cfg["objects"] for cam in ("c1", "c2")}
            tasks.append((tspec, ep, cfg, vis_paths, s1_colors, fallback_intr, ep_m))

        demos = []
        with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
            for f in as_completed([pool.submit(uni._replay_universal_ep, t) for t in tasks]):
                r = f.result()
                if r is not None:
                    demos.append(r)

        if len(demos) < min_demos:
            print(f"[Collect Task {tid} {sid}] SKIP (replayed {len(demos)} < {min_demos})", flush=True)
            continue

        demos.sort(key=lambda x: x["ep_idx"])
        part = uni._demos_to_transitions(demos, tspec)
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
            "name": sid,
            "n_real_demos": len(demos),
            "n_samples": len(part["act"]),
            "q_home": q_home.tolist(),
            "table_z_med": tz_med,
            "xy_bounds": [round(x_lo, 3), round(x_hi, 3), round(y_lo, 3), round(y_hi, 3)],
            "dist_range": [round(min_sep, 3), round(min(0.36, min_sep + 0.20), 3)],
            "ref_ep_idx": int(demos[0]["ep_idx"]),
        }
        print(
            f"[Collect Task {tid:2d} {sid}] Replayed {len(demos)}/{len(pass_ids)} real demos -> "
            f"{len(part['act'])} chunks | tz={tz_med*100:.1f}cm",
            flush=True,
        )

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}
    np.savez_compressed(OUT_DIR / "uni_real_11tasks.npz", **merged)
    (OUT_DIR / "task_meta_11tasks.json").write_text(json.dumps({"active_tids": active_tids, "tasks": task_meta}, indent=2))
    tot_demos = sum(v["n_real_demos"] for k, v in task_meta.items() if int(k) in active_tids)
    print(
        f"[Collect GRAND COMPLETE] {len(active_tids)} active tasks | {tot_demos} verified real passing demos -> "
        f"{len(merged['act'])} real chunks in {time.perf_counter()-t0:.1f}s!",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Scalable QK-Norm TF32 Bridge-Flow DiT Builder (supports 10M or 19M!)
# ---------------------------------------------------------------------------
def build_scalable_uni_dit(d_model: int = 256, n_layers: int = 6, n_heads: int = 8):
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

    class UniGoalBridgeDiT(nn.Module):
        def __init__(self):
            super().__init__()
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
                self.enc_vp(obs[:, 0:6]), self.enc_vb(obs[:, 6:12]),
                self.enc_ls(obs[:, 12:19]), self.enc_lt(obs[:, 19:27]),
                self.enc_ee(obs[:, 27:36]), self.enc_gm(obs[:, 36:43]),
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
# Stage 2: Train v3_5t (Controlled 5-Task) or v3_grand (11-Task x 7-Lab)
# ---------------------------------------------------------------------------
def stage_train_v3(variant: str, elite_pct: float = 60.0, sim_ratio: float = 0.25, ckpt_out: str | None = None) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_specs = build_all_task_specs()
    spec_by_tid = {s["tid"]: s for s in all_specs}

    if variant == "v3_grand":
        real_path = OUT_DIR / "uni_real_11tasks.npz"
        meta_obj = json.loads((OUT_DIR / "task_meta_11tasks.json").read_text())
        active_tids = meta_obj["active_tids"]
        d_model, n_layers, n_heads = 320, 8, 8
        n_steps, lr_max, bs = 8000, 3.0e-4, 528
        ckpt_name = ckpt_out or ("uni_bridge_v3_grand_real.pt" if sim_ratio <= 0.0 else "uni_bridge_v3_grand_mix.pt")
    else:
        real_path = OUT_DIR / "uni_real_dataset.npz"
        active_tids = [0, 1, 2, 3, 4]
        d_model, n_layers, n_heads = 256, 6, 8
        n_steps, lr_max, bs = 6000, 2.8e-4, 500
        ckpt_name = ckpt_out or "uni_bridge_v3_5t.pt"

    real_ds = dict(np.load(real_path))
    n_real = len(real_ds["act"])
    print(f"[{variant} -> {ckpt_name}] Loading real dataset {real_path.name}: {n_real} chunks across tids={active_tids}", flush=True)

    if sim_ratio > 0.0:
        sim_path = OUT_DIR / "uni_sim_harvest_1500_all.npz"
        print(f"[{variant}] Running per-task top-{elite_pct:.0f}% elite filter on {sim_path.name}...", flush=True)
        sim_ds, elite_stats = extract_per_task_elite_sim(sim_path, elite_pct=elite_pct)
        ds = {k: np.concatenate([real_ds[k], sim_ds[k]], axis=0) for k in real_ds.keys()}
        n_sim = len(sim_ds["act"])
        is_sim_flag = np.concatenate([np.zeros(n_real, dtype=bool), np.ones(n_sim, dtype=bool)])
    else:
        ds = real_ds
        n_sim = 0
        elite_stats = {}
        is_sim_flag = np.zeros(n_real, dtype=bool)

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
    tid_t = torch.from_numpy(tid_np).to(device)

    max_tid = max(active_tids) + 1
    pick_z_lut = torch.zeros((max_tid,), device=device, dtype=torch.float32)
    place_z_lut = torch.zeros((max_tid,), device=device, dtype=torch.float32)
    for t in active_tids:
        pick_z_lut[t] = float(spec_by_tid[t]["pick_z_off"])
        place_z_lut[t] = float(spec_by_tid[t]["place_z_off"])

    pools_real = {t: torch.from_numpy(np.where((tid_np == t) & (~is_sim_flag))[0]).to(device) for t in active_tids}
    pools_sim = {t: torch.from_numpy(np.where((tid_np == t) & is_sim_flag)[0]).to(device) for t in active_tids}
    step_ladder = (torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32) * 0.018)[None, :, None]

    def sample_balanced_idx(batch_size: int) -> torch.Tensor:
        per_t = max(8, batch_size // len(active_tids))
        chunks = []
        for t in active_tids:
            pr = pools_real[t]
            ps = pools_sim[t]
            if len(ps) >= 32 and sim_ratio > 0.0:
                ns = max(1, int(round(per_t * sim_ratio)))
                nr = per_t - ns
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

    model = build_scalable_uni_dit(d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    with torch.no_grad():
        _, _, a_stat = make_batch(sample_balanced_idx(6000))
        act_mean = a_stat.mean(dim=(0, 1))
        act_std = a_stat.std(dim=(0, 1)).clamp(min=0.03)

    ema_model = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{variant}] {n_params/1e6:.2f}M params (d={d_model}, L={n_layers}) | {n_steps} steps | "
        f"Real={n_real} + EliteSim={n_sim} (ratio={1-sim_ratio:.0%}:{sim_ratio:.0%})",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-4, fused=True)
    warmup = 250
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
            print(f"  [{variant} step {step:4d}/{n_steps}] loss={loss.item():.4f} | {(time.perf_counter()-t0)/step*1000:.1f} ms/step ({time.perf_counter()-t0:.1f}s)", flush=True)

    train_dt = time.perf_counter() - t0
    with torch.no_grad():
        obs_v, al_v, ae_v = make_batch(sample_balanced_idx(550))
        pred = ema_model.render_chunk(((al_v - act_mean) / act_std).clamp(-6, 6), obs_v, n_steps=ODE_STEPS, seed=42) * act_std + act_mean
        mae_cm = torch.mean(torch.abs(pred[:, :6, :3] - ae_v[:, :6, :3])).item() * 100.0
        g_mae = torch.mean(torch.abs(pred[:, :6, 6] - ae_v[:, :6, 6])).item()
        print(f"[{variant} COMPLETE] {train_dt:.1f}s | 4-step ODE PosMAE={mae_cm:.2f}cm | GripMAE={g_mae:.3f}", flush=True)

    torch.save({
        "ema_state_dict": ema_model.state_dict(),
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "act_mean": act_mean.cpu().numpy(), "act_std": act_std.cpu().numpy(),
        "variant": variant, "active_tids": active_tids,
        "n_real_chunks": n_real, "n_sim_chunks": n_sim,
        "elite_pct": elite_pct, "sim_ratio": sim_ratio, "elite_stats": elite_stats,
        "train_sec": round(train_dt, 1), "val_pos_mae_cm": round(mae_cm, 2), "val_grip_mae": round(g_mae, 3),
    }, OUT_DIR / ckpt_name)


# ---------------------------------------------------------------------------
# Stage 3: Serve Any v3 Checkpoint & Evaluate on Original 125 + New-Task OOD!
# ---------------------------------------------------------------------------
def stage_serve_v3(ckpt_name: str, port: int) -> None:
    import threading
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / ckpt_name, map_location=device, weights_only=False)
    model = build_scalable_uni_dit(
        d_model=int(ck.get("d_model", 256)),
        n_layers=int(ck.get("n_layers", 6)),
        n_heads=int(ck.get("n_heads", 8)),
    ).to(device).eval()
    model.load_state_dict(ck["ema_state_dict"])
    act_mean = torch.from_numpy(ck["act_mean"]).to(device)
    act_std = torch.from_numpy(ck["act_std"]).to(device)

    with torch.no_grad():
        _ = model.render_chunk(torch.zeros((1, HORIZON, ACT_DIM), device=device), torch.zeros((1, uni.OBS_DIM), device=device), n_steps=ODE_STEPS)

    lock = threading.Lock()
    stop_flag = threading.Event()
    req_bytes = 4 + (uni.OBS_DIM + HORIZON * ACT_DIM) * 4

    def handle_client(conn: socket.socket):
        try:
            while not stop_flag.is_set():
                buf = b""
                while len(buf) < req_bytes:
                    c = conn.recv(req_bytes - len(buf))
                    if not c:
                        return
                    buf += c
                seed = struct.unpack("<i", buf[:4])[0]
                if seed == -999999:
                    stop_flag.set()
                    return
                floats = np.frombuffer(buf[4:], dtype=np.float32).copy()
                obs_t = torch.from_numpy(floats[:uni.OBS_DIM]).to(device)[None, :]
                al_t = torch.from_numpy(floats[uni.OBS_DIM:].reshape(HORIZON, ACT_DIM)).to(device)[None, :, :]
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
    srv.bind(("127.0.0.1", port))
    srv.listen(64)
    print(f"[Serve {ckpt_name}] READY on 127.0.0.1:{port} (d={ck.get('d_model')}, L={ck.get('n_layers')})", flush=True)
    try:
        while not stop_flag.is_set():
            try:
                conn, _ = srv.accept()
                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
    finally:
        srv.close()
        print(f"[Serve {ckpt_name}] Shutdown cleanly.", flush=True)


def stage_eval_v3(tag: str, port: int, eval_scope: str = "orig5") -> None:
    uni.PORT = port
    t0 = time.perf_counter()
    n_w = int(os.environ.get("R2S_WORKERS", "20"))

    all_specs = build_all_task_specs()
    spec_by_tid = {s["tid"]: s for s in all_specs}

    if eval_scope == "orig5":
        # EXACT SAME 125 held-out unseen OOD placements (seed=2026) as v1 & v2!
        task_meta = json.loads((OUT_DIR / "task_meta.json").read_text())
        specs = uni._sample_task_ood_specs(task_meta, n_per_task=25, seed=2026)
        eval_tids = [0, 1, 2, 3, 4]
    else:
        meta_obj = json.loads((OUT_DIR / "task_meta_11tasks.json").read_text())
        active_tids = [t for t in meta_obj["active_tids"] if t >= 5]
        eval_tids = active_tids
        rng = np.random.default_rng(2027)
        specs = []
        for t in active_tids:
            ts = spec_by_tid[t]
            tm = meta_obj["tasks"][str(t)]
            x0, x1, y0, y1 = tm["xy_bounds"]
            dmin, dmax = tm["dist_range"]
            tz = float(tm["table_z_med"])
            qh = np.array(tm["q_home"], dtype=np.float32)
            for i in range(25):
                for _ in range(200):
                    sxy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                    txy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                    dist = math.hypot(sxy[0] - txy[0], sxy[1] - txy[1])
                    if dmin <= dist <= dmax:
                        break
                ec = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
                eb = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.010, 0.010)], dtype=np.float32)
                specs.append((ts, i, sxy, txy, tz, qh, ec, eb))

    tasks = [
        ("uni_bridge", ts, i, sxy, txy, tz, qh, ec, eb, 90000 + ts["tid"] * 1000 + i * 17, False, False)
        for ts, i, sxy, txy, tz, qh, ec, eb in specs
    ]

    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
        futs = [pool.submit(uni._run_one_uni_ep, t) for t in tasks]
        for f in as_completed(futs):
            results.append(f.result())

    uni._shutdown_server()

    by_task = {}
    for t in eval_tids:
        ts = spec_by_tid[t]
        rs = [r for r in results if r["tid"] == t]
        ok = sum(int(r["success"]) for r in rs)
        by_task[ts["name"]] = {"pass": ok, "n": len(rs), "rate": round(100.0 * ok / max(1, len(rs)), 1)}

    tot_ok = sum(int(r["success"]) for r in results)
    tot_n = len(results)
    out = {
        "tag": tag, "scope": eval_scope, "pass": tot_ok, "n": tot_n,
        "rate": round(100.0 * tot_ok / max(1, tot_n), 1),
        "per_task": by_task, "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    out_p = OUT_DIR / f"eval_{tag}_{eval_scope}.json"
    out_p.write_text(json.dumps(out, indent=2))
    print("=" * 90, flush=True)
    print(f"[EVAL {tag} ({eval_scope})] TOTAL = {tot_ok}/{tot_n} ({out['rate']:.1f}%) in {out['elapsed_sec']}s", flush=True)
    for k, v in by_task.items():
        print(f"  - {k:28s}: {v['pass']:2d}/{v['n']:2d} ({v['rate']:5.1f}%)", flush=True)
    print("=" * 90, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect_11", "train_v3_5t", "train_v3_grand", "serve", "eval"])
    ap.add_argument("--ckpt", default="uni_bridge_v3_5t.pt")
    ap.add_argument("--port", type=int, default=18958)
    ap.add_argument("--tag", default="v3_5t")
    ap.add_argument("--scope", choices=["orig5", "new6"], default="orig5")
    ap.add_argument("--elite-pct", type=float, default=60.0)
    ap.add_argument("--sim-ratio", type=float, default=0.25)
    args = ap.parse_args()

    if args.cmd == "collect_11":
        stage_collect_11tasks()
    elif args.cmd == "train_v3_5t":
        stage_train_v3("v3_5t", elite_pct=args.elite_pct, sim_ratio=args.sim_ratio)
    elif args.cmd == "train_v3_grand":
        stage_train_v3("v3_grand", elite_pct=args.elite_pct, sim_ratio=args.sim_ratio)
    elif args.cmd == "serve":
        stage_serve_v3(args.ckpt, args.port)
    elif args.cmd == "eval":
        stage_eval_v3(args.tag, args.port, args.scope)


if __name__ == "__main__":
    main()
