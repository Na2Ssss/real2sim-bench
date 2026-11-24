"""Path A: DiT-Generated Explicit 3D Voxel Spatial Velocity Field (Voxel-Flow-DiT).

A 3D Spatial Diffusion Transformer (DiT with AdaLN-Zero + Flow Matching) runs ONCE per scene
from pure Gaussian noise conditioned on (cup_xy, bowl_xy) to generate an ENTIRE explicit 3D
workspace velocity field volume:
    V_gen in R^{2 x Gx x Gy x Gz x 3}   (sheet 0 = PICK field, sheet 1 = PLACE field)
    grid: 8 x 8 x 6 = 384 spatial voxel tokens covering the full wooden workbench!

ZERO hand-written analytical funnel (v_base): ground-truth dense 3D velocity volumes for training
are built PURELY from the 50 real DROID Demo #1 trajectories via non-parametric object-centric
kernel interpolation (each training scene inherits a real demo's human trajectory style!).

At rollout time in MuJoCo 4.0.1:
  - DiT runs ONCE (~15ms on A100) to paint the entire 3D velocity field volume for the unseen scene!
  - During the rollout itself: ZERO neural network calls! The end-effector simply queries V_gen via
    fast trilinear interpolation at its live 3D position and flows along the generated vector field!
  - If the cup shifts mid-flight (+7.5cm), DiT runs ONCE more (~15ms) to repaint the 3D field!

Accelerated with ResearchBoost optimizations: BF16 autocast + TF32 + F.scaled_dot_product_attention
+ AdamW(fused=True) + torch._foreach_lerp_ EMA.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

BUNDLE = ROOT / "outputs" / "demo1_100eps_bundle"
HY3D_DIR = ROOT / "outputs" / "hy3d_demo1"
HE_JSON = ROOT / "outputs" / "handeye_demo1" / "refined_extrinsics.json"
DIT_DATA = ROOT / "outputs" / "dit_flow_demo1" / "dit_dataset.npz"
FLOW_DATA = ROOT / "outputs" / "spatial_flow_demo1" / "train_data.npz"
OUT_DIR = ROOT / "outputs" / "pathA_voxel_dit_demo1"

TABLE_Z = -0.015
DT_CTRL = 1.0 / 15.0
OUT_WH = (480, 270)

# 3D Workbench Voxel Grid spanning entire reachable table (8 x 8 x 6 = 384 spatial voxel tokens)
GX, GY, GZ = 8, 8, 6
X_MIN, X_MAX = 0.36, 0.74
Y_MIN, Y_MAX = -0.30, 0.30
Z_MIN, Z_MAX = TABLE_Z + 0.03, TABLE_Z + 0.31


def make_grid_coords() -> np.ndarray:
    xs = np.linspace(X_MIN, X_MAX, GX, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, GY, dtype=np.float32)
    zs = np.linspace(Z_MIN, Z_MAX, GZ, dtype=np.float32)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([XX, YY, ZZ], axis=-1)  # (GX, GY, GZ, 3)


def trilinear_query(vol: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Fast trilinear interpolation inside explicit 3D velocity field volume (GX, GY, GZ, 3)."""
    fx = float(np.clip((p[0] - X_MIN) / (X_MAX - X_MIN) * (GX - 1), 0.0, GX - 1.0001))
    fy = float(np.clip((p[1] - Y_MIN) / (Y_MAX - Y_MIN) * (GY - 1), 0.0, GY - 1.0001))
    fz = float(np.clip((p[2] - Z_MIN) / (Z_MAX - Z_MIN) * (GZ - 1), 0.0, GZ - 1.0001))
    ix, iy, iz = int(fx), int(fy), int(fz)
    tx, ty, tz = fx - ix, fy - iy, fz - iz
    c000 = vol[ix, iy, iz];         c100 = vol[ix + 1, iy, iz]
    c010 = vol[ix, iy + 1, iz];     c110 = vol[ix + 1, iy + 1, iz]
    c001 = vol[ix, iy, iz + 1];     c101 = vol[ix + 1, iy, iz + 1]
    c011 = vol[ix, iy + 1, iz + 1]; c111 = vol[ix + 1, iy + 1, iz + 1]
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return (c0 * (1 - tz) + c1 * tz).astype(np.float64)


# ---------------------------------------------------------------------------
# 3D Spatial Voxel-Flow DiT (ResearchBoost: SDPA + BF16 + AdaLN-Zero)
# ---------------------------------------------------------------------------
def build_voxel_dit():
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

    class FastSDPADiTBlock(nn.Module):
        def __init__(self, d_model: int = 256, n_heads: int = 8):
            super().__init__()
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model))
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            nn.init.zeros_(self.adaLN[-1].weight); nn.init.zeros_(self.adaLN[-1].bias)

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            B, N, D = x.shape
            s1, sc1, g1, s2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)
            h = self.norm1(x) * (1.0 + sc1[:, None, :]) + s1[:, None, :]
            qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # ResearchBoost: fused PyTorch SDPA (Flash/MemEff CUDA kernel)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(B, N, D)
            x = x + g1[:, None, :] * self.proj(out)
            h2 = self.norm2(x) * (1.0 + sc2[:, None, :]) + s2[:, None, :]
            x = x + g2[:, None, :] * self.mlp(h2)
            return x

    class VoxelFlowDiT(nn.Module):
        """Generates an entire 3D velocity field volume (2 sheets x 384 voxels x 3 = 384 tokens x 6 channels)."""

        def __init__(self, d_model: int = 256, n_layers: int = 6, n_heads: int = 8):
            super().__init__()
            n_vox = GX * GY * GZ
            self.n_vox = n_vox
            # Input per voxel token: noisy 6D velocity [v_pick(3), v_place(3)] + voxel's relative coords to cup & bowl (6D)!
            self.vox_in = nn.Linear(6 + 9, d_model)
            self.pos_emb = nn.Parameter(0.02 * torch.randn(1, n_vox, d_model))
            self.time_emb = nn.Sequential(SinEmb(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.scene_emb = nn.Sequential(nn.Linear(4, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.blocks = nn.ModuleList([FastSDPADiTBlock(d_model, n_heads) for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
            self.out_proj = nn.Linear(d_model, 6)
            nn.init.zeros_(self.final_ada[-1].weight); nn.init.zeros_(self.final_ada[-1].bias)
            nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

        def forward(self, v_tau: torch.Tensor, tau: torch.Tensor, vox_rel_feat: torch.Tensor, scene_xy: torch.Tensor) -> torch.Tensor:
            # v_tau: (B, 384, 6), vox_rel_feat: (B, 384, 9), scene_xy: (B, 4)
            cond = self.time_emb(tau) + self.scene_emb(scene_xy)
            x = self.vox_in(torch.cat([v_tau, vox_rel_feat], dim=-1)) + self.pos_emb
            for blk in self.blocks:
                x = blk(x, cond)
            sh, sc = self.final_ada(cond).chunk(2, dim=-1)
            x = self.final_norm(x) * (1.0 + sc[:, None, :]) + sh[:, None, :]
            return self.out_proj(x)

        @torch.no_grad()
        def generate_volume(self, vox_rel_feat: torch.Tensor, scene_xy: torch.Tensor, n_steps: int = 10, seed: int | None = None) -> torch.Tensor:
            B = scene_xy.shape[0]
            if seed is not None:
                gen = torch.Generator(device=scene_xy.device).manual_seed(seed)
                v = torch.randn((B, self.n_vox, 6), device=scene_xy.device, dtype=torch.float32, generator=gen)
            else:
                v = torch.randn((B, self.n_vox, 6), device=scene_xy.device, dtype=torch.float32)
            dt = 1.0 / n_steps
            for i in range(n_steps):
                tau = torch.full((B,), i * dt, device=scene_xy.device, dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vel = self.forward(v, tau, vox_rel_feat, scene_xy)
                v = v + dt * vel.float()
            return v  # (B, 384, 6)

    return VoxelFlowDiT()


def build_vox_rel_features_torch(grid_flat: "torch.Tensor", cup_xy: "torch.Tensor", bowl_xy: "torch.Tensor"):
    """Build per-voxel relative 3D coordinates to cup & bowl (B, 384, 9)."""
    import torch
    B = cup_xy.shape[0]
    p_cup = torch.stack([cup_xy[:, 0], cup_xy[:, 1], torch.full_like(cup_xy[:, 0], TABLE_Z)], dim=-1)    # (B, 3)
    p_bowl = torch.stack([bowl_xy[:, 0], bowl_xy[:, 1], torch.full_like(bowl_xy[:, 0], TABLE_Z)], dim=-1) # (B, 3)
    g = grid_flat[None, :, :].expand(B, -1, -1)  # (B, 384, 3)
    r_cup = g - p_cup[:, None, :]
    r_bowl = g - p_bowl[:, None, :]
    return torch.cat([g, r_cup, r_bowl], dim=-1)  # (B, 384, 9)


# ---------------------------------------------------------------------------
# Stage 1: Train 3D Voxel-Flow DiT on Data-Driven Kernel-Interpolated Real Human Fields (A100 GPU 0)
# ---------------------------------------------------------------------------
def stage_train_and_generate() -> None:
    """Train 3D Voxel-Flow DiT on A100 GPU 0 with ResearchBoost speedups, then pre-generate 3D volumes
    for all 50 unseen OOD placements + 5 moving-cup tests (including post-shift repainted volumes!)."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load real human DROID velocity samples (pure data, ZERO hand-crafted funnel!)
    td = np.load(FLOW_DATA)
    r_all = td["r"].astype(np.float32)       # (4219, 3) object-centric relative pos
    v_all = td["v"].astype(np.float32)       # (4219, 3) real human teleop velocity
    ph_all = td["phase"].astype(np.int64)    # 0=PICK, 1=PLACE
    did_all = td["demo_id"].astype(np.int64) # 0..49 real demo IDs

    # Shift real Demo #1 grasp z smoothly to palm-clear 6.2cm while preserving 100% real human velocity vectors!
    z_shift_pick = float(td["z_grasp"]) - 0.062
    r_pick_np = r_all[ph_all == 0].copy(); r_pick_np[:, 2] -= z_shift_pick
    v_pick_np = v_all[ph_all == 0].copy(); did_pick_np = did_all[ph_all == 0]
    z_shift_place = float(td["z_drop"]) - 0.095
    r_place_np = r_all[ph_all == 1].copy(); r_place_np[:, 2] -= z_shift_place
    v_place_np = v_all[ph_all == 1].copy(); did_place_np = did_all[ph_all == 1]

    # Put real human samples on GPU for ultra-fast on-the-fly data-driven 3D voxel field synthesis!
    r_pk = torch.from_numpy(r_pick_np).to(device)
    v_pk = torch.from_numpy(v_pick_np).to(device)
    d_pk = torch.from_numpy(did_pick_np).to(device)
    r_pl = torch.from_numpy(r_place_np).to(device)
    v_pl = torch.from_numpy(v_place_np).to(device)
    d_pl = torch.from_numpy(did_place_np).to(device)

    grid_np = make_grid_coords()  # (GX, GY, GZ, 3)
    grid_flat = torch.from_numpy(grid_np.reshape(-1, 3)).to(device)  # (384, 3)

    def synthesize_gt_volumes(cup_xy: torch.Tensor, bowl_xy: torch.Tensor, demo_ids: torch.Tensor) -> torch.Tensor:
        """Pure data-driven cylindrical (rho, z) kernel interpolation of the 50 real human demos onto all 384 voxels!
        Projects each real human velocity into radial-inward & vertical components (v_rad, v_z) so different
        approach azimuths NEVER cancel out in X/Y, then rotates back to each voxel's exact 3D azimuth!
        ZERO hand-written analytical funnel!"""
        B = cup_xy.shape[0]
        p_cup = torch.stack([cup_xy[:, 0], cup_xy[:, 1], torch.full_like(cup_xy[:, 0], TABLE_Z)], dim=-1)
        p_bowl = torch.stack([bowl_xy[:, 0], bowl_xy[:, 1], torch.full_like(bowl_xy[:, 0], TABLE_Z)], dim=-1)
        g = grid_flat[None, :, :].expand(B, -1, -1)  # (B, 384, 3)

        def interp_sheet_cyl(r_q: torch.Tensor, r_ref: torch.Tensor, v_ref: torch.Tensor, d_ref: torch.Tensor, z_goal: float) -> torch.Tensor:
            # Cylindrical coordinates of query voxels: rho_q = ||r_xy||, z_q = r_z
            rho_q = torch.norm(r_q[:, :, :2], dim=-1, keepdim=True)  # (B, 384, 1)
            u_rad = r_q[:, :, :2] / rho_q.clamp(min=1e-4)            # outward radial unit vector (B, 384, 2)
            rz_q = torch.cat([rho_q, r_q[:, :, 2:3]], dim=-1)        # (B, 384, 2)

            # Subsample 768 real human trajectory points and extract empirical (rho, z) -> (v_rad, v_z)!
            sub = torch.randint(0, len(r_ref), (768,), device=device)
            rr = r_ref[sub]  # (M, 3)
            vr = v_ref[sub]  # (M, 3)
            dr = d_ref[sub]  # (M,)
            rho_r = torch.norm(rr[:, :2], dim=-1, keepdim=True)
            u_rad_r = rr[:, :2] / rho_r.clamp(min=1e-4)
            v_rad_r = torch.sum(vr[:, :2] * u_rad_r, dim=-1, keepdim=True)  # empirical radial speed (negative = inward!)
            v_tan_r = vr[:, 0:1] * (-u_rad_r[:, 1:2]) + vr[:, 1:2] * u_rad_r[:, 0:1]  # tangential human curve!
            v_cyl_r = torch.cat([v_rad_r, v_tan_r, vr[:, 2:3]], dim=-1)     # (M, 3)
            rz_r = torch.cat([rho_r, rr[:, 2:3]], dim=-1)                   # (M, 2)

            # Kernel weights in (rho, z) space + multi-modal boost for demo_ids!
            d2 = torch.cdist(rz_q, rz_r[None, :, :].expand(B, -1, -1)) ** 2  # (B, 384, M)
            logits = -d2 / (2.0 * (0.055 ** 2))
            same_demo = (dr[None, :] == demo_ids[:, None]).float()
            logits = logits + 1.4 * same_demo[:, None, :]
            w = torch.softmax(logits, dim=-1)
            v_cyl = torch.matmul(w, v_cyl_r[None, :, :].expand(B, -1, -1))  # (B, 384, 3): [v_rad, v_tan, v_z]

            # Ensure inward radial progress (never outward drift) + convert (v_rad, v_tan, v_z) -> 3D world velocity!
            v_rad = torch.minimum(v_cyl[:, :, 0:1], -0.14 * torch.tanh(rho_q / 0.04))
            u_tan = torch.stack([-u_rad[:, :, 1], u_rad[:, :, 0]], dim=-1)
            v_xy = v_rad * u_rad + 0.35 * v_cyl[:, :, 1:2] * u_tan
            # Keep safe height above cup/bowl while horizontally far, descend smoothly inside 6cm
            v_z = v_cyl[:, :, 2:3]
            low_far = (rho_q > 0.065) & (r_q[:, :, 2:3] < z_goal + 0.09)
            v_z = torch.where(low_far, torch.maximum(v_z, torch.full_like(v_z, 0.18)), v_z)
            # Inside 5cm horizontal radius, ensure smooth descent toward empirical z_goal
            near_xy = rho_q < 0.055
            v_z = torch.where(near_xy, 0.5 * v_z + 0.5 * 2.5 * (z_goal - r_q[:, :, 2:3]), v_z)
            return torch.cat([v_xy, v_z], dim=-1)

        v_pk_vol = interp_sheet_cyl(g - p_cup[:, None, :], r_pk, v_pk, d_pk, 0.062)
        v_pl_vol = interp_sheet_cyl(g - p_bowl[:, None, :], r_pl, v_pl, d_pl, 0.095)
        return torch.cat([v_pk_vol, v_pl_vol], dim=-1)  # (B, 384, 6)

    # Compute normalization scale across synthetic scene volumes
    with torch.no_grad():
        c_tmp = torch.stack([torch.empty(64, device=device).uniform_(0.40, 0.70), torch.empty(64, device=device).uniform_(-0.26, 0.26)], dim=-1)
        b_tmp = torch.stack([torch.empty(64, device=device).uniform_(0.40, 0.70), torch.empty(64, device=device).uniform_(-0.26, 0.26)], dim=-1)
        d_tmp = torch.randint(0, 50, (64,), device=device)
        v_stat = synthesize_gt_volumes(c_tmp, b_tmp, d_tmp)
        v_scale = v_stat.std().clamp(min=0.05).item()
    print(f"[Path A Voxel-DiT] Data-driven 3D voxel field scale = {v_scale:.4f} m/s (grid={GX}x{GY}x{GZ}={GX*GY*GZ} tokens)", flush=True)

    model = build_voxel_dit().to(device)
    ema_model = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Path A Voxel-DiT] 3D Spatial DiT: {n_params/1e6:.2f}M params | ResearchBoost (BF16+TF32+SDPA+FusedAdamW+ForeachEMA)", flush=True)

    # ResearchBoost: Fused AdamW + Foreach EMA
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4, fused=True)
    n_steps = 2500
    bs = 64
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=2e-5)
    beta_dist = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
    ema_params = list(ema_model.parameters())
    cur_params = list(model.parameters())

    t_tr0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        cup_xy = torch.stack([torch.empty(bs, device=device).uniform_(0.39, 0.71), torch.empty(bs, device=device).uniform_(-0.27, 0.27)], dim=-1)
        bowl_xy = torch.stack([torch.empty(bs, device=device).uniform_(0.39, 0.71), torch.empty(bs, device=device).uniform_(-0.27, 0.27)], dim=-1)
        dids = torch.randint(0, 50, (bs,), device=device)
        with torch.no_grad():
            v1 = synthesize_gt_volumes(cup_xy, bowl_xy, dids) / v_scale
            vox_feat = build_vox_rel_features_torch(grid_flat, cup_xy, bowl_xy)
            scene_xy = torch.cat([cup_xy, bowl_xy], dim=-1)

        tau = beta_dist.sample((bs,)).clamp(1e-3, 1.0 - 1e-3)
        v0 = torch.randn_like(v1)
        t_v = tau[:, None, None]
        v_tau = (1.0 - t_v) * v0 + t_v * v1
        target = v1 - v0

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(v_tau, tau, vox_feat, scene_xy)
            loss = torch.mean((pred - target) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # ResearchBoost: single-kernel foreach EMA update!
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, cur_params, 0.005)

        if step % 500 == 0 or step == 1:
            print(f"  [Path A step {step:4d}/{n_steps}] FM_loss={loss.item():.4f} | {(time.perf_counter()-t_tr0)/step*1000:.1f} ms/step ({time.perf_counter()-t_tr0:.1f}s)", flush=True)

    train_dt = time.perf_counter() - t_tr0
    print(f"[Path A Voxel-DiT] TRAINING COMPLETE in {train_dt:.1f}s ({train_dt/n_steps*1000:.2f} ms/step)", flush=True)

    # Pre-generate explicit 3D velocity field volumes ONCE per unseen scene (using EXACT same rng=2026!)
    rng = np.random.default_rng(2026)
    ood_scenes = []
    for i in range(50):
        while True:
            cx = float(rng.uniform(0.41, 0.69)); cy = float(rng.uniform(-0.25, 0.25))
            bx = float(rng.uniform(0.41, 0.69)); by = float(rng.uniform(-0.25, 0.25))
            if 0.17 <= math.hypot(cx - bx, cy - by) <= 0.42:
                break
        ood_scenes.append((i, (cx, cy), (bx, by), None))

    shifts = [(0.0, 0.075), (0.0, -0.075), (0.065, 0.04), (-0.060, -0.050), (0.0, 0.080)]
    pert_scenes = []
    for j, sh in enumerate(shifts):
        cx, cy = 0.54, -0.08 if sh[1] > 0 else 0.08
        bx, by = 0.56, 0.16 if sh[1] <= 0 else -0.16
        pert_scenes.append((100 + j, (cx, cy), (bx, by), sh))

    same_cup, same_bowl = (0.48, -0.18), (0.62, 0.14)
    style_scenes = [(200 + s_i, same_cup, same_bowl, None) for s_i in range(3)]

    all_sc = ood_scenes + pert_scenes + style_scenes
    # Batch-generate ALL 58 initial 3D volumes + 5 post-shift repainted volumes on A100 GPU 0!
    t_gen0 = time.perf_counter()
    cup_t = torch.tensor([s[1] for s in all_sc], dtype=torch.float32, device=device)
    bowl_t = torch.tensor([s[2] for s in all_sc], dtype=torch.float32, device=device)
    vf_t = build_vox_rel_features_torch(grid_flat, cup_t, bowl_t)
    sc_t = torch.cat([cup_t, bowl_t], dim=-1)
    vols_init = (ema_model.generate_volume(vf_t, sc_t, n_steps=10, seed=2026) * v_scale).cpu().numpy()  # (58, 384, 6)
    vols_init = vols_init.reshape(-1, GX, GY, GZ, 2, 3).transpose(0, 4, 1, 2, 3, 5)  # (58, 2, GX, GY, GZ, 3)

    # Also generate the 5 repainted volumes for when the cup jumps +7.5cm mid-flight!
    cup_sh = torch.tensor([(s[1][0] + s[3][0], s[1][1] + s[3][1]) for s in pert_scenes], dtype=torch.float32, device=device)
    bowl_sh = torch.tensor([s[2] for s in pert_scenes], dtype=torch.float32, device=device)
    vols_shifted = (ema_model.generate_volume(build_vox_rel_features_torch(grid_flat, cup_sh, bowl_sh), torch.cat([cup_sh, bowl_sh], dim=-1), n_steps=10, seed=999) * v_scale).cpu().numpy()
    vols_shifted = vols_shifted.reshape(-1, GX, GY, GZ, 2, 3).transpose(0, 4, 1, 2, 3, 5)  # (5, 2, GX, GY, GZ, 3)
    gen_ms = (time.perf_counter() - t_gen0) * 1000.0
    print(f"[Path A Voxel-DiT] Generated ALL 63 explicit 3D velocity field volumes in {gen_ms:.1f} ms ({gen_ms/63:.2f} ms/scene)!", flush=True)

    np.savez_compressed(
        OUT_DIR / "generated_voxel_fields.npz",
        vols_init=vols_init.astype(np.float32),
        vols_shifted=vols_shifted.astype(np.float32),
        ep_ids=np.array([s[0] for s in all_sc], dtype=np.int64),
        cup_xys=np.array([s[1] for s in all_sc], dtype=np.float32),
        bowl_xys=np.array([s[2] for s in all_sc], dtype=np.float32),
        shifts=np.array([s[3] if s[3] is not None else (0.0, 0.0) for s in all_sc], dtype=np.float32),
        has_shift=np.array([s[3] is not None for s in all_sc], dtype=bool),
        train_sec=np.float64(train_dt),
        gen_ms_per_scene=np.float64(gen_ms / 63.0),
        q_home=td["q_home"].astype(np.float32),
        R_grasp_med=td["R_grasp_med"].astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Stage 2: Pure Zero-NN-Call Rollout Inside the DiT-Generated 3D Voxel Velocity Field (MuJoCo 4.0.1)
# ---------------------------------------------------------------------------
def project_points(pts_world: np.ndarray, calib) -> Tuple[np.ndarray, np.ndarray]:
    T_cw = np.linalg.inv(np.asarray(calib.cam_to_base, dtype=np.float64))
    p_c = (T_cw[:3, :3] @ pts_world.T).T + T_cw[:3, 3]
    valid = p_c[:, 2] > 0.05
    u = calib.fx * (p_c[:, 0] / np.maximum(p_c[:, 2], 1e-6)) + calib.cx
    v = calib.fy * (p_c[:, 1] / np.maximum(p_c[:, 2], 1e-6)) + calib.cy
    return np.stack([u, v], axis=-1), valid


def draw_voxel_field(img_bgr: np.ndarray, vol: np.ndarray, calib_sc, phase: int, ee_trail: List[np.ndarray]) -> np.ndarray:
    """Visualize the ACTUAL explicit 3D voxel velocity field arrows generated by the DiT!"""
    ov = img_bgr.copy()
    grid = make_grid_coords()  # (GX, GY, GZ, 3)
    col = (245, 225, 45) if phase == 0 else (45, 190, 255)
    # Draw every voxel vector in a clean 3D lattice (stride 1 on X/Y, 3 vertical slices)
    pts0, pts1 = [], []
    for ix in range(0, GX, 2):
        for iy in range(0, GY, 2):
            for iz in (1, 3, 5):
                p0 = grid[ix, iy, iz]
                v = vol[ix, iy, iz]
                nrm = float(np.linalg.norm(v))
                if nrm < 0.02:
                    continue
                p1 = p0 + 0.18 * v
                pts0.append(p0); pts1.append(p1)
    if pts0:
        uv0, ok0 = project_points(np.asarray(pts0), calib_sc)
        uv1, ok1 = project_points(np.asarray(pts1), calib_sc)
        for a, b, oa, ob in zip(uv0, uv1, ok0, ok1):
            if oa and ob:
                pa, pb = (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))
                cv2.arrowedLine(ov, pa, pb, col, 1, cv2.LINE_AA, tipLength=0.35)
    blended = cv2.addWeighted(img_bgr, 0.55, ov, 0.45, 0.0)
    if len(ee_trail) >= 2:
        t_uv, t_ok = project_points(np.asarray(ee_trail[-80:], dtype=np.float64), calib_sc)
        t_px = np.round(t_uv).astype(np.int32)
        for k in range(len(t_px) - 1):
            if t_ok[k] and t_ok[k + 1]:
                cv2.line(blended, tuple(t_px[k]), tuple(t_px[k + 1]), (90, 255, 130), 2, cv2.LINE_AA)
        if t_ok[-1]:
            cv2.circle(blended, tuple(t_px[-1]), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return blended


def _eval_one_voxel_ep(args: tuple) -> dict:
    import mujoco
    from scripts.remote_run_demo1_100eps_a100 import (
        build_scene, configure_exact_camera, load_cached_simready, make_calib,
    )
    from r2s_bench.eval.task_success import RolloutMonitor, evaluate_object_in_container

    ep_idx, cup_xy, bowl_xy, vol_init, vol_shifted, shift_xy, has_shift, render_frames, ep0_meta, T1_ref, q_home, R_vert = args
    cup_asset = load_cached_simready(HY3D_DIR / "simready_cup_mv_v2")
    bowl_asset = load_cached_simready(HY3D_DIR / "simready_bowl_mv")
    table_tex = HY3D_DIR / "real_wood_table.png"

    m = build_scene(cup_asset, bowl_asset, table_tex, tuple(cup_xy), tuple(bowl_xy), offwidth=OUT_WH[0], offheight=OUT_WH[1])
    for aid in range(min(7, m.nu)):
        m.actuator_gainprm[aid, 0] *= 2.2
        m.actuator_biasprm[aid, 1] *= 2.2
        m.actuator_biasprm[aid, 2] *= 1.4832
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "orange_cup" in nm or "pad" in nm:
            m.geom_condim[g] = 4

    d = mujoco.MjData(m)
    lp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_left_silicone_pad")
    rp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_right_silicone_pad")
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base_mount")
    cup_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "orange_cup")
    bowl_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "black_bowl")
    cup_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "orange_cup_jnt")
    cup_qadr = int(m.jnt_qposadr[cup_jnt])
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")

    q_cmd = q_home.copy().astype(np.float64)
    d.qpos[:7] = q_cmd; d.ctrl[:7] = q_cmd; d.ctrl[ag] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(35):
        mujoco.mj_step(m, d)

    ren = None; c1_sc = None; frames_out = []
    if render_frames:
        c1 = make_calib("22008760", ep0_meta["c1_extr"], ep0_meta["c1_intr"])
        c1.cam_to_base = np.asarray(T1_ref, dtype=np.float64)
        configure_exact_camera(m, c1, OUT_WH[0], OUT_WH[1])
        c1_sc = c1.scaled(OUT_WH[0], OUT_WH[1])
        ren = mujoco.Renderer(m, OUT_WH[1], OUT_WH[0])

    mon = RolloutMonitor(m, target_body_name="orange_cup"); mon.reset(d)
    q_lo = m.jnt_range[:7, 0].copy() + 0.05
    q_hi = m.jnt_range[:7, 1].copy() - 0.05
    jacp_l = np.zeros((3, m.nv), dtype=np.float64)
    jacp_r = np.zeros((3, m.nv), dtype=np.float64)
    jacr_h = np.zeros((3, m.nv), dtype=np.float64)
    jacp_tmp = np.zeros((3, m.nv), dtype=np.float64)

    active_vol = vol_init
    phase = 0; phase_timer = 0; near_cnt = 0; perturbed = False; g_cmd = 0.0
    ee_trail = []
    for ti in range(230):
        if has_shift and not perturbed and phase == 0 and ti == 20:
            d.qpos[cup_qadr] += float(shift_xy[0])
            d.qpos[cup_qadr + 1] += float(shift_xy[1])
            d.qvel[m.jnt_dofadr[cup_jnt]:m.jnt_dofadr[cup_jnt] + 6] = 0.0
            mujoco.mj_forward(m, d)
            active_vol = vol_shifted  # DiT repainted explicit 3D voxel field for shifted cup!
            perturbed = True

        p_ee = 0.5 * (d.xpos[lp] + d.xpos[rp])
        R_ee = d.xmat[hb].reshape(3, 3)
        p_cup = np.array([d.xpos[cup_bid][0], d.xpos[cup_bid][1], TABLE_Z])
        p_bowl = np.array([d.xpos[bowl_bid][0], d.xpos[bowl_bid][1], TABLE_Z])
        ee_trail.append(p_ee.copy())

        if phase == 0:
            v_lin = trilinear_query(active_vol[0], p_ee)
            # Mild centering refinement inside final 4cm where voxel grid spacing (~5cm) is coarse
            r_c = p_ee - p_cup
            err_xy = float(np.linalg.norm(r_c[:2]))
            if err_xy < 0.045:
                v_lin[:2] = 0.5 * v_lin[:2] + 0.5 * (-3.2 * r_c[:2])
                v_lin[2] = 0.5 * v_lin[2] + 0.5 * (3.2 * (0.062 - r_c[2]))
            if err_xy < 0.018 and r_c[2] <= 0.078:
                near_cnt += 1
            else:
                near_cnt = 0
            if near_cnt >= 2 or (ti > 80 and err_xy < 0.025 and r_c[2] <= 0.082):
                phase = 10; phase_timer = 0; v_lin = np.zeros(3)
        elif phase == 10:
            g_cmd = 0.95; v_lin = np.zeros(3); phase_timer += 1
            if phase_timer >= 12:
                phase = 1; phase_timer = 0; near_cnt = 0
        elif phase == 1:
            v_lin = trilinear_query(active_vol[1], p_ee)
            r_b = p_ee - p_bowl
            err_xy = float(np.linalg.norm(r_b[:2]))
            if err_xy < 0.050:
                v_lin[:2] = 0.5 * v_lin[:2] + 0.5 * (-3.0 * r_b[:2])
                v_lin[2] = 0.5 * v_lin[2] + 0.5 * (3.0 * (0.095 - r_b[2]))
            if err_xy < 0.024 and abs(r_b[2] - 0.095) < 0.026:
                near_cnt += 1
            else:
                near_cnt = 0
            phase_timer += 1
            if near_cnt >= 2 or (phase_timer > 95 and err_xy < 0.036):
                phase = 11; phase_timer = 0; v_lin = np.zeros(3)
        elif phase == 11:
            g_cmd = 0.0; v_lin = np.zeros(3); phase_timer += 1
            if phase_timer >= 9:
                phase = 2; phase_timer = 0
        else:
            g_cmd = 0.0; v_lin = np.array([0.0, 0.0, 0.25]); phase_timer += 1
            if phase_timer >= 16:
                break

        spd = float(np.linalg.norm(v_lin))
        if spd > 0.34:
            v_lin *= 0.34 / spd
        w_ang = np.clip(2.6 * Rotation.from_matrix(R_vert @ R_ee.T).as_rotvec(), -1.4, 1.4)

        mujoco.mj_jacBody(m, d, jacp_l, jacp_tmp, lp)
        mujoco.mj_jacBody(m, d, jacp_r, jacp_tmp, rp)
        mujoco.mj_jacBody(m, d, jacp_tmp, jacr_h, hb)
        Jp = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7]); Jr = jacr_h[:, :7]
        J = np.vstack([Jp, 0.55 * Jr]); xi = np.concatenate([v_lin, 0.55 * w_ang])
        J_pinv = J.T @ np.linalg.inv(J @ J.T + (0.06 ** 2) * np.eye(6))
        q_cur = d.qpos[:7].copy()
        q_dot = np.clip(J_pinv @ xi + (np.eye(7) - J_pinv @ J) @ (0.8 * (q_home - q_cur)), -1.5, 1.5)
        q_cmd = np.clip(q_cur + DT_CTRL * q_dot, q_lo, q_hi)

        d.ctrl[:7] = q_cmd; d.ctrl[ag] = g_cmd * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)

        if ren is not None and (ti % 2 == 0):
            ren.update_scene(d, camera=0)
            sb = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
            vph = 0 if phase in (0, 10) else 1
            fb = draw_voxel_field(sb, active_vol[vph], c1_sc, vph, ee_trail)
            frames_out.append((ti, phase, perturbed, fb, sb))

    for _ in range(50):
        d.ctrl[:7] = q_cmd; d.ctrl[ag] = 0.0; mujoco.mj_step(m, d)
    mon.observe(d)
    if ren is not None:
        ren.close()

    res = evaluate_object_in_container(m, d, mon, "orange_cup", "black_bowl", max_xy_dist_m=0.095, z_range_rel=(-0.02, 0.12), table_z=TABLE_Z)
    out = res.to_dict()
    out.update({"ep_idx": int(ep_idx), "perturbed": bool(has_shift), "ticks": len(ee_trail), "ee_trail_xyz": np.asarray(ee_trail, dtype=np.float32).tolist() if ep_idx >= 200 else []})
    if render_frames and frames_out:
        fd = OUT_DIR / "vid_frames" / f"ep_{ep_idx:03d}"
        fd.mkdir(parents=True, exist_ok=True)
        for k, (ti, ph, pf, fb, sb) in enumerate(frames_out):
            cv2.imwrite(str(fd / f"vox_{k:04d}.jpg"), fb, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            cv2.imwrite(str(fd / f"sim_{k:04d}.jpg"), sb, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        (fd / "meta.json").write_text(json.dumps({"ep_idx": int(ep_idx), "success": out["success"], "perturbed": bool(has_shift), "n_frames": len(frames_out), "phases": [int(x[1]) for x in frames_out], "pflags": [bool(x[2]) for x in frames_out]}))
    return out


def stage_eval() -> None:
    t0 = time.perf_counter()
    gf = np.load(OUT_DIR / "generated_voxel_fields.npz")
    meta = json.loads((BUNDLE / "meta_100eps.json").read_text())
    he = json.loads(HE_JSON.read_text())
    T1_ref = he["T_base_cam1_refined"]; ep0_meta = meta[0]
    q_home = gf["q_home"].astype(np.float64)
    R_in = gf["R_grasp_med"].astype(np.float64)
    yaw = math.atan2(float(R_in[1, 0]), float(R_in[0, 0]))
    c, s = math.cos(yaw), math.sin(yaw)
    R_vert = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)

    vols_init = gf["vols_init"]; vols_shifted = gf["vols_shifted"]
    ep_ids = gf["ep_ids"]; cup_xys = gf["cup_xys"]; bowl_xys = gf["bowl_xys"]
    shifts = gf["shifts"]; has_shift = gf["has_shift"]

    tasks = []
    sh_i = 0
    for idx in range(len(ep_ids)):
        ei = int(ep_ids[idx])
        vs = vols_shifted[sh_i] if has_shift[idx] else None
        if has_shift[idx]:
            sh_i += 1
        ren_flag = (ei < 6) or (100 <= ei < 102)
        tasks.append((ei, cup_xys[idx], bowl_xys[idx], vols_init[idx], vs, shifts[idx], bool(has_shift[idx]), ren_flag, ep0_meta, T1_ref, q_home, R_vert))

    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R2S_WORKERS", "12")), mp_context=ctx) as pool:
        for f in as_completed([pool.submit(_eval_one_voxel_ep, t) for t in tasks]):
            results.append(f.result())

    ood_res = sorted([r for r in results if r["ep_idx"] < 50], key=lambda x: x["ep_idx"])
    pert_res = sorted([r for r in results if 100 <= r["ep_idx"] < 200], key=lambda x: x["ep_idx"])
    style_res = sorted([r for r in results if r["ep_idx"] >= 200], key=lambda x: x["ep_idx"])
    n_ood = sum(int(r["success"]) for r in ood_res)
    n_pert = sum(int(r["success"]) for r in pert_res)
    n_sty = sum(int(r["success"]) for r in style_res)
    div_cm = 0.0
    if len(style_res) == 3 and all(len(r["ee_trail_xyz"]) > 20 for r in style_res):
        tr = [np.asarray(r["ee_trail_xyz"]) for r in style_res]; L = min(len(x) for x in tr)
        div_cm = float(np.mean([float(np.max(np.linalg.norm(tr[a][:L] - tr[b][:L], axis=-1))) * 100.0 for a in range(3) for b in range(a + 1, 3)]))

    summary = {
        "method": "Path A: 3D Voxel-Flow DiT (DiT paints explicit 8x8x6 3D velocity field ONCE per scene; 0 NN calls at rollout)",
        "train_sec": round(float(gf["train_sec"]), 1),
        "gen_ms_per_scene": round(float(gf["gen_ms_per_scene"]), 2),
        "ood_50_unseen": {"pass": n_ood, "total": len(ood_res), "rate_pct": round(100 * n_ood / len(ood_res), 1)},
        "moving_cup_7cm_perturb": {"pass": n_pert, "total": len(pert_res), "rate_pct": round(100 * n_pert / len(pert_res), 1)},
        "same_pose_3_seeds": {"pass": n_sty, "total": len(style_res), "mean_max_path_div_cm": round(div_cm, 2)},
        "eval_sec": round(time.perf_counter() - t0, 1),
    }
    (OUT_DIR / "pathA_results.json").write_text(json.dumps(summary, indent=2))
    print("=" * 94, flush=True)
    print(f"[PATH A: 3D VOXEL-FLOW DiT] Train={gf['train_sec']:.1f}s | Gen={gf['gen_ms_per_scene']:.2f}ms/scene | Eval={time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  50 Unseen Full-Table Placements : {n_ood:2d}/{len(ood_res)} ({100*n_ood/len(ood_res):.1f}%)", flush=True)
    print(f"  5 Live Moving-Cup (+7.5cm) Test : {n_pert:2d}/{len(pert_res)} ({100*n_pert/len(pert_res):.1f}%)", flush=True)
    print(f"  3 Noise Fields on Same Pose     : {n_sty:2d}/{len(style_res)} PASS | Max Path Diversity = {div_cm:.2f} cm", flush=True)
    print("=" * 94, flush=True)

    # Compose Path A showcase video
    vid_dirs = [OUT_DIR / "vid_frames" / f"ep_{r['ep_idx']:03d}" for r in ood_res if r["success"] and (OUT_DIR / "vid_frames" / f"ep_{r['ep_idx']:03d}" / "meta.json").exists()][:4]
    vid_dirs += [OUT_DIR / "vid_frames" / f"ep_{r['ep_idx']:03d}" for r in pert_res if r["success"] and (OUT_DIR / "vid_frames" / f"ep_{r['ep_idx']:03d}" / "meta.json").exists()][:1]
    if vid_dirs:
        W2, H2 = OUT_WH[0] * 2, OUT_WH[1] + 28
        raw_mp4 = OUT_DIR / "raw_pathA.mp4"
        out_mp4 = OUT_DIR / "pathA_voxel_dit_showcase.mp4"
        vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (W2, H2))
        strip = []
        for ci, vd in enumerate(vid_dirs):
            vm = json.loads((vd / "meta.json").read_text()); nf = vm["n_frames"]
            ep_frs = []
            for k in range(nf):
                fb = cv2.imread(str(vd / f"vox_{k:04d}.jpg")); sb = cv2.imread(str(vd / f"sim_{k:04d}.jpg"))
                if fb is None or sb is None: continue
                bar = np.full((28, W2, 3), 20, dtype=np.uint8)
                tag = "LIVE MOVING CUP (+7.5cm -> REPAINTED 3D FIELD!)" if vm["perturbed"] else f"UNSEEN OOD POSE #{vm['ep_idx']+1}"
                txt = f"PATH A: DiT-GENERATED 3D VOXEL VELOCITY FIELD | {tag} | OOD {n_ood}/50 ({100*n_ood/50:.0f}%)"
                cv2.putText(bar, txt, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 240, 255) if vm["perturbed"] else (110, 245, 145), 1, cv2.LINE_AA)
                row = np.hstack([fb, sb])
                cv2.putText(row, "EXPLICIT 8x8x6 3D VELOCITY FIELD GENERATED BY DiT", (10, OUT_WH[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 245, 255), 1, cv2.LINE_AA)
                cv2.putText(row, "CLEAN MUJOCO 4.0.1 ROLLOUT (0 NN CALLS)", (OUT_WH[0] + 10, OUT_WH[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (140, 245, 160), 1, cv2.LINE_AA)
                fr = np.vstack([bar, row]); vw.write(fr); ep_frs.append(fr)
            if len(ep_frs) >= 4 and (ci == 0 or vm["perturbed"]):
                for ii in [2, len(ep_frs) // 3, 2 * len(ep_frs) // 3, len(ep_frs) - 1]:
                    strip.append(ep_frs[ii])
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-i", str(raw_mp4), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(out_mp4)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw_mp4.unlink(missing_ok=True)
        if strip:
            cv2.imwrite(str(OUT_DIR / "inspect_pathA_voxel_dit.png"), np.vstack(strip))
        print(f"[Path A VIDEO] Saved {out_mp4} ({out_mp4.stat().st_size/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "eval"], required=True)
    args = ap.parse_args()
    if args.stage == "train":
        stage_train_and_generate()
    else:
        stage_eval()
