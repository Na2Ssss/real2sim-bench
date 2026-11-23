"""Demo #4 (TRI 2023-09-21: Remove Marker from Mug, 60 episodes) — Stage 1 on A100 GPU.

Follows `docs/TRANSFER_PLAYBOOK.md`:
  1. Batch SAM 3.1 segmentation across all 60 episodes (120 images) for:
       - `mug` ("mug" / "cup")
       - `marker` ("marker pen" / "orange marker")
       - `robot` ("robot arm" on cluster representative episodes for Hand-Eye check/refine)
  2. Automatic selection of sharpest unoccluded dual-view reference episode ->
     extrinsic gravity de-rolling of Cam1 + Cam2 RGBA cutouts ->
     Multi-View `Hunyuan3D-2mv` generation of `tri_mug_hy3d_mv_raw.glb`.
  3. Saves `/data2/r2s_bench/outputs/demo4_tri_bundle/masks_demo4_60eps.npz` and
     `/data2/r2s_bench/outputs/hy3d_demo4/tri_mug_hy3d_mv_raw.glb` + visual audit strip.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", "/data2/r2s_hy3d/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/data2/r2s_bench")
BUNDLE = ROOT / "outputs" / "demo4_tri_bundle"
ASSET_DIR = ROOT / "outputs" / "hy3d_demo4"
SAM_CKPT = "/data1/public/sam3/modelscope_facebook_sam3_1/sam3.1_multiplex.pt"


def droid_ext6_to_R(ext6: list) -> np.ndarray:
    rx, ry, rz = [float(v) for v in ext6[3:6]]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def gravity_deroll_cutout(img_bgr: np.ndarray, mask: np.ndarray, ext6: list, out_size: int = 512) -> Image.Image:
    """De-roll camera cutout so world +Z points straight UP before Multi-View Hunyuan3D-2mv."""
    R_b_c = droid_ext6_to_R(ext6)
    g_cam = R_b_c.T @ np.array([0.0, 0.0, 1.0])
    # In image coords (u right, v down), world-up projects along (g_cam[0], g_cam[1]) -> want (0, -1)
    ang_rad = math.atan2(g_cam[0], -g_cam[1])
    ang_deg = math.degrees(ang_rad)

    h, w = img_bgr.shape[:2]
    ys, xs = np.where(mask)
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    M = cv2.getRotationMatrix2D((cx, cy), -ang_deg, 1.0)
    rot_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rot_m = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (w, h), flags=cv2.INTER_NEAREST) > 127

    ys2, xs2 = np.where(rot_m)
    y0, y1 = max(0, int(ys2.min()) - 6), min(h, int(ys2.max()) + 6)
    x0, x1 = max(0, int(xs2.min()) - 6), min(w, int(xs2.max()) + 6)
    crop_rgb = cv2.cvtColor(rot_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    crop_a = (rot_m[y0:y1, x0:x1].astype(np.uint8)) * 255

    ch, cw = crop_rgb.shape[:2]
    side = int(max(ch, cw) * 1.25)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    oy = (side - ch) // 2
    ox = (side - cw) // 2
    canvas[oy : oy + ch, ox : ox + cw, :3] = crop_rgb
    canvas[oy : oy + ch, ox : ox + cw, 3] = crop_a
    pil = Image.fromarray(canvas, mode="RGBA").resize((out_size, out_size), Image.LANCZOS)
    return pil


def pick_best_mask(res: dict, min_area: int = 80, max_area_frac: float = 0.25) -> np.ndarray | None:
    masks = res.get("masks")
    scores = res.get("scores")
    if masks is None or len(masks) == 0:
        return None
    m_np = masks.detach().float().cpu().numpy()
    s_np = scores.detach().float().cpu().numpy().reshape(-1)
    if m_np.ndim == 4:
        m_np = m_np[:, 0]
    h, w = m_np.shape[-2:]
    max_area = int(h * w * max_area_frac)
    best_idx, best_sc = -1, -1.0
    for i in range(len(s_np)):
        area = int((m_np[i] > 0.5).sum())
        if min_area <= area <= max_area and float(s_np[i]) > best_sc:
            best_sc = float(s_np[i])
            best_idx = i
    if best_idx < 0:
        return None
    return (m_np[best_idx] > 0.5)


def run_sam3_stage(meta: list) -> dict:
    t0 = time.perf_counter()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=SAM_CKPT, load_from_HF=False, enable_inst_interactivity=False)
    proc = Sam3Processor(model, confidence_threshold=0.22)

    out_masks = {}
    mug_ok_c1 = mug_ok_c2 = mrk_ok_c1 = mrk_ok_c2 = 0
    ref_robot_eps = {0, 10, 20, 30, 45, 55}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ep in meta:
            ei = ep["ep_idx"]
            ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
            for cam, fn in (("c1", "cam1_f0.jpg"), ("c2", "cam2_f0.jpg")):
                pil = Image.open(ed / fn).convert("RGB")
                st = proc.set_image(pil)

                # Mug (try "mug", fallback "cup")
                m_mug = pick_best_mask(proc.set_text_prompt(state=st, prompt="mug"), min_area=400, max_area_frac=0.18)
                if m_mug is None:
                    m_mug = pick_best_mask(proc.set_text_prompt(state=st, prompt="cup"), min_area=400, max_area_frac=0.18)
                if m_mug is not None:
                    out_masks[f"ep{ei:03d}_{cam}_mug"] = m_mug
                    if cam == "c1": mug_ok_c1 += 1
                    else: mug_ok_c2 += 1
                else:
                    out_masks[f"ep{ei:03d}_{cam}_mug"] = np.zeros((pil.height, pil.width), dtype=bool)

                # Marker (try "marker", fallback "orange marker")
                m_mrk = pick_best_mask(proc.set_text_prompt(state=st, prompt="marker"), min_area=30, max_area_frac=0.05)
                if m_mrk is None:
                    m_mrk = pick_best_mask(proc.set_text_prompt(state=st, prompt="orange marker"), min_area=30, max_area_frac=0.05)
                if m_mrk is not None:
                    out_masks[f"ep{ei:03d}_{cam}_marker"] = m_mrk
                    if cam == "c1": mrk_ok_c1 += 1
                    else: mrk_ok_c2 += 1
                else:
                    out_masks[f"ep{ei:03d}_{cam}_marker"] = np.zeros((pil.height, pil.width), dtype=bool)

                if ei in ref_robot_eps:
                    m_rob = pick_best_mask(proc.set_text_prompt(state=st, prompt="robot arm"), min_area=2000, max_area_frac=0.65)
                    if m_rob is not None:
                        out_masks[f"ep{ei:03d}_{cam}_robot"] = m_rob

            if (ei + 1) % 20 == 0 or ei + 1 == len(meta):
                print(
                    f"  [SAM3.1 {ei+1:2d}/{len(meta)}] mug(C1={mug_ok_c1}, C2={mug_ok_c2}) "
                    f"marker(C1={mrk_ok_c1}, C2={mrk_ok_c2}) ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )

    # Save compressed downsampled masks (480x270) for fast MuJoCo eval + full-res best ref masks for Hunyuan3D
    save_dict = {}
    for k, v in out_masks.items():
        save_dict[k] = cv2.resize(v.astype(np.uint8), (480, 270), interpolation=cv2.INTER_NEAREST).astype(bool)
    np.savez_compressed(BUNDLE / "masks_demo4_60eps.npz", **save_dict)

    # Pick sharpest unoccluded reference episode (large mug in BOTH Cam1 and Cam2, away from frame borders)
    best_ei, best_score = 39, -1.0
    for ep in meta:
        ei = ep["ep_idx"]
        m1 = out_masks[f"ep{ei:03d}_c1_mug"]
        m2 = out_masks[f"ep{ei:03d}_c2_mug"]
        a1, a2 = int(m1.sum()), int(m2.sum())
        if a1 < 1500 or a2 < 1500:
            continue
        # Reject masks touching image edges
        h, w = m1.shape
        y1, x1 = np.where(m1)
        y2, x2 = np.where(m2)
        if y1.min() < 20 or y1.max() > h - 20 or y2.min() < 20 or y2.max() > h - 20:
            continue
        sc = min(a1, a2)
        if sc > best_score:
            best_score, best_ei = sc, ei

    print(f"[SAM3.1] Selected best dual-view ref Ep#{best_ei:02d} (min_area={best_score} px)", flush=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    ref_ep = meta[best_ei]
    ed_ref = BUNDLE / "episodes" / f"ep_{best_ei:03d}"
    c1_bgr = cv2.imread(str(ed_ref / "cam1_f0.jpg"))
    c2_bgr = cv2.imread(str(ed_ref / "cam2_f0.jpg"))
    cut1 = gravity_deroll_cutout(c1_bgr, out_masks[f"ep{best_ei:03d}_c1_mug"], ref_ep["c1_extr"])
    cut2 = gravity_deroll_cutout(c2_bgr, out_masks[f"ep{best_ei:03d}_c2_mug"], ref_ep["c2_extr"])
    cut1.save(ASSET_DIR / "ref_mug_c1_rgba.png")
    cut2.save(ASSET_DIR / "ref_mug_c2_rgba.png")
    mug_rgb = (np.median(cv2.cvtColor(c1_bgr, cv2.COLOR_BGR2RGB)[out_masks[f"ep{best_ei:03d}_c1_mug"]], axis=0) / 255.0).tolist()
    (ASSET_DIR / "demo4_colors.json").write_text(json.dumps({"ref_ep_idx": best_ei, "mug_rgb": mug_rgb}, indent=2))

    # Save SAM mask visual audit on 3 spread episodes
    rows = []
    for ei in (best_ei, 19, 59):
        ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
        pair = []
        for cam, fn in (("c1", "cam1_f0.jpg"), ("c2", "cam2_f0.jpg")):
            im = cv2.resize(cv2.imread(str(ed / fn)), (480, 270))
            mm = save_dict[f"ep{ei:03d}_{cam}_mug"]
            mk = save_dict[f"ep{ei:03d}_{cam}_marker"]
            im[mm] = (im[mm] * 0.45 + np.array([60, 230, 80]) * 0.55).astype(np.uint8)
            im[mk] = (im[mk] * 0.30 + np.array([30, 120, 255]) * 0.70).astype(np.uint8)
            cv2.putText(im, f"Ep#{ei:02d} {cam} mug={int(mm.sum())} mrk={int(mk.sum())}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2)
            pair.append(im)
        rows.append(np.hstack(pair))
    cv2.imwrite(str(ASSET_DIR / "demo4_sam_audit.png"), np.vstack(rows))

    del proc, model
    gc.collect()
    torch.cuda.empty_cache()
    return {"ref_ep_idx": best_ei, "sam_sec": round(time.perf_counter() - t0, 1)}


def run_hy3d_mv_stage() -> float:
    t0 = time.perf_counter()
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv",
        variant="fp16",
    )
    cut1 = Image.open(ASSET_DIR / "ref_mug_c1_rgba.png").convert("RGBA")
    cut2 = Image.open(ASSET_DIR / "ref_mug_c2_rgba.png").convert("RGBA")
    images = {"front": cut1, "left": cut2}
    mesh = pipe(
        image=images,
        num_inference_steps=35,
        octree_resolution=320,
        num_chunks=20000,
        generator=torch.manual_seed(42),
        output_type="trimesh",
    )[0]
    out_glb = ASSET_DIR / "tri_mug_hy3d_mv_raw.glb"
    mesh.export(str(out_glb))
    dt = round(time.perf_counter() - t0, 1)
    print(f"[Hunyuan3D-2mv] Exported {out_glb} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces) in {dt}s", flush=True)
    return dt


def main() -> None:
    meta = json.loads((BUNDLE / "meta.json").read_text())
    info = run_sam3_stage(meta)
    info["hy3d_sec"] = run_hy3d_mv_stage()
    (ASSET_DIR / "stage1_info.json").write_text(json.dumps(info, indent=2))
    print("STAGE 1 COMPLETE:", info, flush=True)


if __name__ == "__main__":
    main()
