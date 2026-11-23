"""Demo #6 (CLVR 2023-05-09: Move Mug / Cup on Lab Table, 50 episodes) — Stage 1 on A100 GPU.

1. Batch SAM 3.1 segmentation across 50 episodes (100 images) for `mug` ("mug" / "cup").
2. Per-episode median real mug RGB color extraction (handles yellow, lime-green, and grey cups seamlessly!).
3. Gravity-rectified dual-view `Hunyuan3D-2mv` generation of the CLVR mug (`clvr_mug_hy3d_mv_raw.glb`).
"""

from __future__ import annotations

import gc
import json
import math
import os
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
BUNDLE = ROOT / "outputs" / "demo6_clvr_bundle"
ASSET_DIR = ROOT / "outputs" / "hy3d_demo6"
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
    R_b_c = droid_ext6_to_R(ext6)
    g_cam = R_b_c.T @ np.array([0.0, 0.0, 1.0])
    ang_deg = math.degrees(math.atan2(g_cam[0], -g_cam[1]))
    h, w = img_bgr.shape[:2]
    ys, xs = np.where(mask)
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    M = cv2.getRotationMatrix2D((cx, cy), ang_deg, 1.0)
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
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy : oy + ch, ox : ox + cw, :3] = crop_rgb
    canvas[oy : oy + ch, ox : ox + cw, 3] = crop_a
    return Image.fromarray(canvas, mode="RGBA").resize((out_size, out_size), Image.LANCZOS)


def pick_best(res: dict, min_area: int = 250, max_frac: float = 0.15) -> np.ndarray | None:
    masks, scores = res.get("masks"), res.get("scores")
    if masks is None or len(masks) == 0:
        return None
    m_np = masks.detach().float().cpu().numpy()
    s_np = scores.detach().float().cpu().numpy().reshape(-1)
    if m_np.ndim == 4:
        m_np = m_np[:, 0]
    h, w = m_np.shape[-2:]
    bi, bs = -1, -1.0
    for i in range(len(s_np)):
        a = int((m_np[i] > 0.5).sum())
        if min_area <= a <= int(h * w * max_frac) and float(s_np[i]) > bs:
            bi, bs = i, float(s_np[i])
    return (m_np[bi] > 0.5) if bi >= 0 else None


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads((BUNDLE / "meta.json").read_text())
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=SAM_CKPT, load_from_HF=False, enable_inst_interactivity=False)
    proc = Sam3Processor(model, confidence_threshold=0.22)

    out_masks = {}
    ep_rgbs = {}
    c1_ok = c2_ok = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ep in meta:
            ei = ep["ep_idx"]
            ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
            for cam, fn in (("c1", "cam1_f0.jpg"), ("c2", "cam2_f0.jpg")):
                pil = Image.open(ed / fn).convert("RGB")
                st = proc.set_image(pil)
                mm = pick_best(proc.set_text_prompt(state=st, prompt="mug"), 250, 0.15)
                if mm is None:
                    mm = pick_best(proc.set_text_prompt(state=st, prompt="cup"), 250, 0.15)
                out_masks[f"ep{ei:03d}_{cam}_mug"] = mm if mm is not None else np.zeros((pil.height, pil.width), dtype=bool)
                if cam == "c1":
                    c1_ok += int(mm is not None)
                    if mm is not None and mm.sum() > 100:
                        arr = np.asarray(pil)
                        ep_rgbs[str(ei)] = (np.median(arr[mm], axis=0) / 255.0).tolist()
                else:
                    c2_ok += int(mm is not None)
            if (ei + 1) % 25 == 0 or ei + 1 == len(meta):
                print(f"  [D6 SAM3.1 {ei+1}/{len(meta)}] c1={c1_ok} c2={c2_ok} ({time.perf_counter()-t0:.1f}s)", flush=True)

    save_dict = {k: cv2.resize(v.astype(np.uint8), (480, 270), interpolation=cv2.INTER_NEAREST).astype(bool) for k, v in out_masks.items()}
    np.savez_compressed(BUNDLE / "masks_demo6_50eps.npz", **save_dict)

    best_ei, best_a = 32, -1
    for ep in meta:
        ei = ep["ep_idx"]
        a = min(int(out_masks[f"ep{ei:03d}_c1_mug"].sum()), int(out_masks[f"ep{ei:03d}_c2_mug"].sum()))
        if a > best_a:
            best_a, best_ei = a, ei
    ref_ep = meta[best_ei]
    ed = BUNDLE / "episodes" / f"ep_{best_ei:03d}"
    f1, f2 = cv2.imread(str(ed / "cam1_f0.jpg")), cv2.imread(str(ed / "cam2_f0.jpg"))
    cut1 = gravity_deroll_cutout(f1, out_masks[f"ep{best_ei:03d}_c1_mug"], ref_ep["c1_extr"])
    cut2 = gravity_deroll_cutout(f2, out_masks[f"ep{best_ei:03d}_c2_mug"], ref_ep["c2_extr"])
    cut1.save(ASSET_DIR / "ref_mug_c1_rgba.png"); cut2.save(ASSET_DIR / "ref_mug_c2_rgba.png")
    (ASSET_DIR / "demo6_colors.json").write_text(json.dumps({"ref_ep_idx": best_ei, "ep_rgbs": ep_rgbs}, indent=2))

    del proc, model; gc.collect(); torch.cuda.empty_cache()

    t1 = time.perf_counter()
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", variant="fp16")
    mesh = pipe(image={"front": cut1, "left": cut2}, num_inference_steps=30, octree_resolution=288, num_chunks=20000, generator=torch.manual_seed(42), output_type="trimesh")[0]
    mesh.export(str(ASSET_DIR / "clvr_mug_hy3d_mv_raw.glb"))
    print(f"[D6 STAGE1 DONE] ref=ep{best_ei:02d} sam={t1-t0:.1f}s hy3d={time.perf_counter()-t1:.1f}s faces={len(mesh.faces)}", flush=True)


if __name__ == "__main__":
    main()
