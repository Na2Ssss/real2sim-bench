"""Universal Config-Driven Stage 1 (A100 GPU): SAM 3.1 Multi-View Segmentation + Multi-View Hunyuan3D-2mv.

Works out-of-the-box for ANY DROID scene described by a JSON config:
  python remote_universal_stage1_gpu.py <scene_cfg.json>

Steps:
  1. Batch SAM 3.1 segmentation across all episodes (Cam1 + Cam2) for every object in `cfg["objects"]`.
  2. Extracts per-episode real median RGB color for every object.
  3. Selects sharpest unoccluded dual-view reference episode -> extrinsic gravity de-rolling of
     Cam1 + Cam2 RGBA cutouts -> Multi-View `Hunyuan3D-2mv` `.glb` mesh generation.
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
    if len(ys2) < 10:
        ys2, xs2, rot_bgr, rot_m = ys, xs, img_bgr, mask
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


def pick_best(res: dict, min_area: int = 120, max_frac: float = 0.25) -> np.ndarray | None:
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
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    sid = cfg["scene_id"]
    bundle = Path(cfg["bundle_dir"])
    asset_dir = Path(cfg["asset_dir"])
    asset_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((bundle / "meta.json").read_text())
    objs = cfg["objects"]

    t0 = time.perf_counter()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=SAM_CKPT, load_from_HF=False, enable_inst_interactivity=False)
    proc = Sam3Processor(model, confidence_threshold=0.20)

    out_masks = {}
    ep_rgbs = {o["name"]: {} for o in objs}
    counts = {o["name"]: [0, 0] for o in objs}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ep in meta:
            ei = ep["ep_idx"]
            ed = bundle / "episodes" / f"ep_{ei:03d}"
            for ci, (cam, fn) in enumerate((("c1", "cam1_f0.jpg"), ("c2", "cam2_f0.jpg"))):
                pil = Image.open(ed / fn).convert("RGB")
                arr = np.asarray(pil)
                st = proc.set_image(pil)
                for o in objs:
                    oname = o["name"]
                    m = None
                    for pr in o["prompts"]:
                        m = pick_best(proc.set_text_prompt(state=st, prompt=pr), o.get("min_area", 150), o.get("max_frac", 0.22))
                        if m is not None:
                            break
                    if m is not None:
                        out_masks[f"ep{ei:03d}_{cam}_{oname}"] = m
                        counts[oname][ci] += 1
                        if cam == "c1" and m.sum() >= 60:
                            ep_rgbs[oname][str(ei)] = (np.median(arr[m], axis=0) / 255.0).tolist()
                    else:
                        out_masks[f"ep{ei:03d}_{cam}_{oname}"] = np.zeros((pil.height, pil.width), dtype=bool)
            if (ei + 1) % 25 == 0 or ei + 1 == len(meta):
                print(f"  [{sid} SAM3.1 {ei+1}/{len(meta)}] counts={counts} ({time.perf_counter()-t0:.1f}s)", flush=True)

    save_dict = {k: cv2.resize(v.astype(np.uint8), (480, 270), interpolation=cv2.INTER_NEAREST).astype(bool) for k, v in out_masks.items()}
    np.savez_compressed(bundle / "masks.npz", **save_dict)

    # Extract gravity-rectified dual-view RGBA cutouts for each object that requests Hunyuan3D-2mv
    cutouts = {}
    ref_eps = {}
    med_rgbs = {}
    for o in objs:
        oname = o["name"]
        vals = list(ep_rgbs[oname].values())
        med_rgbs[oname] = np.median(np.array(vals), axis=0).tolist() if vals else o.get("default_rgb", [0.6, 0.6, 0.6])
        if not o.get("generate_hy3d", False):
            continue
        best_ei, best_a = 0, -1
        for ep in meta:
            ei = ep["ep_idx"]
            m1, m2 = out_masks[f"ep{ei:03d}_c1_{oname}"], out_masks[f"ep{ei:03d}_c2_{oname}"]
            a = min(int(m1.sum()), int(m2.sum()))
            if a > best_a:
                best_a, best_ei = a, ei
        ref_eps[oname] = best_ei
        ref_ep = meta[best_ei]
        ed = bundle / "episodes" / f"ep_{best_ei:03d}"
        m1, m2 = out_masks[f"ep{best_ei:03d}_c1_{oname}"], out_masks[f"ep{best_ei:03d}_c2_{oname}"]
        if m1.sum() >= 100 and m2.sum() >= 100:
            c1_pil = gravity_deroll_cutout(cv2.imread(str(ed / "cam1_f0.jpg")), m1, ref_ep["c1_extr"])
            c2_pil = gravity_deroll_cutout(cv2.imread(str(ed / "cam2_f0.jpg")), m2, ref_ep["c2_extr"])
            c1_pil.save(asset_dir / f"ref_{oname}_c1.png")
            c2_pil.save(asset_dir / f"ref_{oname}_c2.png")
            cutouts[oname] = (c1_pil, c2_pil)

    (asset_dir / "stage1_colors.json").write_text(json.dumps({"ref_eps": ref_eps, "med_rgbs": med_rgbs, "ep_rgbs": ep_rgbs}, indent=2))
    del proc, model; gc.collect(); torch.cuda.empty_cache()

    t1 = time.perf_counter()
    if cutouts:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", variant="fp16")
        for oname, (c1_pil, c2_pil) in cutouts.items():
            ts = time.perf_counter()
            mesh = pipe(image={"front": c1_pil, "left": c2_pil}, num_inference_steps=28, octree_resolution=256, num_chunks=20000, generator=torch.manual_seed(42), output_type="trimesh")[0]
            mesh.export(str(asset_dir / f"{oname}_hy3d_mv_raw.glb"))
            print(f"  [{sid} HY3D-2mv] {oname}: faces={len(mesh.faces)} in {time.perf_counter()-ts:.1f}s", flush=True)
    print(f"[{sid} STAGE1 DONE] sam={t1-t0:.1f}s hy3d={time.perf_counter()-t1:.1f}s counts={counts}", flush=True)


if __name__ == "__main__":
    main()
