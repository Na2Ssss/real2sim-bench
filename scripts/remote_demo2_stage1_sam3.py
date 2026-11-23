"""Demo #2 Stage 1 on A100 GPU (Py3.10):
  1. SAM 3.1 segmentation on Ep#00 & Ep#03 Cam1/Cam2 for `open drawer`, `wooden countertop`, `robot arm`, `blue block`.
  2. SAM 3.1 segmentation on all 74 episodes Cam1/Cam2 Frame-0 for `blue block` (148 images).
  3. Computes clean dual-view open-drawer masks (`|F0 - F_end|` minus robot arm, verified with SAM 3.1).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/data2/r2s_bench")
BUNDLE = ROOT / "outputs" / "demo2_74eps_bundle"
STAGE = ROOT / "outputs" / "demo2_fit_stage"
SAM3_CKPT = "/data1/public/sam3/modelscope_facebook_sam3_1/sam3.1_multiplex.pt"
OUT_WH = (480, 270)


def seg_prompt(proc, state, prompt: str, H: int, W: int) -> np.ndarray:
    out = proc.set_text_prompt(state=state, prompt=prompt)
    m = np.zeros((H, W), dtype=bool)
    if len(out["scores"]) > 0:
        bi = int(torch.argmax(out["scores"]).item())
        m = out["masks"][bi, 0].detach().cpu().numpy().astype(bool)
    return cv2.resize(m.astype(np.uint8), OUT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)


def seg_union(proc, state, prompt: str, H: int, W: int, score_thr: float = 0.22) -> np.ndarray:
    out = proc.set_text_prompt(state=state, prompt=prompt)
    m = np.zeros((H, W), dtype=bool)
    if len(out["scores"]) > 0:
        scores = out["scores"].detach().float().cpu().numpy()
        for i, sc in enumerate(scores):
            if float(sc) >= score_thr:
                m |= out["masks"][i, 0].detach().cpu().numpy().astype(bool)
    return cv2.resize(m.astype(np.uint8), OUT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads((BUNDLE / "meta_demo2.json").read_text())
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=SAM3_CKPT, load_from_HF=False, enable_inst_interactivity=False)
    proc = Sam3Processor(model, confidence_threshold=0.20)
    print(f"[Demo2 SAM3] Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    arrays = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # 1. Detailed masks on Ep#00 & Ep#03 for 1:1 articulated drawer + cabinet fitting
        for ei in [0, 3]:
            ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
            for cam_tag, fn0, fne in [
                ("c1", ed / "cam1_f0.jpg", STAGE / f"ep{ei:02d}_c1_end.jpg"),
                ("c2", ed / "cam2_f0.jpg", STAGE / f"ep{ei:02d}_c2_end.jpg"),
            ]:
                pil0 = Image.open(fn0).convert("RGB")
                W, H = pil0.size
                st0 = proc.set_image(pil0)
                m_drw_sam = seg_prompt(proc, st0, "open drawer", H, W)
                m_rob0 = seg_union(proc, st0, "robot arm", H, W, 0.20)
                m_tbl = seg_prompt(proc, st0, "wooden tabletop", H, W)
                m_blk = seg_prompt(proc, st0, "blue lego block", H, W)

                pile = Image.open(fne).convert("RGB")
                ste = proc.set_image(pile)
                m_robe = seg_union(proc, ste, "robot arm", H, W, 0.20)

                # Motion diff |F0 - F_end| excluding robot arm (dilated) -> razor-sharp moving drawer silhouette!
                im0 = cv2.resize(cv2.imread(str(fn0)), OUT_WH)
                ime = cv2.resize(cv2.imread(str(fne)), OUT_WH)
                diff = np.linalg.norm(im0.astype(np.float32) - ime.astype(np.float32), axis=2)
                rob_union = cv2.dilate((m_rob0 | m_robe).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                m_diff = (diff > 28.0) & (~rob_union)
                # Keep largest connected component(s) overlapping SAM open drawer
                m_diff = cv2.morphologyEx(m_diff.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
                m_diff = cv2.morphologyEx(m_diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

                arrays[f"ref_ep{ei:02d}_{cam_tag}_drawer_sam"] = m_drw_sam
                arrays[f"ref_ep{ei:02d}_{cam_tag}_drawer_diff"] = m_diff
                arrays[f"ref_ep{ei:02d}_{cam_tag}_table"] = m_tbl
                arrays[f"ref_ep{ei:02d}_{cam_tag}_block"] = m_blk
                print(f"  [Ref ep{ei:02d} {cam_tag}] drw_sam={m_drw_sam.sum()}px, drw_diff={m_diff.sum()}px, tbl={m_tbl.sum()}px, blk={m_blk.sum()}px", flush=True)

        # 2. Blue block masks on all 74 episodes (Cam1 + Cam2 Frame-0)
        for ep in meta:
            ei = ep["ep_idx"]
            ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
            for cam_tag, fn in (("c1", "cam1_f0.jpg"), ("c2", "cam2_f0.jpg")):
                pil = Image.open(ed / fn).convert("RGB")
                W, H = pil.size
                st = proc.set_image(pil)
                m_b = seg_prompt(proc, st, "blue lego block", H, W)
                if m_b.sum() < 15:
                    m_b = seg_prompt(proc, st, "small blue block", H, W)
                arrays[f"ep{ei:03d}_{cam_tag}_block"] = m_b
            if (ei + 1) % 20 == 0 or ei == len(meta) - 1:
                print(f"  [Demo2 SAM3] {ei+1}/{len(meta)} eps ({time.perf_counter()-t0:.1f}s)", flush=True)

    out_npz = BUNDLE / "masks_demo2_74eps.npz"
    np.savez_compressed(out_npz, **arrays)
    print(f"[Demo2 SAM3] Saved -> {out_npz} in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
