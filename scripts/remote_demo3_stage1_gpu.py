"""Demo #3 (Pouring Beans, 74 episodes) Stage 1 on A100 GPU (Py3.10):
  1. SAM 3.1 dual-view segmentation on all 74 episodes (148 images).
  2. Gravity-rectified dual-view RGBA cutouts of `red_cup` & `pour_bowl` from Ep#0.
  3. Multi-View `Hunyuan3D-2mv` 3D mesh generation on A100 -> `red_cup_hy3d_mv_raw.glb`, `pour_bowl_hy3d_mv_raw.glb`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

from r2s_bench.adapters.droid import droid_ext6_to_matrix
from r2s_bench.schema import CameraCalib

BUNDLE = ROOT / "outputs" / "demo3_74eps_bundle"
HE_JSON = ROOT / "outputs" / "handeye_demo3" / "refined_extrinsics.json"
WORK_MV = Path("/data2/r2s_hy3d/work_demo3_mv")
SAM3_CKPT = "/data1/public/sam3/modelscope_facebook_sam3_1/sam3.1_multiplex.pt"
OUT_WH = (480, 270)


def make_calib(serial: str, ext6: list, intr_dict: dict) -> CameraCalib:
    fx, cx, fy, cy = intr_dict["cameraMatrix"]
    return CameraCalib(
        name=f"ext_{serial}", serial=serial,
        width=int(intr_dict["width"]), height=int(intr_dict["height"]),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        cam_to_base=droid_ext6_to_matrix(ext6).tolist(),
    )


def save_gravity_rectified_cutout(
    bgr_720: np.ndarray, mask_small: np.ndarray, calib: CameraCalib, out_png: Path
) -> list:
    h, w = bgr_720.shape[:2]
    m = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    m_bin = (m > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(m_bin)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    side = int(max(y1 - y0, x1 - x0) * 1.5)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    sy0, sx0 = max(0, cy - side // 2), max(0, cx - side // 2)
    sy1, sx1 = min(h, sy0 + side), min(w, sx0 + side)

    crop_rgb = cv2.cvtColor(bgr_720[sy0:sy1, sx0:sx1], cv2.COLOR_BGR2RGB)
    crop_a = m_bin[sy0:sy1, sx0:sx1]
    med_rgb = (np.median(crop_rgb[crop_a > 128], axis=0) / 255.0).tolist()

    canvas = np.zeros((512, 512, 4), dtype=np.uint8)
    ch, cw = crop_rgb.shape[:2]
    scale = 420.0 / max(ch, cw)
    nh, nw = int(round(ch * scale)), int(round(cw * scale))
    r_rgb = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    r_a = cv2.resize(crop_a, (nw, nh), interpolation=cv2.INTER_NEAREST)
    oy, ox = (512 - nh) // 2, (512 - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw, :3] = r_rgb
    canvas[oy : oy + nh, ox : ox + nw, 3] = r_a

    # Gravity de-roll using exact camera extrinsics
    T_c_b = np.linalg.inv(np.asarray(calib.cam_to_base, dtype=np.float64))
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]], dtype=np.float64)
    p0 = np.array([0.55, 0.0, 0.0])
    p1 = p0 + np.array([0.0, 0.0, 0.1])
    pc0 = T_c_b[:3, :3] @ p0 + T_c_b[:3, 3]
    pc1 = T_c_b[:3, :3] @ p1 + T_c_b[:3, 3]
    uv0 = (K @ (pc0 / pc0[2]))[:2]
    uv1 = (K @ (pc1 / pc1[2]))[:2]
    d = uv1 - uv0
    ang_deg = float(np.degrees(np.arctan2(d[0], -d[1])))
    M = cv2.getRotationMatrix2D((256.0, 256.0), ang_deg, 1.0)
    rot = cv2.warpAffine(canvas, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    cv2.imwrite(str(out_png), cv2.cvtColor(rot, cv2.COLOR_RGBA2BGRA))
    print(f"  Saved {out_png.name} (de-roll={ang_deg:+.1f} deg, med_rgb={[round(x,3) for x in med_rgb]})", flush=True)
    return med_rgb


def main() -> None:
    meta = json.loads((BUNDLE / "meta_demo3.json").read_text())
    he = json.loads(HE_JSON.read_text())
    WORK_MV.mkdir(parents=True, exist_ok=True)
    masks_npz = BUNDLE / "masks_demo3_74eps.npz"

    # 1. SAM 3.1 on all 74 episodes (148 images)
    if not masks_npz.exists():
        t0 = time.perf_counter()
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(checkpoint_path=SAM3_CKPT, load_from_HF=False, enable_inst_interactivity=False)
        proc = Sam3Processor(model, confidence_threshold=0.20)
        print(f"[Demo3 SAM3] Loaded in {time.perf_counter()-t0:.1f}s", flush=True)
        arrays = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for ep in meta:
                ei = ep["ep_idx"]
                ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
                for cam_tag, fname in (("c1", "cam1_f0.jpg"), ("c2", "cam2_f0.jpg")):
                    pil = Image.open(ed / fname).convert("RGB")
                    W, H = pil.size
                    state = proc.set_image(pil)
                    for obj_key, prompt in (("cup", "red cup"), ("bowl", "bowl")):
                        out = proc.set_text_prompt(state=state, prompt=prompt)
                        m = np.zeros((H, W), dtype=bool)
                        if len(out["scores"]) > 0:
                            bi = int(torch.argmax(out["scores"]).item())
                            m = out["masks"][bi, 0].detach().cpu().numpy().astype(bool)
                        arrays[f"ep{ei:03d}_{cam_tag}_{obj_key}"] = cv2.resize(
                            m.astype(np.uint8), OUT_WH, interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                if (ei + 1) % 20 == 0 or ei == len(meta) - 1:
                    print(f"  [Demo3 SAM3] {ei+1}/{len(meta)} eps ({time.perf_counter()-t0:.1f}s)", flush=True)
        np.savez_compressed(masks_npz, **arrays)
        del model, proc
        torch.cuda.empty_cache()
        print(f"[Demo3 SAM3] Done in {time.perf_counter()-t0:.1f}s -> {masks_npz}", flush=True)
    else:
        print(f"[Demo3 SAM3] Cached -> {masks_npz}", flush=True)

    # 2. Extract gravity-rectified dual-view RGBA cutouts from best clean episode (ep_000)
    masks = dict(np.load(masks_npz))
    ep0 = meta[0]
    for cand in meta[:10]:
        ci = cand["ep_idx"]
        if masks[f"ep{ci:03d}_c1_cup"].sum() > 400 and masks[f"ep{ci:03d}_c2_cup"].sum() > 300 and masks[f"ep{ci:03d}_c1_bowl"].sum() > 500 and masks[f"ep{ci:03d}_c2_bowl"].sum() > 400:
            ep0 = cand
            break
    ei0 = ep0["ep_idx"]
    ed0 = BUNDLE / "episodes" / f"ep_{ei0:03d}"
    f1 = cv2.imread(str(ed0 / "cam1_f0.jpg"))
    f2 = cv2.imread(str(ed0 / "cam2_f0.jpg"))
    c1 = make_calib(ep0["c1_serial"], ep0["c1_extr"], ep0["c1_intr"]); c1.cam_to_base = he["T_base_cam1_refined"]
    c2 = make_calib(ep0["c2_serial"], ep0["c2_extr"], ep0["c2_intr"]); c2.cam_to_base = he["T_base_cam2_refined"]

    cup_rgb = save_gravity_rectified_cutout(f1, masks[f"ep{ei0:03d}_c1_cup"], c1, WORK_MV / "red_cup_cam1_rect.png")
    save_gravity_rectified_cutout(f2, masks[f"ep{ei0:03d}_c2_cup"], c2, WORK_MV / "red_cup_cam2_rect.png")
    bowl_rgb = save_gravity_rectified_cutout(f1, masks[f"ep{ei0:03d}_c1_bowl"], c1, WORK_MV / "pour_bowl_cam1_rect.png")
    save_gravity_rectified_cutout(f2, masks[f"ep{ei0:03d}_c2_bowl"], c2, WORK_MV / "pour_bowl_cam2_rect.png")
    (WORK_MV / "demo3_colors.json").write_text(json.dumps({"ep_ref": ei0, "cup_rgb": cup_rgb, "bowl_rgb": bowl_rgb}, indent=2))

    # 3. Multi-View Hunyuan3D-2mv generation on A100
    cup_glb = WORK_MV / "red_cup_hy3d_mv_raw.glb"
    bowl_glb = WORK_MV / "pour_bowl_hy3d_mv_raw.glb"
    if not (cup_glb.exists() and bowl_glb.exists()):
        t1 = time.perf_counter()
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        print("[Demo3 HY3D-2mv] Loading cached multi-view pipeline on A100...", flush=True)
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", variant="fp16"
        )
        for name, p1, p2, out_g in [
            ("red_cup", WORK_MV / "red_cup_cam1_rect.png", WORK_MV / "red_cup_cam2_rect.png", cup_glb),
            ("pour_bowl", WORK_MV / "pour_bowl_cam1_rect.png", WORK_MV / "pour_bowl_cam2_rect.png", bowl_glb),
        ]:
            ts = time.perf_counter()
            imgs = {"front": Image.open(p1).convert("RGBA"), "left": Image.open(p2).convert("RGBA")}
            mesh = pipe(image=imgs, num_inference_steps=35, octree_resolution=320, num_chunks=20000, generator=torch.manual_seed(42), output_type="trimesh")[0]
            mesh.export(str(out_g))
            print(f"  [Demo3 HY3D-2mv] Generated {name}: faces={len(mesh.faces)} in {time.perf_counter()-ts:.1f}s -> {out_g}", flush=True)
        print(f"[Demo3 HY3D-2mv] Total time: {time.perf_counter()-t1:.1f}s", flush=True)
    else:
        print("[Demo3 HY3D-2mv] Using cached GLBs.", flush=True)


if __name__ == "__main__":
    main()
