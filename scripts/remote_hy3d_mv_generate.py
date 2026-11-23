"""Remote Multi-View Hunyuan3D-2mv shape generation script (runs on A100 GPU).

Uses real dual-view DROID cutouts (`front` = Cam1, `left` = Cam2) with
`tencent/Hunyuan3D-2mv` (`hunyuan3d-dit-v2-mv`, fp16) to reconstruct accurate
3D geometry (eliminating single-view depth/taper ambiguity).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def main() -> None:
    work_dir = Path(sys.argv[1])
    t0 = time.time()
    print("Loading Hunyuan3D-2mv (multi-view) shape pipeline on CUDA...", flush=True)
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv",
        variant="fp16",
    )
    print(f"Loaded Hunyuan3D-2mv in {time.time() - t0:.1f}s", flush=True)

    jobs = [
        (
            "orange_cup",
            {
                "front": work_dir / "orange_cup_cam1_rect_rgba.png",
                "left": work_dir / "orange_cup_cam2_rect_rgba.png",
            },
            work_dir / "orange_cup_hy3d_mv_raw.glb",
        ),
        (
            "black_bowl",
            {
                "front": work_dir / "black_bowl_cam1_rect_rgba.png",
                "left": work_dir / "black_bowl_cam2_rect_rgba.png",
            },
            work_dir / "black_bowl_hy3d_mv_raw.glb",
        ),
    ]

    for name, view_paths, out_glb in jobs:
        ts = time.time()
        images = {k: Image.open(p).convert("RGBA") for k, p in view_paths.items()}
        mesh = pipe(
            image=images,
            num_inference_steps=35,
            octree_resolution=320,
            num_chunks=20000,
            generator=torch.manual_seed(42),
            output_type="trimesh",
        )[0]
        mesh.export(str(out_glb))
        print(
            f"Generated MV {name}: verts={len(mesh.vertices)}, faces={len(mesh.faces)} "
            f"in {time.time() - ts:.1f}s -> {out_glb}",
            flush=True,
        )


if __name__ == "__main__":
    main()
