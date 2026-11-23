"""Vertically stack multiple 3-column Real2Sim showcase videos into one synchronized H.264 compilation."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def stitch_vertical(
    items: List[Tuple[str, Path]],
    out_mp4: Path,
    title: str = "REAL2SIM-BENCH (MuJoCo 4.0.1, All-on-A100)",
    row_w: int = 1080,
    row_h: int = 210,
    n_frames: int = 270,
    fps: float = 15.0,
    snap_png: Path | None = None,
) -> Path:
    hdr_h, sep_h = 44, 2
    frame_lists = []
    for label, vp in items:
        cap = cv2.VideoCapture(str(vp))
        frs = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frs.append(cv2.resize(f, (row_w, row_h), interpolation=cv2.INTER_AREA))
        cap.release()
        if not frs:
            raise RuntimeError(f"Empty video: {vp}")
        frame_lists.append((label, frs))

    total_h = hdr_h + len(frame_lists) * row_h + (len(frame_lists) - 1) * sep_h
    if total_h % 2 != 0:
        total_h += 1

    header = np.zeros((hdr_h, row_w, 3), dtype=np.uint8)
    header[:] = (16, 18, 24)
    cv2.putText(header, title, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (110, 245, 160), 2, cv2.LINE_AA)
    sep = np.full((sep_h, row_w, 3), 55, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as td:
        raw_mp4 = Path(td) / "stack_raw.mp4"
        vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (row_w, total_h))
        snaps = []
        for fi in range(n_frames):
            parts = [header]
            for r_idx, (label, frs) in enumerate(frame_lists):
                fr = frs[fi % len(frs)].copy()
                cv2.rectangle(fr, (0, 0), (row_w, 22), (20, 22, 28), -1)
                cv2.putText(fr, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (95, 245, 145), 1, cv2.LINE_AA)
                cv2.putText(fr, "REAL", (row_w - 330, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(fr, "SIM (MuJoCo 4.0.1)", (row_w - 245, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 245, 150), 1, cv2.LINE_AA)
                cv2.putText(fr, "50/50 OVERLAY", (row_w - 108, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 220, 255), 1, cv2.LINE_AA)
                parts.append(fr)
                if r_idx < len(frame_lists) - 1:
                    parts.append(sep)
            canvas = np.vstack(parts)
            if canvas.shape[0] < total_h:
                canvas = np.vstack([canvas, np.zeros((total_h - canvas.shape[0], row_w, 3), dtype=np.uint8)])
            vw.write(canvas)
            if snap_png is not None and fi in (40, 115):
                snaps.append(canvas)
        vw.release()
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_mp4), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", str(out_mp4)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if snap_png is not None and snaps:
            cv2.imwrite(str(snap_png), np.hstack(snaps))
    return out_mp4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output H.264 MP4 path")
    ap.add_argument("--snap", default="", help="Optional snapshot PNG path")
    ap.add_argument("--title", default="REAL2SIM-BENCH (MuJoCo 4.0.1, All-on-A100)", help="Top banner title")
    ap.add_argument("--item", action="append", required=True, help="LABEL::/path/to/video.mp4")
    args = ap.parse_args()
    pairs = []
    for raw in args.item:
        lbl, vp = raw.split("::", 1)
        pairs.append((lbl, Path(vp)))
    stitch_vertical(pairs, Path(args.out), title=args.title, snap_png=Path(args.snap) if args.snap else None)
