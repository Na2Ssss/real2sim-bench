"""Reusable All-on-A100 Parallel DROID Scene Bundle Fetcher (works for ANY new DROID scene manifest).

Downloads `trajectory.h5` and both external stereo camera `.mp4`s directly from public DROID GCS,
extracts `trajectory.npz` (`qpos`, `gripper`), Frame-0 & End keyframes, verifies intrinsics,
and builds a 4-episode visual preview sheet (`real_preview.png`).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

GCS_BASE = "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1"


def download_file(url: str, dst: Path, timeout: int = 30) -> bool:
    if dst.exists() and dst.stat().st_size > 1024:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "real2sim-bench/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 18)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp.stat().st_size > 1024:
                tmp.replace(dst)
                return True
        except Exception:
            if tmp.exists():
                tmp.unlink()
            time.sleep(0.5 * (attempt + 1))
    return False


def extract_f0_and_end(mp4: Path, out_f0: Path, out_end: Path) -> Tuple[bool, int]:
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok0, f0 = cap.read()
    if not ok0 or f0 is None:
        cap.release()
        return False, 0
    cv2.imwrite(str(out_f0), f0, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n - 2, 0))
    ok1, f1 = cap.read()
    if ok1 and f1 is not None:
        cv2.imwrite(str(out_end), f1, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    else:
        cv2.imwrite(str(out_end), f0, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cap.release()
    return True, n


def fetch_one(args: tuple) -> dict | None:
    idx, item, out_dir = args
    ed = out_dir / "episodes" / f"ep_{idx:03d}"
    ed.mkdir(parents=True, exist_ok=True)
    rel = item["gcs_rel"]
    c1s, c2s = item["c1_serial"], item["c2_serial"]

    h5_path = ed / "trajectory.h5"
    c1_mp4 = ed / f"{c1s}.mp4"
    c2_mp4 = ed / f"{c2s}.mp4"

    ok_h5 = download_file(f"{GCS_BASE}/{rel}/trajectory.h5", h5_path)
    ok_c1 = download_file(f"{GCS_BASE}/{rel}/recordings/MP4/{c1s}.mp4", c1_mp4)
    ok_c2 = download_file(f"{GCS_BASE}/{rel}/recordings/MP4/{c2s}.mp4", c2_mp4)
    if not (ok_h5 and ok_c1 and ok_c2):
        return None

    try:
        with h5py.File(h5_path, "r") as f:
            qpos = np.asarray(f["observation/robot_state/joint_positions"][:], dtype=np.float64)
            grip = np.asarray(f["observation/robot_state/gripper_position"][:], dtype=np.float64).reshape(-1)
        np.savez_compressed(ed / "trajectory.npz", qpos=qpos, gripper=grip)
        h5_path.unlink(missing_ok=True)
    except Exception:
        return None

    ok1, n1 = extract_f0_and_end(c1_mp4, ed / "cam1_f0.jpg", ed / "cam1_end.jpg")
    ok2, n2 = extract_f0_and_end(c2_mp4, ed / "cam2_f0.jpg", ed / "cam2_end.jpg")
    if not (ok1 and ok2):
        return None

    # Keep mp4s for top 16 episodes (for showcase video compositing); remove rest to save disk
    if idx >= 16:
        c1_mp4.unlink(missing_ok=True)
        c2_mp4.unlink(missing_ok=True)

    return {
        "ep_idx": idx,
        **item,
        "n_steps": int(len(qpos)),
        "c1_frames": int(n1),
        "c2_frames": int(n2),
    }


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: remote_fetch_droid_scene.py <manifest.json> <out_bundle_dir>")
        sys.exit(1)
    t0 = time.perf_counter()
    manifest = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = int(os.environ.get("FETCH_WORKERS", "12"))

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(fetch_one, (i, item, out_dir)) for i, item in enumerate(manifest)]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is not None:
                results.append(r)
            if k % 15 == 0 or k == len(manifest):
                print(f"  [Fetch {k:2d}/{len(manifest)}] ok={len(results)} ({time.perf_counter()-t0:.1f}s)", flush=True)

    results.sort(key=lambda x: x["ep_idx"])
    # Re-index cleanly if any failed, keeping original ep_dir reference
    (out_dir / "meta.json").write_text(json.dumps(results, indent=2))

    # Build 4x4 visual preview sheet (4 spread-out episodes x [Cam1 F0, Cam1 End, Cam2 F0, Cam2 End])
    picks = [results[i] for i in np.linspace(0, len(results) - 1, min(4, len(results)), dtype=int)]
    rows = []
    for r in picks:
        ed = out_dir / "episodes" / f"ep_{r['ep_idx']:03d}"
        imgs = [cv2.resize(cv2.imread(str(ed / fn)), (360, 202)) for fn in ("cam1_f0.jpg", "cam1_end.jpg", "cam2_f0.jpg", "cam2_end.jpg")]
        row = np.hstack(imgs)
        cv2.putText(row, f"Ep#{r['ep_idx']:02d} ({r['episode_id']}) | {r['instruction'][:60]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 245, 130), 1, cv2.LINE_AA)
        rows.append(row)
    if rows:
        cv2.imwrite(str(out_dir / "real_preview.png"), np.vstack(rows))
    print(f"DONE: fetched {len(results)}/{len(manifest)} episodes in {time.perf_counter()-t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
