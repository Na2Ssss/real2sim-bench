"""High-Throughput 28-Worker Elite Sim-in-the-Loop OOD Trajectory Harvester (1,500 rollouts).

Runs Uni-Goal-Bridge v1 Policy Server on GPU 2 (port 18959) while 28 parallel MuJoCo 4.0.1 EGL
workers sample 300 brand-new unseen full-table OOD placements per task (1,500 rollouts total,
disjoint seed=77777), keeping:
  1. ALL physically passing trajectories (`uni_sim_harvest_1500_all.npz`)
  2. ELITE crisp trajectories (`ticks <= 148`, fast & decisive! -> `uni_sim_harvest_1500_elite.npz`)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import numpy as np

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

import scripts.remote_uni_goal_bridge_scaling as uni

# Use dedicated port 18959 so it never conflicts with any other eval server
uni.PORT = 18959
OUT_DIR = uni.OUT_DIR


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "harvest"
    if mode == "serve":
        uni.stage_serve(round_idx=1)
        return

    t0 = time.perf_counter()
    n_per_task = int(os.environ.get("HARVEST_PER_TASK", "300"))
    n_w = int(os.environ.get("HARVEST_WORKERS", "28"))
    elite_max_ticks = int(os.environ.get("ELITE_MAX_TICKS", "150"))

    task_meta = json.loads((OUT_DIR / "task_meta.json").read_text())
    specs = uni._sample_task_ood_specs(task_meta, n_per_task=n_per_task, seed=77777)
    tasks = [
        ("uni_bridge", ts, i, sxy, txy, tz, qh, ec, eb, 500000 + ts["tid"] * 10000 + i * 29, False, True)
        for ts, i, sxy, txy, tz, qh, ec, eb in specs
    ]

    print(f"[Elite Harvest] Launching {len(tasks)} OOD rollouts ({n_per_task}/task x 5 tasks) on {n_w} EGL workers...", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as pool:
        futs = [pool.submit(uni._run_one_uni_ep, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if k % 250 == 0 or k == len(tasks):
                ok_n = sum(int(r["success"]) for r in results)
                el_n = sum(int(r["success"] and "traj" in r and len(r["traj"]["p_ee"]) <= elite_max_ticks) for r in results)
                print(f"  [Elite Harvest {k:4d}/{len(tasks)}] PASS={ok_n} ({100*ok_n/k:.1f}%) | ELITE(<={elite_max_ticks}t)={el_n} ({time.perf_counter()-t0:.1f}s)", flush=True)

    uni._shutdown_server()

    all_parts, elite_parts = [], []
    all_cnt, elite_cnt = {}, {}
    for ts in uni.TASK_SPECS:
        tid = ts["tid"]
        ok_trajs = [r["traj"] for r in results if r["tid"] == tid and r["success"] and "traj" in r]
        el_trajs = [tr for tr in ok_trajs if len(tr["p_ee"]) <= elite_max_ticks]
        all_cnt[ts["name"]] = len(ok_trajs)
        elite_cnt[ts["name"]] = len(el_trajs)
        if ok_trajs:
            all_parts.append(uni._demos_to_transitions(ok_trajs, ts))
        if el_trajs:
            elite_parts.append(uni._demos_to_transitions(el_trajs, ts))

    n_all_chunks = 0
    if all_parts:
        m_all = {k: np.concatenate([p[k] for p in all_parts], axis=0) for k in all_parts[0].keys()}
        np.savez_compressed(OUT_DIR / "uni_sim_harvest_1500_all.npz", **m_all)
        n_all_chunks = len(m_all["act"])

    n_el_chunks = 0
    if elite_parts:
        m_el = {k: np.concatenate([p[k] for p in elite_parts], axis=0) for k in elite_parts[0].keys()}
        np.savez_compressed(OUT_DIR / "uni_sim_harvest_1500_elite.npz", **m_el)
        n_el_chunks = len(m_el["act"])

    summary = {
        "total_rollouts": len(tasks),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "all_passing_trajs": sum(all_cnt.values()),
        "all_chunks": n_all_chunks,
        "all_per_task": all_cnt,
        "elite_max_ticks": elite_max_ticks,
        "elite_trajs": sum(elite_cnt.values()),
        "elite_chunks": n_el_chunks,
        "elite_per_task": elite_cnt,
    }
    (OUT_DIR / "elite_harvest_1500_summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 94, flush=True)
    print(f"[ELITE SIM HARVEST COMPLETE] {len(tasks)} rollouts in {time.perf_counter()-t0:.1f}s!", flush=True)
    print(f"  ALL PASSING : {sum(all_cnt.values()):4d} trajs ({n_all_chunks:6d} chunks) | {all_cnt}", flush=True)
    print(f"  ELITE CRISP : {sum(elite_cnt.values()):4d} trajs ({n_el_chunks:6d} chunks) | {elite_cnt}", flush=True)
    print("=" * 94, flush=True)


if __name__ == "__main__":
    main()
