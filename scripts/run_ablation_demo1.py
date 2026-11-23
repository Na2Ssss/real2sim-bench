"""Leave-One-Out Ablation Study on Demo #1 (10 real episodes, zero trajectory leakage).

Measures the exact impact of removing each of our 6 reconstruction modules one at a time,
evaluated strictly under:
  1. Fact-Based Task Success (`evaluate_cup_in_bowl`: causal contact + lift>=5cm + seated<=8.5cm + upright cos>=0.5 + settled)
  2. Physical Lift Rate (`lift >= 5cm`) & Seated-in-Bowl Rate (`dist <= 8.5cm`)
  3. Frame-0 Visual Alignment (`F0 Cup IoU`, `F0 Bowl IoU`)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from r2s_bench.calibration.silhouette_refiner import refine_object_xy_two_view
from r2s_bench.eval.task_success import RolloutMonitor, evaluate_cup_in_bowl
from r2s_bench.grading.l1_visual import compute_mask_iou
from r2s_bench.robots.franka_robotiq import build_cup_in_bowl_scene
from run_demo1_cup_in_bowl import configure_exact_camera, ray_plane_intersect_z
from run_demo1_dual_view_triangulation import make_calib

WORK = ROOT / "outputs" / "demo1_10eps_strict"
DATA = ROOT / "data" / "samples" / "demo1_scale25"
CACHE = ROOT / "data" / "hf_meta_cache" / "karlp_droid"
OUT_JSON = ROOT / "outputs" / "ablation_demo1_results.json"
OPT_WH = (320, 180)


def eval_config(
    name: str,
    meta: dict,
    masks: dict,
    sup: dict,
    intr: dict,
    T1_ref,
    T2_ref,
    use_handeye: bool = True,
    use_tz_rim_calib: bool = True,
    use_hollow: bool = True,
    use_contact_phys: bool = True,
    use_sil_refine: bool = True,
    exclude_cam2: bool = True,
) -> Dict[str, object]:
    table_z = -0.040 if use_tz_rim_calib else -0.022
    cup_dims = (0.028, 0.043, 0.094) if use_tz_rim_calib else (0.030, 0.038, 0.090)
    bowl_dims = (0.0775, 0.085, 0.0477) if use_tz_rim_calib else (0.065, 0.085, 0.050)
    cup_mu = 2.2 if use_contact_phys else 1.0
    torsional = use_contact_phys

    fact_pass = 0
    lift_cnt = 0
    seated_cnt = 0
    f0_cup_ious = []
    f0_bowl_ious = []
    reasons: Dict[str, int] = {}

    for ep in meta["episodes"]:
        ei = ep["ep_idx"]
        eid = ep["episode_id"]
        ep_dir = DATA / eid.replace("+", "_")
        tr = np.load(ep_dir / "trajectory.npz")
        qs, gs = tr["qpos"], tr["gripper"]

        c1 = make_calib("22008760", sup[eid]["22008760"], intr[eid]["22008760"])
        c2 = make_calib("24400334", sup[eid]["24400334"], intr[eid]["24400334"])
        if use_handeye:
            c1.extrinsic_matrix = T1_ref
            c2.extrinsic_matrix = T2_ref

        m1_c = cv2.resize(masks[f"ep{ei:02d}_c1_cup"][0].astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
        m2_c = cv2.resize(masks[f"ep{ei:02d}_c2_cup0"].astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
        m1_b = cv2.resize(masks[f"ep{ei:02d}_c1_bowl"][0].astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
        m2_b = cv2.resize(masks[f"ep{ei:02d}_c2_bowl0"].astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)

        ys_c, xs_c = np.nonzero(m1_c)
        ys_b, xs_b = np.nonzero(m1_b)
        sc1 = c1.scaled(*OPT_WH)
        init_c = ray_plane_intersect_z(float(xs_c.mean()), float(ys_c.mean()), sc1, table_z + cup_dims[2] * 0.5)
        init_b = ray_plane_intersect_z(float(xs_b.mean()), float(ys_b.mean()), sc1, table_z + bowl_dims[2] * 0.5)

        if use_sil_refine:
            m_r = build_cup_in_bowl_scene(init_c, init_b, table_z=table_z, hollow=use_hollow, cup_dims=cup_dims, bowl_dims=bowl_dims, offwidth=OPT_WH[0], offheight=OPT_WH[1])
            d_r = mujoco.MjData(m_r); d_r.qpos[:7] = qs[0]; mujoco.mj_forward(m_r, d_r)
            if exclude_cam2:
                cup_xy = refine_object_xy_two_view(m_r, d_r, "orange_cup", init_c, table_z, [c1], [m1_c], max_shift_m=0.08, max_iter=45)[0]
                bowl_xy = refine_object_xy_two_view(m_r, d_r, "black_bowl", init_b, table_z, [c1], [m1_b], max_shift_m=0.08, max_iter=45)[0]
            else:
                cup_xy = refine_object_xy_two_view(m_r, d_r, "orange_cup", init_c, table_z, [c1, c2], [m1_c, m2_c], camera_weights=[0.75, 0.25], max_shift_m=0.08, max_iter=45)[0]
                bowl_xy = refine_object_xy_two_view(m_r, d_r, "black_bowl", init_b, table_z, [c1, c2], [m1_b, m2_b], camera_weights=[0.75, 0.25], max_shift_m=0.08, max_iter=45)[0]
        else:
            cup_xy, bowl_xy = init_c, init_b

        # Build physics scene & compute Frame-0 visual alignment (always measured in true refined Cam1 frame!)
        m = build_cup_in_bowl_scene(
            cup_xy, bowl_xy, table_z=table_z, hollow=use_hollow, cup_friction=cup_mu,
            cup_dims=cup_dims, bowl_dims=bowl_dims, torsional_condim=torsional,
            offwidth=OPT_WH[0], offheight=OPT_WH[1],
        )
        d = mujoco.MjData(m); d.qpos[:7] = qs[0]
        c1_eval = make_calib("22008760", sup[eid]["22008760"], intr[eid]["22008760"])
        c1_eval.extrinsic_matrix = T1_ref
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "eval_cam")
        configure_exact_camera(m, cam_id, c1_eval.scaled(*OPT_WH))
        mujoco.mj_forward(m, d)

        cup_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "orange_cup")
        bowl_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "black_bowl")
        cup_gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == cup_bid]
        bowl_gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bowl_bid]
        act_grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")

        for _ in range(40):
            d.ctrl[:7] = qs[0]; d.ctrl[act_grip] = float(np.clip(gs[0], 0.0, 1.0)) * 255.0
            mujoco.mj_step(m, d)

        ren_seg = mujoco.Renderer(m, height=OPT_WH[1], width=OPT_WH[0])
        ren_seg.enable_segmentation_rendering()
        ren_seg.update_scene(d, camera=cam_id)
        seg = ren_seg.render()[:, :, 0]
        ren_seg.close()
        f0_cup_ious.append(compute_mask_iou(m1_c, np.isin(seg, cup_gids)))
        f0_bowl_ious.append(compute_mask_iou(m1_b, np.isin(seg, bowl_gids)))

        mon = RolloutMonitor(m, target_body_name="orange_cup")
        mon.reset(d)
        for t in range(len(qs)):
            d.ctrl[:7] = qs[t]; d.ctrl[act_grip] = float(np.clip(gs[t], 0.0, 1.0)) * 255.0
            for _ in range(33):
                mujoco.mj_step(m, d)
            mon.observe(d)
        for _ in range(100):
            d.ctrl[:7] = qs[-1]; d.ctrl[act_grip] = float(np.clip(gs[-1], 0.0, 1.0)) * 255.0
            mujoco.mj_step(m, d)
        mon.observe(d)

        res = evaluate_cup_in_bowl(m, d, mon, table_z=table_z)
        fact_pass += int(res.success)
        lift_cnt += int(res.metrics["max_lift_cm"] >= 5.0)
        seated_cnt += int(res.metrics["max_lift_cm"] >= 5.0 and res.metrics["end_xy_dist_cm"] <= 8.5)
        reasons[res.failure_reason] = reasons.get(res.failure_reason, 0) + 1

    n = len(meta["episodes"])
    return {
        "name": name,
        "fact_pass": fact_pass,
        "seated": seated_cnt,
        "lift": lift_cnt,
        "n": n,
        "f0_cup_iou": round(float(np.mean(f0_cup_ious)), 3),
        "f0_bowl_iou": round(float(np.mean(f0_bowl_ious)), 3),
        "reasons": reasons,
    }


def main() -> None:
    t0 = time.time()
    meta = json.loads((WORK / "meta.json").read_text())
    masks = dict(np.load(WORK / "strict10_masks.npz"))
    sup = json.loads((CACHE / "cam2base_extrinsic_superset.json").read_text())
    intr = json.loads((CACHE / "intrinsics.json").read_text())
    he = json.loads((ROOT / "outputs" / "handeye_demo1" / "refined_extrinsics.json").read_text())
    T1_ref, T2_ref = he["T_base_cam1_refined"], he["T_base_cam2_refined"]

    configs = [
        ("Full Pipeline (All Modules)", dict()),
        ("w/o Hollow Thin-Wall Topology", dict(use_hollow=False)),
        ("w/o 6-DoF Hand-Eye Calib", dict(use_handeye=False)),
        ("w/o TableZ & Rim Contact Calib", dict(use_tz_rim_calib=False)),
        ("w/o Silhouette Refine (Ray Only)", dict(use_sil_refine=False)),
        ("w/o Contact Physics (condim4/mu)", dict(use_contact_phys=False)),
        ("w/o Cam2 Exclusion (Biased Cam2)", dict(exclude_cam2=False)),
    ]

    print("=" * 102)
    print(f"{'Ablation Variant':<34} | {'FactPASS':>8} | {'Seated':>6} | {'Lift':>5} | {'F0 Cup':>6} | {'F0 Bwl':>6} | {'ΔPASS':>6}")
    print("-" * 102)
    results = []
    full_pass = None
    for name, kwargs in configs:
        r = eval_config(name, meta, masks, sup, intr, T1_ref, T2_ref, **kwargs)
        if full_pass is None:
            full_pass = r["fact_pass"]
        delta = r["fact_pass"] - full_pass
        r["delta_pass"] = delta
        results.append(r)
        d_str = "—" if name.startswith("Full") else f"{delta:+d}"
        print(
            f"{name:<34} | {r['fact_pass']:2d}/{r['n']:<5} | {r['seated']:2d}/{r['n']:<3} | {r['lift']:2d}/{r['n']:<2} | "
            f"{r['f0_cup_iou']:6.3f} | {r['f0_bowl_iou']:6.3f} | {d_str:>6}"
        )
    print("=" * 102)
    print(f"Completed 7-config x 10-ep ablation in {time.time() - t0:.1f}s")
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
