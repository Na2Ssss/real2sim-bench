"""All-on-A100 100-Episode Real-to-Sim Benchmark for Demo #1 (Hunyuan3D-2mv + CoACD + Real Wood Table).

Runs completely on the remote A100 server (`/data2/r2s_bench`):
  Stage 1 (A100 GPU): SAM 3.1 segmentation on all 100 episodes' Frame-0 Cam1 & Cam2 (200 images)
  Stage 2 (12 parallel workers + EGL GPU): Dual-View [0.75, 0.25] Silhouette Refinement +
          MuJoCo rollout with Multi-View Hunyuan3D-2mv + CoACD hulls + Real Wood Table +
          Strict Fact-Based `evaluate_cup_in_bowl`.
  Stage 3: Full 100-episode statistical summary + failure attribution + visual audit mosaic.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

from r2s_bench.assets.sim_ready_pipeline import SimReadyAsset
from r2s_bench.calibration.silhouette_refiner import refine_object_xy_two_view
from r2s_bench.eval.task_success import RolloutMonitor, evaluate_cup_in_bowl
from r2s_bench.grading.l1_visual import compute_mask_iou
from r2s_bench.robots.franka_robotiq import build_droid_franka_robotiq_spec
from r2s_bench.schema import CameraCalib
from r2s_bench.adapters.droid import droid_ext6_to_matrix
from r2s_bench.backends.mujoco_backend import opencv_cam_to_mujoco_pose

BUNDLE = ROOT / "outputs" / "demo1_100eps_bundle"
HY3D_DIR = ROOT / "outputs" / "hy3d_demo1"
HE_JSON = ROOT / "outputs" / "handeye_demo1" / "refined_extrinsics.json"
SAM3_CKPT = "/data1/public/sam3/modelscope_facebook_sam3_1/sam3.1_multiplex.pt"

OPT_WH = (320, 180)
OUT_WH = (480, 270)
TABLE_Z = -0.015


def make_calib(serial: str, extr_vec: list, intr_dict: dict) -> CameraCalib:
    fx, cx, fy, cy = intr_dict["cameraMatrix"]
    return CameraCalib(
        name=f"ext_{serial}",
        serial=serial,
        width=int(intr_dict["width"]),
        height=int(intr_dict["height"]),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        cam_to_base=droid_ext6_to_matrix(extr_vec).tolist(),
    )


def configure_exact_camera(m: mujoco.MjModel, calib: CameraCalib, width: int, height: int) -> int:
    cid = 0
    sc = calib.scaled(width, height)
    pos, quat = opencv_cam_to_mujoco_pose(sc.T_base_cam)
    m.cam_pos[cid] = pos
    m.cam_quat[cid] = quat
    px = 1e-5
    ox = -(sc.cx - sc.width / 2.0) * px
    oy = (sc.cy - sc.height / 2.0) * px
    m.cam_resolution[cid] = [sc.width, sc.height]
    m.cam_sensorsize[cid] = [sc.width * px, sc.height * px]
    m.cam_intrinsic[cid] = [sc.fx * px, sc.fy * px, ox, oy]
    return cid


def ray_plane_intersect_z(u: float, v: float, calib: CameraCalib, z_plane: float) -> Tuple[float, float]:
    T = np.asarray(calib.cam_to_base, dtype=np.float64)
    origin = T[:3, 3]
    d_cam = np.array([(u - calib.cx) / calib.fx, (v - calib.cy) / calib.fy, 1.0], dtype=np.float64)
    d_world = T[:3, :3] @ d_cam
    t = (z_plane - origin[2]) / (d_world[2] + 1e-12)
    p = origin + t * d_world
    return float(p[0]), float(p[1])


def load_cached_simready(asset_dir: Path) -> SimReadyAsset:
    info = json.loads((asset_dir / "asset_meta.json").read_text())
    # Rewrite paths relative to ROOT if they came from local Mac
    vis_name = Path(info["visual_mesh_path"]).name
    hull_names = [Path(p).name for p in info["collision_hull_paths"]]
    return SimReadyAsset(
        name=info["name"],
        asset_dir=asset_dir,
        visual_mesh_path=asset_dir / vis_name,
        collision_hull_paths=[asset_dir / h for h in hull_names],
        size_xyz_m=tuple(info["size_xyz_m"]),
        rgba=tuple(info["rgba"]),
        is_container=bool(info["is_container"]),
    )


def build_scene(
    cup_asset: SimReadyAsset,
    bowl_asset: SimReadyAsset,
    table_tex_png: Path,
    cup_xy: Tuple[float, float],
    bowl_xy: Tuple[float, float],
    offwidth: int = 320,
    offheight: int = 180,
) -> mujoco.MjModel:
    spec = build_droid_franka_robotiq_spec(offwidth=offwidth, offheight=offheight)
    spec.add_texture(name="real_wood_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(table_tex_png))
    mat = spec.add_material(name="real_wood_mat", rgba=[1.0, 1.0, 1.0, 1.0], specular=0.15, shininess=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "real_wood_tex"
    spec.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.55, 0.0, TABLE_Z - 0.02],
        size=[0.45, 0.55, 0.02],
        material="real_wood_mat",
        friction=[0.9, 0.005, 0.0001],
    )

    def add_body(asset: SimReadyAsset, body_name: str, xy: Tuple[float, float], mass: float, friction: float):
        vis_mname = f"{body_name}_vis_mesh"
        spec.add_mesh(name=vis_mname, file=str(asset.visual_mesh_path))
        body = spec.worldbody.add_body(name=body_name, pos=[xy[0], xy[1], TABLE_Z + 0.001])
        body.add_freejoint(name=f"{body_name}_jnt")
        body.add_geom(
            name=f"{body_name}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=vis_mname,
            rgba=list(asset.rgba),
            contype=0,
            conaffinity=0,
            mass=0.0,
            group=0,
        )
        part_mass = mass / max(len(asset.collision_hull_paths), 1)
        for idx, hp in enumerate(asset.collision_hull_paths):
            cname = f"{body_name}_col_{idx:02d}"
            spec.add_mesh(name=cname, file=str(hp))
            body.add_geom(
                name=cname,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=cname,
                rgba=[asset.rgba[0], asset.rgba[1], asset.rgba[2], 0.0],
                mass=part_mass,
                friction=[friction, 0.005, 0.0001],
                solref=[0.018, 1.4],
            )

    add_body(cup_asset, "orange_cup", cup_xy, mass=0.045, friction=1.0)
    add_body(bowl_asset, "black_bowl", bowl_xy, mass=0.250, friction=1.2)
    return spec.compile()


def stage1_sam3_masks(meta: List[dict], masks_npz: Path) -> float:
    if masks_npz.exists():
        print(f"[Stage 1] Using cached SAM 3.1 masks -> {masks_npz}", flush=True)
        return 0.0
    t0 = time.perf_counter()
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=SAM3_CKPT, load_from_HF=False, enable_inst_interactivity=False)
    proc = Sam3Processor(model, confidence_threshold=0.22)
    print(f"[Stage 1] Loaded SAM 3.1 on A100 in {time.perf_counter() - t0:.1f}s", flush=True)

    arrays = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ep in meta:
            ei = ep["ep_idx"]
            ed = BUNDLE / "episodes" / f"ep_{ei:03d}"
            for cam_tag, fname in (("c1", "cam1_f0.jpg"), ("c2", "cam2_f0.jpg")):
                pil = Image.open(ed / fname).convert("RGB")
                W, H = pil.size
                state = proc.set_image(pil)
                for obj_key, prompt in (("cup", "orange cup"), ("bowl", "black bowl")):
                    out = proc.set_text_prompt(state=state, prompt=prompt)
                    m = np.zeros((H, W), dtype=bool)
                    if len(out["scores"]) > 0:
                        bi = int(torch.argmax(out["scores"]).item())
                        m = out["masks"][bi, 0].detach().cpu().numpy().astype(bool)
                    m_rs = cv2.resize(m.astype(np.uint8), OUT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
                    arrays[f"ep{ei:03d}_{cam_tag}_{obj_key}"] = m_rs
            if (ei + 1) % 20 == 0 or ei == len(meta) - 1:
                print(f"  [SAM3] Segmented {ei + 1}/{len(meta)} episodes ({time.perf_counter() - t0:.1f}s)", flush=True)

    np.savez_compressed(masks_npz, **arrays)
    dt = time.perf_counter() - t0
    print(f"[Stage 1] Saved 100-ep dual-view masks -> {masks_npz} in {dt:.1f}s", flush=True)
    return dt


def eval_single_episode(args: tuple) -> dict:
    ep, T1_ref, T2_ref, m1_c, m1_b, m2_c, m2_b, render_snap = args
    ei = ep["ep_idx"]
    eid = ep["episode_id"]
    ed = BUNDLE / "episodes" / f"ep_{ei:03d}"

    # Validate mask presence
    if m1_c.sum() < 15 or m1_b.sum() < 25:
        return {
            "ep_idx": ei,
            "episode_id": eid,
            "f0_cup_iou": 0.0,
            "f0_bowl_iou": 0.0,
            "success": False,
            "failure_reason": "MASK_MISSING",
            "metrics": {"max_lift_cm": 0.0, "end_xy_dist_cm": 99.0, "cup_upright_cos": 0.0, "pre_contact_disp_cm": 0.0},
        }

    cup_asset = load_cached_simready(HY3D_DIR / "simready_cup_mv_v2")
    bowl_asset = load_cached_simready(HY3D_DIR / "simready_bowl_mv")
    table_tex = HY3D_DIR / "real_wood_table.png"

    c1 = make_calib("22008760", ep["c1_extr"], ep["c1_intr"])
    c2 = make_calib("24400334", ep["c2_extr"], ep["c2_intr"])
    c1.cam_to_base = np.asarray(T1_ref, dtype=np.float64)
    c2.cam_to_base = np.asarray(T2_ref, dtype=np.float64)

    sc1 = c1.scaled(OUT_WH[0], OUT_WH[1])
    ys_c, xs_c = np.nonzero(m1_c)
    ys_b, xs_b = np.nonzero(m1_b)
    init_c = ray_plane_intersect_z(float(xs_c.mean()), float(ys_c.mean()), sc1, TABLE_Z + 0.047)
    init_b = ray_plane_intersect_z(float(xs_b.mean()), float(ys_b.mean()), sc1, TABLE_Z + 0.024)

    traj = np.load(ed / "trajectory.npz")
    qs = traj["qpos"]
    gs = traj["gripper"]

    # Step 3: Dual-View [0.75, 0.25] Silhouette Pose Refinement (zero trajectory leakage)
    m_r = build_scene(cup_asset, bowl_asset, table_tex, init_c, init_b, offwidth=OUT_WH[0], offheight=OUT_WH[1])
    d_r = mujoco.MjData(m_r)
    d_r.qpos[:7] = qs[0]
    mujoco.mj_forward(m_r, d_r)

    use_c2_cup = m2_c.sum() >= 15
    use_c2_bowl = m2_b.sum() >= 25
    if use_c2_cup:
        cup_xy = refine_object_xy_two_view(
            m_r, d_r, "orange_cup", init_c, TABLE_Z, [c1, c2], [m1_c, m2_c],
            camera_weights=[0.75, 0.25], max_shift_m=0.08, max_iter=40,
        )[0]
    else:
        cup_xy = refine_object_xy_two_view(
            m_r, d_r, "orange_cup", init_c, TABLE_Z, [c1], [m1_c],
            max_shift_m=0.08, max_iter=40,
        )[0]

    if use_c2_bowl:
        bowl_xy = refine_object_xy_two_view(
            m_r, d_r, "black_bowl", init_b, TABLE_Z, [c1, c2], [m1_b, m2_b],
            camera_weights=[0.75, 0.25], max_shift_m=0.08, max_iter=40,
        )[0]
    else:
        bowl_xy = refine_object_xy_two_view(
            m_r, d_r, "black_bowl", init_b, TABLE_Z, [c1], [m1_b],
            max_shift_m=0.08, max_iter=40,
        )[0]

    # Step 4: Full MuJoCo Rollout with Strict Fact-Based Monitor
    m = build_scene(cup_asset, bowl_asset, table_tex, cup_xy, bowl_xy, offwidth=OUT_WH[0], offheight=OUT_WH[1])
    d = mujoco.MjData(m)
    cid1 = configure_exact_camera(m, c1, OUT_WH[0], OUT_WH[1])
    act_grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")
    cup_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "orange_cup")
    bowl_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "black_bowl")
    cup_gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == cup_bid]
    bowl_gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bowl_bid]

    d.qpos[:7] = qs[0]
    d.ctrl[:7] = qs[0]
    d.ctrl[act_grip] = float(np.clip(gs[0], 0.0, 1.0)) * 255.0
    mujoco.mj_forward(m, d)
    for _ in range(30):
        mujoco.mj_step(m, d)

    # Measure Frame-0 Cup & Bowl IoU
    ren_seg = mujoco.Renderer(m, height=OUT_WH[1], width=OUT_WH[0])
    ren_seg.enable_segmentation_rendering()
    ren_seg.update_scene(d, camera=cid1)
    seg = ren_seg.render()[:, :, 0]
    f0_c_iou = compute_mask_iou(m1_c, np.isin(seg, cup_gids))
    f0_b_iou = compute_mask_iou(m1_b, np.isin(seg, bowl_gids))
    ren_seg.close()

    mon = RolloutMonitor(m, target_body_name="orange_cup")
    mon.reset(d)

    for t_idx in range(len(qs)):
        d.ctrl[:7] = qs[t_idx]
        d.ctrl[act_grip] = float(np.clip(gs[t_idx], 0.0, 1.0)) * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)
    for _ in range(110):
        d.ctrl[:7] = qs[-1]
        d.ctrl[act_grip] = float(np.clip(gs[-1], 0.0, 1.0)) * 255.0
        mujoco.mj_step(m, d)
    mon.observe(d)

    res = evaluate_cup_in_bowl(m, d, mon, table_z=TABLE_Z)
    snap_path = None
    if render_snap:
        ren_rgb = mujoco.Renderer(m, height=OUT_WH[1], width=OUT_WH[0])
        ren_rgb.update_scene(d, camera=cid1)
        sim_end = cv2.cvtColor(ren_rgb.render(), cv2.COLOR_RGB2BGR)
        ren_rgb.close()
        snap_dir = BUNDLE / "snaps"
        snap_dir.mkdir(exist_ok=True)
        snap_path = str(snap_dir / f"ep_{ei:03d}_end.jpg")
        cv2.imwrite(snap_path, sim_end)

    return {
        "ep_idx": ei,
        "episode_id": eid,
        "f0_cup_iou": round(float(f0_c_iou), 3),
        "f0_bowl_iou": round(float(f0_b_iou), 3),
        "snap_path": snap_path,
        **res.to_dict(),
    }


def main() -> None:
    t_all = time.perf_counter()
    meta = json.loads((BUNDLE / "meta_100eps.json").read_text())
    he = json.loads(HE_JSON.read_text())
    T1_ref, T2_ref = he["T_base_cam1_refined"], he["T_base_cam2_refined"]

    masks_npz = BUNDLE / "masks_100eps.npz"
    t_sam3 = stage1_sam3_masks(meta, masks_npz)

    t_sim0 = time.perf_counter()
    masks = dict(np.load(masks_npz))
    n_workers = int(os.environ.get("R2S_WORKERS", "12"))
    print(f"[Stage 2] Launching 100-episode parallel Sim-Ready evaluation ({n_workers} workers, EGL GPU)...", flush=True)

    tasks = []
    for ep in meta:
        ei = ep["ep_idx"]
        tasks.append((
            ep,
            T1_ref,
            T2_ref,
            masks[f"ep{ei:03d}_c1_cup"],
            masks[f"ep{ei:03d}_c1_bowl"],
            masks[f"ep{ei:03d}_c2_cup"],
            masks[f"ep{ei:03d}_c2_bowl"],
            ei < 20,  # save end snapshots for first 20 episodes for visual mosaic
        ))

    results: List[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(eval_single_episode, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if k % 10 == 0 or k == len(tasks):
                cur_pass = sum(int(x["success"]) for x in results)
                cur_lift = sum(int(x["metrics"]["max_lift_cm"] >= 5.0) for x in results)
                cur_seat = sum(int(x["metrics"]["max_lift_cm"] >= 5.0 and x["metrics"]["end_xy_dist_cm"] <= 8.5) for x in results)
                print(
                    f"  [Progress {k:3d}/{len(tasks)}] PASS={cur_pass}/{k} ({100*cur_pass/k:.1f}%) | "
                    f"Seated={cur_seat}/{k} ({100*cur_seat/k:.1f}%) | Lift={cur_lift}/{k} ({100*cur_lift/k:.1f}%) | "
                    f"elapsed={time.perf_counter()-t_sim0:.1f}s",
                    flush=True,
                )

    results.sort(key=lambda x: x["ep_idx"])
    t_sim = time.perf_counter() - t_sim0
    t_tot = time.perf_counter() - t_all

    n = len(results)
    n_pass = sum(int(x["success"]) for x in results)
    n_lift = sum(int(x["metrics"]["max_lift_cm"] >= 5.0) for x in results)
    n_seat = sum(int(x["metrics"]["max_lift_cm"] >= 5.0 and x["metrics"]["end_xy_dist_cm"] <= 8.5) for x in results)
    n_tipped_in_bowl = sum(
        int((not x["success"]) and x["metrics"]["max_lift_cm"] >= 5.0 and x["metrics"]["end_xy_dist_cm"] <= 8.5)
        for x in results
    )
    n_missed_grasp = sum(int(x["metrics"]["max_lift_cm"] < 5.0) for x in results)
    reasons = Counter(x["failure_reason"] for x in results if not x["success"])
    f0_cups = [x["f0_cup_iou"] for x in results if x["f0_cup_iou"] > 0]
    f0_bowls = [x["f0_bowl_iou"] for x in results if x["f0_bowl_iou"] > 0]

    summary = {
        "n_episodes": n,
        "pipeline": "Hunyuan3D-2mv (Multi-View) + CoACD (24 hulls) + Real Wood Table + Dual-View Silhouette Refine",
        "timing_sec": {
            "stage1_sam3_200imgs": round(t_sam3, 1),
            "stage2_100eps_parallel_sim": round(t_sim, 1),
            "total_wall_clock": round(t_tot, 1),
            "sec_per_episode_effective": round(t_tot / max(n, 1), 2),
        },
        "metrics": {
            "fact_pass": n_pass,
            "fact_pass_rate_pct": round(100.0 * n_pass / n, 1),
            "seated_in_bowl": n_seat,
            "seated_rate_pct": round(100.0 * n_seat / n, 1),
            "lifted_ge_5cm": n_lift,
            "lift_rate_pct": round(100.0 * n_lift / n, 1),
            "tipped_inside_bowl": n_tipped_in_bowl,
            "missed_grasp_lt_5cm": n_missed_grasp,
            "mean_f0_cup_iou": round(float(np.mean(f0_cups)), 3) if f0_cups else 0.0,
            "mean_f0_bowl_iou": round(float(np.mean(f0_bowls)), 3) if f0_bowls else 0.0,
        },
        "failure_reasons": dict(reasons),
        "episodes": results,
    }
    out_json = BUNDLE / "demo1_100eps_hy3d_mv_results.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Build a compact 2x5 visual audit mosaic (5 PASS + 5 FAIL from first 20 snaps)
    snapped_pass = [x for x in results if x.get("snap_path") and x["success"]][:5]
    snapped_fail = [x for x in results if x.get("snap_path") and not x["success"]][:5]
    rows = []
    for grp_name, grp, col in [("PASS", snapped_pass, (60, 230, 100)), ("FAIL", snapped_fail, (80, 110, 255))]:
        if not grp:
            continue
        tiles = []
        for x in grp:
            im = cv2.imread(x["snap_path"])
            if im is None:
                continue
            im = cv2.resize(im, (320, 180))
            cv2.rectangle(im, (0, 0), (320, 22), (15, 18, 22), -1)
            txt = f"Ep#{x['ep_idx']:02d} {grp_name} | L={x['metrics']['max_lift_cm']:.0f}cm D={x['metrics']['end_xy_dist_cm']:.1f}cm c={x['metrics']['cup_upright_cos']:.2f}"
            cv2.putText(im, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            tiles.append(im)
        while len(tiles) < 5:
            tiles.append(np.zeros((180, 320, 3), np.uint8))
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(BUNDLE / "demo1_100eps_audit_mosaic.png"), np.vstack(rows))

    print("=" * 95, flush=True)
    print(f"100-EPISODE A100 BENCHMARK COMPLETE in {t_tot:.1f}s (SAM3={t_sam3:.1f}s, Sim100={t_sim:.1f}s, {t_tot/n:.2f}s/ep)", flush=True)
    print(f"  Strict FACT PASS : {n_pass:3d}/{n} ({100*n_pass/n:.1f}%)", flush=True)
    print(f"  Seated in Bowl   : {n_seat:3d}/{n} ({100*n_seat/n:.1f}%)", flush=True)
    print(f"  Lifted >= 5 cm   : {n_lift:3d}/{n} ({100*n_lift/n:.1f}%)", flush=True)
    print(f"  Tipped in Bowl   : {n_tipped_in_bowl:3d}/{n} | Missed Grasp (<5cm): {n_missed_grasp:3d}/{n}", flush=True)
    print(f"  Mean F0 IoU      : Cup={summary['metrics']['mean_f0_cup_iou']:.3f}, Bowl={summary['metrics']['mean_f0_bowl_iou']:.3f}", flush=True)
    print(f"  Failure Reasons  : {dict(reasons)}", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
