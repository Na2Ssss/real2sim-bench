"""Demo #3 (Pouring Beans, 74 episodes) Stage 2 on A100 + MuJoCo 4.0.1 (Py3.12 venv_mj4):
  1. Extract real tabletop Homography texture from Ep#0 Cam1.
  2. Build Sim-Ready Open-Top Hollow + CoACD assets from `Hunyuan3D-2mv` GLBs (`red_cup`, `pour_bowl`).
  3. Run 16-worker parallel Dual-View [0.75, 0.25] Silhouette Pose Refinement + 10-bean pouring physics
     rollout on MuJoCo 4.0.1 + strict Fact-Based `evaluate_pouring_beans` across ALL 74 episodes!
  4. Export JSON report + 2x5 visual audit mosaic.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("R2S_MENAGERIE_DIR", "/data2/r2s_bench/menagerie")

import cv2
import mujoco
import numpy as np

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

from r2s_bench.adapters.droid import droid_ext6_to_matrix
from r2s_bench.assets.sim_ready_pipeline import SimReadyAsset, build_sim_ready_asset, extract_real_table_texture
from r2s_bench.backends.mujoco_backend import opencv_cam_to_mujoco_pose
from r2s_bench.calibration.silhouette_refiner import refine_object_xy_two_view
from r2s_bench.eval.task_success import RolloutMonitor, evaluate_pouring_beans
from r2s_bench.grading.l1_visual import compute_mask_iou
from r2s_bench.robots.franka_robotiq import build_droid_franka_robotiq_spec
from r2s_bench.schema import CameraCalib

BUNDLE = ROOT / "outputs" / "demo3_74eps_bundle"
HE_JSON = ROOT / "outputs" / "handeye_demo3" / "refined_extrinsics.json"
WORK_MV = Path("/data2/r2s_hy3d/work_demo3_mv")
ASSET_DIR = ROOT / "outputs" / "hy3d_demo3"
OUT_WH = (480, 270)
TABLE_Z = 0.165


def make_calib(serial: str, ext6: list, intr_dict: dict) -> CameraCalib:
    fx, cx, fy, cy = intr_dict["cameraMatrix"]
    return CameraCalib(
        name=f"ext_{serial}", serial=serial,
        width=int(intr_dict["width"]), height=int(intr_dict["height"]),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        cam_to_base=droid_ext6_to_matrix(ext6).tolist(),
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
    return SimReadyAsset(
        name=info["name"], asset_dir=asset_dir,
        visual_mesh_path=asset_dir / Path(info["visual_mesh_path"]).name,
        collision_hull_paths=[asset_dir / Path(p).name for p in info["collision_hull_paths"]],
        size_xyz_m=tuple(info["size_xyz_m"]),
        rgba=tuple(info["rgba"]),
        is_container=bool(info["is_container"]),
    )


def triangulate_2rays(c1: CameraCalib, u1: float, v1: float, c2: CameraCalib, u2: float, v2: float) -> np.ndarray:
    def ray(c: CameraCalib, u: float, v: float):
        T = np.asarray(c.cam_to_base, dtype=np.float64)
        o = T[:3, 3]
        d = T[:3, :3] @ np.array([(u - c.cx) / c.fx, (v - c.cy) / c.fy, 1.0], dtype=np.float64)
        return o, d / (np.linalg.norm(d) + 1e-12)
    o1, d1 = ray(c1, u1, v1)
    o2, d2 = ray(c2, u2, v2)
    w0 = o1 - o2
    a, b, c_ = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
    d_, e_ = float(d1 @ w0), float(d2 @ w0)
    denom = a * c_ - b * b + 1e-12
    s = (b * e_ - c_ * d_) / denom
    t = (a * e_ - b * d_) / denom
    return 0.5 * ((o1 + s * d1) + (o2 + t * d2))


def build_demo3_scene(
    cup_asset: SimReadyAsset,
    bowl_asset: SimReadyAsset,
    table_tex_png: Path,
    cup_xy: Tuple[float, float],
    bowl_xy: Tuple[float, float],
    table_z: float = TABLE_Z,
    n_beans: int = 10,
    offwidth: int = 480,
    offheight: int = 270,
) -> mujoco.MjModel:
    spec = build_droid_franka_robotiq_spec(offwidth=offwidth, offheight=offheight)
    spec.add_texture(name="real_table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(table_tex_png))
    mat = spec.add_material(name="real_table_mat", rgba=[1.0, 1.0, 1.0, 1.0], specular=0.12, shininess=0.15)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "real_table_tex"
    spec.worldbody.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.55, 0.0, table_z - 0.02], size=[0.50, 0.55, 0.02],
        material="real_table_mat", friction=[0.9, 0.005, 0.0001],
    )

    def add_body(asset: SimReadyAsset, body_name: str, xy: Tuple[float, float], mass: float, friction: float):
        vis_mname = f"{body_name}_vis_mesh"
        spec.add_mesh(name=vis_mname, file=str(asset.visual_mesh_path))
        body = spec.worldbody.add_body(name=body_name, pos=[xy[0], xy[1], table_z + 0.001])
        body.add_freejoint(name=f"{body_name}_jnt")
        body.add_geom(
            name=f"{body_name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=vis_mname,
            rgba=list(asset.rgba), contype=0, conaffinity=0, mass=0.0, group=0,
        )
        part_mass = mass / max(len(asset.collision_hull_paths), 1)
        for idx, hp in enumerate(asset.collision_hull_paths):
            cname = f"{body_name}_col_{idx:02d}"
            spec.add_mesh(name=cname, file=str(hp))
            body.add_geom(
                name=cname, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=cname,
                rgba=[asset.rgba[0], asset.rgba[1], asset.rgba[2], 0.0],
                mass=part_mass, friction=[friction, 0.005, 0.0001], solref=[0.018, 1.3],
            )

    add_body(cup_asset, "red_cup", cup_xy, mass=0.055, friction=1.2)
    add_body(bowl_asset, "pour_bowl", bowl_xy, mass=0.350, friction=1.2)

    # Add low-bounce pouring beans neatly stacked inside the hollow cup
    for bi in range(n_beans):
        layer = bi // 4
        slot = bi % 4
        ang = slot * (np.pi / 2.0) + layer * 0.4
        rr = 0.011 if slot > 0 or layer > 0 else 0.0
        bx = cup_xy[0] + rr * np.cos(ang)
        by = cup_xy[1] + rr * np.sin(ang)
        bz = table_z + 0.016 + layer * 0.015
        bb = spec.worldbody.add_body(name=f"bean_{bi:02d}", pos=[bx, by, bz])
        bb.add_freejoint(name=f"bean_{bi:02d}_jnt")
        bb.add_geom(
            name=f"bean_{bi:02d}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.007, 0, 0], rgba=[0.88, 0.72, 0.25, 1.0],
            mass=0.004, friction=[0.8, 0.005, 0.0001], solref=[0.020, 1.5],
        )

    return spec.compile()


def eval_demo3_ep(args: tuple) -> dict:
    ep, T1_ref, T2_ref, m1_c, m1_b, m2_c, m2_b, render_snap = args
    ei = ep["ep_idx"]
    eid = ep["episode_id"]
    ed = BUNDLE / "episodes" / f"ep_{ei:03d}"

    if m1_c.sum() < 20 or m1_b.sum() < 25:
        return {
            "ep_idx": ei, "episode_id": eid, "f0_cup_iou": 0.0, "f0_bowl_iou": 0.0,
            "success": False, "failure_reason": "MASK_MISSING",
            "metrics": {"max_lift_cm": 0.0, "beans_in_bowl": 0, "beans_total": 10},
        }

    cup_asset = load_cached_simready(ASSET_DIR / "simready_red_cup")
    bowl_asset = load_cached_simready(ASSET_DIR / "simready_pour_bowl")
    table_tex = ASSET_DIR / "demo3_real_table.png"

    c1 = make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"]); c1.cam_to_base = T1_ref
    c2 = make_calib(ep["c2_serial"], ep["c2_extr"], ep["c2_intr"])
    # Only apply ep0's refined Cam2 extrinsic when Cam2 tripod has not moved (>3cm)
    t2_raw = np.asarray(ep["c2_extr"][:3], dtype=np.float64)
    t2_ref_pos = np.asarray(T2_ref, dtype=np.float64)[:3, 3]
    if float(np.linalg.norm(t2_raw - t2_ref_pos)) < 0.04:
        c2.cam_to_base = T2_ref
    sc1, sc2 = c1.scaled(*OUT_WH), c2.scaled(*OUT_WH)

    ys_c, xs_c = np.nonzero(m1_c); ys_b, xs_b = np.nonzero(m1_b)
    # Pure Frame-0 Dual-View 3D Triangulation of table_z (handles motorized height-adjustable desks with zero trajectory leakage!)
    tz_ep = TABLE_Z
    if m2_c.sum() >= 20:
        y2c, x2c = np.nonzero(m2_c)
        p3_cup = triangulate_2rays(sc1, float(xs_c.mean()), float(ys_c.mean()), sc2, float(x2c.mean()), float(y2c.mean()))
        if 0.02 <= p3_cup[2] <= 0.35:
            tz_ep = float(p3_cup[2] - 0.072)
    elif m2_b.sum() >= 25:
        y2b, x2b = np.nonzero(m2_b)
        p3_bwl = triangulate_2rays(sc1, float(xs_b.mean()), float(ys_b.mean()), sc2, float(x2b.mean()), float(y2b.mean()))
        if 0.02 <= p3_bwl[2] <= 0.35:
            tz_ep = float(p3_bwl[2] - 0.045)

    init_c = ray_plane_intersect_z(float(xs_c.mean()), float(ys_c.mean()), sc1, tz_ep + 0.052)
    init_b = ray_plane_intersect_z(float(xs_b.mean()), float(ys_b.mean()), sc1, tz_ep + 0.028)

    tr = np.load(ed / "trajectory.npz")
    qs, gs = tr["qpos"], tr["gripper"]

    # Dual-View [0.75, 0.25] 2D Silhouette Pose Refinement (zero trajectory leakage)
    m_r = build_demo3_scene(cup_asset, bowl_asset, table_tex, init_c, init_b, table_z=tz_ep, n_beans=0, offwidth=OUT_WH[0], offheight=OUT_WH[1])
    d_r = mujoco.MjData(m_r); d_r.qpos[:7] = qs[0]; mujoco.mj_forward(m_r, d_r)
    c_cams, c_ms, c_ws = ([c1, c2], [m1_c, m2_c], [0.75, 0.25]) if m2_c.sum() >= 20 else ([c1], [m1_c], [1.0])
    b_cams, b_ms, b_ws = ([c1, c2], [m1_b, m2_b], [0.75, 0.25]) if m2_b.sum() >= 25 else ([c1], [m1_b], [1.0])
    cup_xy = refine_object_xy_two_view(m_r, d_r, "red_cup", init_c, tz_ep, c_cams, c_ms, camera_weights=c_ws, max_shift_m=0.10, max_iter=40)[0]
    bowl_xy = refine_object_xy_two_view(m_r, d_r, "pour_bowl", init_b, tz_ep, b_cams, b_ms, camera_weights=b_ws, max_shift_m=0.10, max_iter=40)[0]

    # Full 10-bean pouring physics rollout on MuJoCo 4.0.1
    m = build_demo3_scene(cup_asset, bowl_asset, table_tex, cup_xy, bowl_xy, table_z=tz_ep, n_beans=10, offwidth=OUT_WH[0], offheight=OUT_WH[1])
    d = mujoco.MjData(m)
    cid1 = configure_exact_camera(m, c1, OUT_WH[0], OUT_WH[1])
    act_grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")
    cup_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "red_cup")
    bowl_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pour_bowl")
    cup_gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == cup_bid]
    bowl_gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bowl_bid]

    d.qpos[:7] = qs[0]; d.ctrl[:7] = qs[0]; d.ctrl[act_grip] = float(np.clip(gs[0], 0.0, 1.0)) * 255.0
    mujoco.mj_forward(m, d)
    for _ in range(45):
        mujoco.mj_step(m, d)

    ren_seg = mujoco.Renderer(m, height=OUT_WH[1], width=OUT_WH[0])
    ren_seg.enable_segmentation_rendering()
    ren_seg.update_scene(d, camera=cid1)
    seg = ren_seg.render()[:, :, 0]
    f0_c_iou = compute_mask_iou(m1_c, np.isin(seg, cup_gids))
    f0_b_iou = compute_mask_iou(m1_b, np.isin(seg, bowl_gids))
    ren_seg.close()

    mon = RolloutMonitor(m, target_body_name="red_cup")
    mon.reset(d)
    for t_idx in range(len(qs)):
        d.ctrl[:7] = qs[t_idx]
        d.ctrl[act_grip] = (1.0 if gs[t_idx] > 0.15 else float(np.clip(gs[t_idx], 0.0, 1.0))) * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)
    for _ in range(80):
        d.ctrl[:7] = qs[-1]
        d.ctrl[act_grip] = float(np.clip(gs[-1], 0.0, 1.0)) * 255.0
        mujoco.mj_step(m, d)
    mon.observe(d)

    res = evaluate_pouring_beans(
        m, d, mon, bowl_body="pour_bowl", bean_prefix="bean_",
        bowl_radius_m=0.100, min_ratio_in_bowl=0.50, min_lift_m=0.06, table_z=tz_ep,
    )
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
        "ep_idx": ei, "episode_id": eid,
        "f0_cup_iou": round(float(f0_c_iou), 3),
        "f0_bowl_iou": round(float(f0_b_iou), 3),
        "snap_path": snap_path,
        **res.to_dict(),
    }


def main() -> None:
    t0 = time.perf_counter()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((BUNDLE / "meta_demo3.json").read_text())
    he = json.loads(HE_JSON.read_text())
    T1_ref, T2_ref = he["T_base_cam1_refined"], he["T_base_cam2_refined"]
    masks = dict(np.load(BUNDLE / "masks_demo3_74eps.npz"))
    colors = json.loads((WORK_MV / "demo3_colors.json").read_text())
    ei0 = colors["ep_ref"]

    # 1. Extract real tabletop Homography texture
    table_tex = ASSET_DIR / "demo3_real_table.png"
    ep0 = meta[ei0]
    f0_bgr = cv2.imread(str(BUNDLE / "episodes" / f"ep_{ei0:03d}" / "cam1_f0.jpg"))
    c1 = make_calib(ep0["c1_serial"], ep0["c1_extr"], ep0["c1_intr"]); c1.cam_to_base = T1_ref
    extract_real_table_texture(
        f0_bgr, c1, table_center_xy=(0.55, 0.0), table_half_xy=(0.50, 0.55),
        table_z=TABLE_Z, out_png=table_tex,
        exclude_masks=[masks[f"ep{ei0:03d}_c1_cup"], masks[f"ep{ei0:03d}_c1_bowl"]], tex_size=512,
    )
    print(f"[Demo3 Stage 2] Real tabletop texture -> {table_tex}", flush=True)

    # 2. Load cached CoACD assets if already built
    if (ASSET_DIR / "simready_red_cup" / "asset_meta.json").exists() and (ASSET_DIR / "simready_pour_bowl" / "asset_meta.json").exists():
        cup_asset = load_cached_simready(ASSET_DIR / "simready_red_cup")
        bowl_asset = load_cached_simready(ASSET_DIR / "simready_pour_bowl")
    else:
        cr = [min(1.0, c * 1.15) for c in colors["cup_rgb"]]
        br = [min(1.0, c * 1.10) for c in colors["bowl_rgb"]]
        cup_asset = build_sim_ready_asset(
            name="red_cup", raw_mesh_path=WORK_MV / "red_cup_hy3d_mv_raw.glb",
            out_dir=ASSET_DIR / "simready_red_cup", target_size_xyz=(0.074, 0.074, 0.105),
            rgba=(cr[0], cr[1], cr[2], 1.0), is_container=True, y_up_to_z_up=True,
            auto_upright_pca=False, taper_bot_scale=1.0, wall_thickness_m=0.003,
            coacd_threshold=0.025, max_convex_hulls=24,
        )
        bowl_asset = build_sim_ready_asset(
            name="pour_bowl", raw_mesh_path=WORK_MV / "pour_bowl_hy3d_mv_raw.glb",
            out_dir=ASSET_DIR / "simready_pour_bowl", target_size_xyz=(0.155, 0.155, 0.056),
            rgba=(br[0], br[1], br[2], 1.0), is_container=True, y_up_to_z_up=True,
            auto_upright_pca=False, wall_thickness_m=0.005, coacd_threshold=0.04, max_convex_hulls=20,
        )
    print(f"[Demo3 Stage 2] Built CoACD assets: cup={len(cup_asset.collision_hull_paths)} hulls, bowl={len(bowl_asset.collision_hull_paths)} hulls ({time.perf_counter()-t0:.1f}s)", flush=True)

    # 3. Run ALL 74 episodes in parallel on MuJoCo 4.0.1 + A100 EGL
    t_sim0 = time.perf_counter()
    n_workers = int(os.environ.get("R2S_WORKERS", "16"))
    tasks = []
    for ep in meta:
        ei = ep["ep_idx"]
        tasks.append((
            ep, T1_ref, T2_ref,
            masks[f"ep{ei:03d}_c1_cup"], masks[f"ep{ei:03d}_c1_bowl"],
            masks[f"ep{ei:03d}_c2_cup"], masks[f"ep{ei:03d}_c2_bowl"],
            ei < 20,
        ))

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(eval_demo3_ep, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if k % 15 == 0 or k == len(tasks):
                cp = sum(int(x["success"]) for x in results)
                cl = sum(int(x["metrics"]["max_lift_cm"] >= 6.0) for x in results)
                cb = sum(int(x["metrics"].get("beans_in_bowl", 0) >= 1) for x in results)
                print(f"  [Demo3 {k:2d}/{len(tasks)}] PASS={cp}/{k} ({100*cp/k:.1f}%) | Poured>=1={cb}/{k} | Lift={cl}/{k} ({time.perf_counter()-t_sim0:.1f}s)", flush=True)

    results.sort(key=lambda x: x["ep_idx"])
    n = len(results)
    n_pass = sum(int(x["success"]) for x in results)
    n_lift = sum(int(x["metrics"]["max_lift_cm"] >= 6.0) for x in results)
    n_poured = sum(int(x["metrics"].get("beans_in_bowl", 0) >= 1) for x in results)
    f0_cups = [x["f0_cup_iou"] for x in results if x["f0_cup_iou"] > 0]
    f0_bowls = [x["f0_bowl_iou"] for x in results if x["f0_bowl_iou"] > 0]
    reasons = Counter(x["failure_reason"] for x in results if not x["success"])

    summary = {
        "scene": "Demo #3 Pouring Beans (GuptaLab, 74 episodes)",
        "mujoco_version": mujoco.__version__,
        "timing_sec": {"total_stage2": round(time.perf_counter()-t0, 1), "sim_74eps": round(time.perf_counter()-t_sim0, 1)},
        "metrics": {
            "fact_pass": n_pass, "fact_pass_rate_pct": round(100*n_pass/n, 1),
            "lifted_ge_6cm": n_lift, "lift_rate_pct": round(100*n_lift/n, 1),
            "poured_ge_1_bean": n_poured,
            "mean_f0_cup_iou": round(float(np.mean(f0_cups)), 3) if f0_cups else 0.0,
            "mean_f0_bowl_iou": round(float(np.mean(f0_bowls)), 3) if f0_bowls else 0.0,
        },
        "failure_reasons": dict(reasons),
        "episodes": results,
    }
    (BUNDLE / "demo3_74eps_hy3d_mv_results.json").write_text(json.dumps(summary, indent=2))

    # Build 2x5 visual audit mosaic
    sp = [x for x in results if x.get("snap_path") and x["success"]][:5]
    sf = [x for x in results if x.get("snap_path") and not x["success"]][:5]
    rows = []
    for gn, grp, col in [("PASS", sp, (60, 230, 100)), ("FAIL", sf, (80, 110, 255))]:
        if not grp: continue
        tiles = []
        for x in grp:
            im = cv2.resize(cv2.imread(x["snap_path"]), (320, 180))
            cv2.rectangle(im, (0, 0), (320, 22), (15, 18, 22), -1)
            txt = f"Ep#{x['ep_idx']:02d} {gn} | L={x['metrics']['max_lift_cm']:.0f}cm Beans={x['metrics'].get('beans_in_bowl',0)}/10"
            cv2.putText(im, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            tiles.append(im)
        while len(tiles) < 5: tiles.append(np.zeros((180, 320, 3), np.uint8))
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(BUNDLE / "demo3_74eps_audit_mosaic.png"), np.vstack(rows))

    print("=" * 95, flush=True)
    print(f"DEMO #3 (74-EPISODE, MuJoCo {mujoco.__version__}) COMPLETE in {time.perf_counter()-t0:.1f}s (Sim74={time.perf_counter()-t_sim0:.1f}s)", flush=True)
    print(f"  Strict FACT PASS : {n_pass:2d}/{n} ({100*n_pass/n:.1f}%)", flush=True)
    print(f"  Poured >=1 Bean  : {n_poured:2d}/{n} | Lifted >= 6 cm: {n_lift:2d}/{n} ({100*n_lift/n:.1f}%)", flush=True)
    print(f"  Mean F0 IoU      : Cup={summary['metrics']['mean_f0_cup_iou']:.3f}, Bowl={summary['metrics']['mean_f0_bowl_iou']:.3f}", flush=True)
    print(f"  Failure Reasons  : {dict(reasons)}", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
