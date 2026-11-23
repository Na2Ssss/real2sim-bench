"""Universal Config-Driven Stage 2 (A100 MuJoCo 4.0.1): Sim-Ready Assets + Parallel Rollout + Showcase Video.

Works out-of-the-box for ANY 2-object tabletop DROID scene described by a JSON config:
  python remote_universal_stage2_mj4.py <scene_cfg.json>

Supported object collision primitives (decoupled from `Hunyuan3D-2mv` visual meshes):
  - `hollow_cylinder`: guaranteed-hollow 16-facet cavity ring (pots, bowls, mugs — never bridged by CoACD!)
  - `box`: crisp graspable cube/block (stacking blocks, bricks)
  - `cylinder`: crisp graspable cylinder (cans, cups, bottles)
  - `flat_plate`: wide low-rim receiving plate/tray
  - `ring`: hollow tape roll / torus approximated by graspable flat cylinder

Supported outcome predicates (from `r2s_bench.eval.task_success`):
  - `in_container`: `evaluate_object_in_container` (duck in pot, can in bowl, cup in bowl)
  - `on_target`: `evaluate_object_on_target` (orange block on green block, tape on plate, lid on pot)
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import subprocess
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
from scipy.optimize import minimize

ROOT = Path("/data2/r2s_bench")
sys.path.insert(0, str(ROOT))

from r2s_bench.adapters.droid import droid_ext6_to_matrix
from r2s_bench.assets.sim_ready_pipeline import build_sim_ready_asset, extract_real_table_texture
from r2s_bench.backends.mujoco_backend import opencv_cam_to_mujoco_pose
from r2s_bench.eval.task_success import RolloutMonitor, evaluate_object_in_container, evaluate_object_on_target
from r2s_bench.grading.l1_visual import compute_mask_iou
from r2s_bench.robots.franka_robotiq import build_droid_franka_robotiq_spec
from r2s_bench.schema import CameraCalib

OUT_WH = (480, 270)
OPT_WH = (240, 135)


def make_calib(serial: str, ext6: list, intr_dict: dict | None, fallback_intr: dict) -> CameraCalib:
    if not intr_dict or int(intr_dict.get("width", 0)) <= 0:
        intr_dict = fallback_intr
    fx, cx, fy, cy = intr_dict["cameraMatrix"]
    return CameraCalib(
        name=f"ext_{serial}", serial=serial,
        width=int(intr_dict["width"]), height=int(intr_dict["height"]),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        cam_to_base=droid_ext6_to_matrix(ext6).tolist(),
    )


def configure_cam(m: mujoco.MjModel, cid: int, calib: CameraCalib, w: int, h: int) -> None:
    sc = calib.scaled(w, h)
    pos, quat = opencv_cam_to_mujoco_pose(sc.T_base_cam)
    m.cam_pos[cid] = pos; m.cam_quat[cid] = quat
    px = 1e-5
    m.cam_resolution[cid] = [sc.width, sc.height]
    m.cam_sensorsize[cid] = [sc.width * px, sc.height * px]
    m.cam_intrinsic[cid] = [sc.fx * px, sc.fy * px, -(sc.cx - sc.width / 2.0) * px, (sc.cy - sc.height / 2.0) * px]


def ray_plane_z(calib: CameraCalib, uv: Tuple[float, float], z_plane: float) -> np.ndarray:
    T = np.asarray(calib.cam_to_base, dtype=np.float64)
    o = T[:3, 3]
    d = T[:3, :3] @ np.array([(uv[0] - calib.cx) / calib.fx, (uv[1] - calib.cy) / calib.fy, 1.0])
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    t = float(np.clip((z_plane - o[2]) / (d[2] if abs(d[2]) > 1e-5 else -1e-5), 0.15, 2.5))
    return o + t * d


def extract_grasp(qs: np.ndarray, gs: np.ndarray, pinch_frac: float = 0.60) -> dict:
    m_fk = build_droid_franka_robotiq_spec(offwidth=64, offheight=64).compile()
    d_fk = mujoco.MjData(m_fk)
    lp = [g for g in range(m_fk.ngeom) if "left_silicone_pad" in (mujoco.mj_id2name(m_fk, mujoco.mjtObj.mjOBJ_BODY, int(m_fk.geom_bodyid[g])) or "")][0]
    rp = [g for g in range(m_fk.ngeom) if "right_silicone_pad" in (mujoco.mj_id2name(m_fk, mujoco.mjtObj.mjOBJ_BODY, int(m_fk.geom_bodyid[g])) or "")][0]
    ag = mujoco.mj_name2id(m_fk, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")
    d_fk.qpos[:7] = qs[0]; d_fk.ctrl[:7] = qs[0]; d_fk.ctrl[ag] = pinch_frac * 255.0
    for _ in range(150):
        mujoco.mj_step(m_fk, d_fk)
    q_pinch = d_fk.qpos[7:].copy()
    T = len(qs); mids = np.zeros((T, 3), dtype=np.float64); axes = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        d_fk.qpos[:7] = qs[t]; d_fk.qpos[7:] = q_pinch; mujoco.mj_forward(m_fk, d_fk)
        pl, pr = d_fk.geom_xpos[lp], d_fk.geom_xpos[rp]
        mids[t] = 0.5 * (pl + pr)
        v = pl - pr; axes[t] = v / max(float(np.linalg.norm(v)), 1e-6)
    n1 = int(T * 0.65)
    g_peak = float(np.max(gs[:n1]))
    thresh = max(0.16, g_peak * 0.55)
    cc = [t for t in range(3, n1) if gs[t] >= thresh and gs[t - 1] < thresh]
    t_close = min(cc, key=lambda t: mids[t, 2]) if cc else int(np.argmax(gs[:n1]))
    t_hold = min(t_close + 4, T - 1)
    oc = [t for t in range(t_hold + 5, T) if gs[t] < thresh * 0.45]
    t_open = oc[0] if oc else T - 1
    return {
        "t_close": int(t_close), "t_open": int(t_open),
        "p_close": mids[t_hold], "p_open": mids[t_open],
        "yaw_grasp": float(math.atan2(axes[t_hold, 1], axes[t_hold, 0])),
    }


def add_scene_object(spec: mujoco.MjSpec, obj_cfg: dict, xy: Tuple[float, float], yaw: float, table_z: float, vis_obj: Path | None, rgb: List[float]) -> None:
    name = obj_cfg["name"]
    shape = obj_cfg["collision_shape"]
    sx, sy, sz = [float(v) for v in obj_cfg["size_xyz"]]
    mass = float(obj_cfg.get("mass_kg", 0.08))
    fric = list(obj_cfg.get("friction", [1.8, 0.02, 0.002]))
    q = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], table_z + 0.001], quat=q)
    if obj_cfg.get("freejoint", True):
        body.add_freejoint(name=f"{name}_jnt")

    has_vis = vis_obj is not None and vis_obj.exists()
    if has_vis:
        spec.add_mesh(name=f"{name}_vis_mesh", file=str(vis_obj))
        body.add_geom(
            name=f"{name}_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_vis_mesh",
            rgba=[rgb[0], rgb[1], rgb[2], 1.0], contype=0, conaffinity=0, mass=0.0,
        )

    col_alpha = 0.0 if has_vis else 1.0
    col_rgba = [rgb[0], rgb[1], rgb[2], col_alpha]

    if shape == "hollow_cylinder":
        r_out = 0.5 * max(sx, sy)
        wall = float(obj_cfg.get("wall_m", 0.004))
        body.add_geom(name=f"{name}_bot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.004], size=[r_out, 0.004, 0], rgba=col_rgba, mass=mass * 0.65, friction=[1.1, 0.01, 0.001])
        r_mid = r_out - 0.5 * wall
        n_seg = 16
        seg_w = 2.0 * math.pi * r_mid / n_seg * 0.56
        for k in range(n_seg):
            a = 2.0 * math.pi * k / n_seg
            body.add_geom(
                name=f"{name}_w{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[r_mid * math.cos(a), r_mid * math.sin(a), 0.5 * sz],
                quat=[math.cos(a / 2.0), 0, 0, math.sin(a / 2.0)],
                size=[0.5 * wall, seg_w, 0.5 * sz], rgba=col_rgba, mass=(mass * 0.35) / n_seg, friction=fric, solref=[0.022, 1.4],
            )
    elif shape == "flat_plate":
        r_out = 0.5 * max(sx, sy)
        body.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.5 * sz], size=[r_out, 0.5 * sz, 0], rgba=col_rgba, mass=mass, friction=[1.1, 0.01, 0.001], solref=[0.022, 1.4])
    elif shape == "box":
        body.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.5 * sz], size=[0.5 * sx, 0.5 * sy, 0.5 * sz], rgba=col_rgba, mass=mass, friction=fric, solref=[0.022, 1.5])
    elif shape == "cylinder":
        r_out = 0.5 * min(sx, sy)
        body.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.5 * sz], size=[r_out, 0.5 * sz, 0], rgba=col_rgba, mass=mass, friction=fric, solref=[0.022, 1.5])
    elif shape == "ring":
        # Graspable tape roll / ring: solid graspable inner cylinder + rim (d matches sx)
        r_out = 0.5 * max(sx, sy)
        body.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.5 * sz], size=[r_out, 0.5 * sz, 0], rgba=col_rgba, mass=mass, friction=fric, solref=[0.022, 1.5])


def build_universal_scene(
    cfg: dict,
    obj_xys: dict,
    obj_yaws: dict,
    obj_rgbs: dict,
    vis_paths: dict,
    table_z: float,
    table_tex: Path | None = None,
    with_robot: bool = True,
    offwidth: int = 480,
    offheight: int = 270,
) -> mujoco.MjModel:
    if with_robot:
        spec = build_droid_franka_robotiq_spec(offwidth=offwidth, offheight=offheight)
    else:
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth = offwidth; spec.visual.global_.offheight = offheight
        spec.worldbody.add_light(pos=[0.5, 0.0, 1.5], dir=[0, 0, -1])
        spec.worldbody.add_camera(name="eval_cam", pos=[0, 0, 1], fovy=60)

    tbl_mat = None
    if table_tex is not None and table_tex.exists():
        spec.add_texture(name="u_tbl_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(table_tex))
        m_tbl = spec.add_material(name="u_tbl_mat", rgba=[1, 1, 1, 1], specular=0.10, shininess=0.18)
        m_tbl.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "u_tbl_tex"
        tbl_mat = "u_tbl_mat"
    tcx, tcy = cfg.get("table_center_xy", [0.55, 0.0])
    thx, thy = cfg.get("table_half_xy", [0.55, 0.65])
    g_tbl = spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[tcx, tcy, table_z - 0.025], size=[thx, thy, 0.025], friction=[0.9, 0.005, 0.0001])
    if tbl_mat:
        g_tbl.material = tbl_mat
    else:
        g_tbl.rgba = cfg.get("table_rgba", [0.85, 0.82, 0.76, 1.0])

    for o in cfg["objects"]:
        oname = o["name"]
        add_scene_object(spec, o, obj_xys[oname], obj_yaws.get(oname, 0.0), table_z, vis_paths.get(oname), obj_rgbs[oname])

    model = spec.compile()
    if with_robot:
        src_name = cfg["objects"][0]["name"]
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if src_name in nm or "pad" in nm:
                model.geom_condim[g] = 4
    return model


def eval_ep(args: tuple) -> dict:
    ep, cfg, vis_paths, s1_colors, fallback_intr, ep_masks, save_vid = args
    ei = ep["ep_idx"]; eid = ep["episode_id"]
    bundle = Path(cfg["bundle_dir"]); asset_dir = Path(cfg["asset_dir"])
    ed = bundle / "episodes" / f"ep_{ei:03d}"
    tr = np.load(ed / "trajectory.npz")
    qs, gs = tr["qpos"], tr["gripper"]
    c1 = make_calib(ep["c1_serial"], ep["c1_extr"], ep["c1_intr"], fallback_intr["c1"])
    c2 = make_calib(ep["c2_serial"], ep["c2_extr"], ep["c2_intr"], fallback_intr["c2"])

    src_cfg, tgt_cfg = cfg["objects"][0], cfg["objects"][1]
    src_name, tgt_name = src_cfg["name"], tgt_cfg["name"]
    ev = extract_grasp(qs, gs, pinch_frac=float(cfg.get("pinch_frac", 0.62)))
    pc, po = ev["p_close"], ev["p_open"]

    # Ground table_z cleanly per episode (handles motorized/varying desks automatically!)
    grasp_h = float(src_cfg.get("grasp_height_m", 0.5 * src_cfg["size_xyz"][2]))
    tz = float(np.clip(pc[2] - grasp_h, -0.18, 0.35))

    obj_rgbs = {}
    for o in cfg["objects"]:
        nm = o["name"]
        obj_rgbs[nm] = s1_colors["ep_rgbs"].get(nm, {}).get(str(ei), s1_colors["med_rgbs"].get(nm, o.get("default_rgb", [0.6, 0.6, 0.6])))

    # Refine both objects' XY via Dual-View silhouette IoU, anchored safely to grasp/drop region
    obj_xys = {}
    obj_ious = {}
    anchors = {src_name: pc[:2], tgt_name: po[:2]}
    for o in cfg["objects"]:
        nm = o["name"]
        m1, m2 = ep_masks[f"c1_{nm}"], ep_masks[f"c2_{nm}"]
        has1, has2 = int(m1.sum()) >= 80, int(m2.sum()) >= 80
        anc = anchors[nm]
        xy0 = anc.copy()
        if has1:
            y1, x1 = np.where(m1)
            cand = ray_plane_z(c1.scaled(OUT_WH[0], OUT_WH[1]), (float(np.median(x1)), float(np.median(y1))), tz + 0.5 * o["size_xyz"][2])[:2]
            if float(np.linalg.norm(cand - anc)) <= 0.15:
                xy0 = cand
        iou_v = 0.0
        vis_xy = (float(xy0[0]), float(xy0[1]))
        if has1 or has2:
            # Quick single-object MuJoCo silhouette opt
            tmp_cfg = dict(cfg, objects=[o])
            m_o = build_universal_scene(tmp_cfg, {nm: vis_xy}, {nm: 0.0}, {nm: obj_rgbs[nm]}, {nm: vis_paths.get(nm)}, tz, with_robot=False, offwidth=OPT_WH[0], offheight=OPT_WH[1])
            d_o = mujoco.MjData(m_o)
            bid = mujoco.mj_name2id(m_o, mujoco.mjtObj.mjOBJ_BODY, nm)
            gids = [g for g in range(m_o.ngeom) if m_o.geom_bodyid[g] == bid]
            ren_o = mujoco.Renderer(m_o, OPT_WH[1], OPT_WH[0]); ren_o.enable_segmentation_rendering()
            rm1 = cv2.resize(m1.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
            rm2 = cv2.resize(m2.astype(np.uint8), OPT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)

            def ev_xy(x):
                if abs(x[0] - xy0[0]) > 0.12 or abs(x[1] - xy0[1]) > 0.12:
                    return 0.0
                d_o.qpos[:3] = [float(x[0]), float(x[1]), tz + 0.001]; d_o.qpos[3:7] = [1, 0, 0, 0]
                s, ws = 0.0, 0.0
                if has1:
                    configure_cam(m_o, 0, c1, OPT_WH[0], OPT_WH[1]); mujoco.mj_forward(m_o, d_o); ren_o.update_scene(d_o, camera=0)
                    s += 0.75 * compute_mask_iou(rm1, np.isin(ren_o.render()[:, :, 0], gids)); ws += 0.75
                if has2:
                    configure_cam(m_o, 0, c2, OPT_WH[0], OPT_WH[1]); mujoco.mj_forward(m_o, d_o); ren_o.update_scene(d_o, camera=0)
                    s += 0.25 * compute_mask_iou(rm2, np.isin(ren_o.render()[:, :, 0], gids)); ws += 0.25
                return s / max(ws, 1e-6)

            r_opt = minimize(lambda x: 1.0 - ev_xy(x), xy0, method="Nelder-Mead", options={"maxiter": 38, "xatol": 2e-3, "fatol": 2e-3})
            vis_xy = (float(r_opt.x[0]), float(r_opt.x[1]))
            iou_v = float(1.0 - r_opt.fun)
            ren_o.close()
        max_dev = float(o.get("anchor_max_dev_m", 0.032 if nm == src_name else 0.065))
        obj_xys[nm] = vis_xy if float(np.linalg.norm(np.array(vis_xy) - anc)) <= max_dev else (float(anc[0]), float(anc[1]))
        obj_ious[nm] = round(iou_v, 3)

    # Ensure source and target don't overlap at t=0
    if float(np.linalg.norm(np.array(obj_xys[src_name]) - np.array(obj_xys[tgt_name]))) < 0.07:
        obj_xys[tgt_name] = (float(po[0]), float(po[1]))

    tbl_tex = None
    if save_vid and cfg.get("use_homography_table", True):
        tbl_tex = asset_dir / f"tbl_ep{ei:03d}.png"
        tcx, tcy = cfg.get("table_center_xy", [0.55, 0.0])
        thx, thy = cfg.get("table_half_xy", [0.55, 0.65])
        extract_real_table_texture(
            frame_bgr=cv2.imread(str(ed / "cam1_f0.jpg")), calib=c1,
            table_center_xy=(tcx, tcy), table_half_xy=(thx, thy), table_z=tz,
            out_png=tbl_tex, exclude_masks=[ep_masks[f"c1_{src_name}"], ep_masks[f"c1_{tgt_name}"]],
        )

    m = build_universal_scene(
        cfg, obj_xys, {src_name: ev["yaw_grasp"], tgt_name: 0.0}, obj_rgbs, vis_paths, tz,
        table_tex=tbl_tex, with_robot=True, offwidth=OUT_WH[0], offheight=OUT_WH[1],
    )
    d = mujoco.MjData(m)
    ag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator")
    d.qpos[:7] = qs[0]; d.ctrl[:7] = qs[0]; d.ctrl[ag] = float(np.clip(gs[0], 0, 1)) * 255.0
    mujoco.mj_forward(m, d)
    for _ in range(30):
        mujoco.mj_step(m, d)

    ren = None; od = None; ts_saved = []
    if save_vid:
        configure_cam(m, 0, c1, OUT_WH[0], OUT_WH[1])
        ren = mujoco.Renderer(m, OUT_WH[1], OUT_WH[0])
        od = ROOT / "outputs" / "showcase_sim_frames" / f"{cfg['scene_id']}_ep{ei:03d}"
        od.mkdir(parents=True, exist_ok=True)

    hold_ctrl = float(cfg.get("hold_ctrl", 0.82))
    mon = RolloutMonitor(m, target_body_name=src_name); mon.reset(d)
    for ti in range(len(qs)):
        d.ctrl[:7] = qs[ti]
        d.ctrl[ag] = (hold_ctrl if (max(0, ev["t_close"] - 2) <= ti < ev["t_open"]) else float(np.clip(gs[ti], 0, 1))) * 255.0
        for s in range(33):
            mujoco.mj_step(m, d)
            if (s & 3) == 0:
                mon.observe(d)
        if ren is not None and (ti % 3 == 0 or ti == len(qs) - 1):
            ren.update_scene(d, camera=0)
            cv2.imwrite(str(od / f"f_{ti:04d}.jpg"), cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            ts_saved.append(ti)

    for _ in range(60):
        d.ctrl[:7] = qs[-1]; d.ctrl[ag] = float(np.clip(gs[-1], 0, 1)) * 255.0
        mujoco.mj_step(m, d)
    mon.observe(d)
    if ren is not None:
        ren.close()
        (od / "info.json").write_text(json.dumps({"ep_idx": ei, "episode_id": eid, "c1_serial": ep["c1_serial"], "ts": ts_saved, "ious": obj_ious}))

    oc = cfg["outcome"]
    if oc["type"] == "in_container":
        res = evaluate_object_in_container(
            m, d, mon, object_body=src_name, container_body=tgt_name,
            max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [-0.02, 0.12]),
            table_z=tz, task_name=cfg["scene_id"],
        )
    else:
        res = evaluate_object_on_target(
            m, d, mon, object_body=src_name, target_body=tgt_name,
            max_xy_dist_m=float(oc.get("max_xy_m", 0.095)), z_range_rel=oc.get("z_range", [0.005, 0.14]),
            task_name=cfg["scene_id"],
        )
    out = res.to_dict()
    for nm, iv in obj_ious.items():
        out["metrics"][f"f0_{nm}_iou"] = iv
    return {"ep_idx": ei, "episode_id": eid, **out}


def main() -> None:
    t0 = time.perf_counter()
    cfg = json.loads(Path(sys.argv[1]).read_text())
    sid = cfg["scene_id"]
    bundle = Path(cfg["bundle_dir"]); asset_dir = Path(cfg["asset_dir"])
    meta = json.loads((bundle / "meta.json").read_text())
    masks = dict(np.load(bundle / "masks.npz"))
    s1_colors = json.loads((asset_dir / "stage1_colors.json").read_text())

    # Build fallback intrinsics from first valid episode
    fb_c1 = next(m["c1_intr"] for m in meta if m.get("c1_intr") and int(m["c1_intr"].get("width", 0)) > 0)
    fb_c2 = next(m["c2_intr"] for m in meta if m.get("c2_intr") and int(m["c2_intr"].get("width", 0)) > 0)
    fallback_intr = {"c1": fb_c1, "c2": fb_c2}

    # Build visual sim-ready meshes (auto_upright_pca=False!) for objects that generated Hunyuan3D-2mv GLBs
    vis_paths = {}
    for o in cfg["objects"]:
        nm = o["name"]
        raw_glb = asset_dir / f"{nm}_hy3d_mv_raw.glb"
        if raw_glb.exists():
            odir = asset_dir / f"sim_ready_{nm}"
            mj = odir / "asset_meta.json"
            if not mj.exists():
                rgb = s1_colors["med_rgbs"].get(nm, [0.7, 0.7, 0.7])
                asset = build_sim_ready_asset(
                    name=nm, raw_mesh_path=raw_glb, out_dir=odir,
                    target_size_xyz=tuple(o["size_xyz"]), rgba=(rgb[0], rgb[1], rgb[2], 1.0),
                    is_container=(o["collision_shape"] == "hollow_cylinder"),
                    auto_upright_pca=False, wall_thickness_m=float(o.get("wall_m", 0.004)), max_convex_hulls=12,
                )
                vis_paths[nm] = asset.visual_mesh_path
            else:
                vis_paths[nm] = Path(json.loads(mj.read_text())["visual_mesh_path"])

    t_sim0 = time.perf_counter()
    tasks = []
    for ep in meta:
        ei = ep["ep_idx"]
        ep_m = {}
        for o in cfg["objects"]:
            nm = o["name"]
            ep_m[f"c1_{nm}"] = masks[f"ep{ei:03d}_c1_{nm}"]
            ep_m[f"c2_{nm}"] = masks[f"ep{ei:03d}_c2_{nm}"]
        tasks.append((ep, cfg, vis_paths, s1_colors, fallback_intr, ep_m, ei < 16))

    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R2S_WORKERS", "12")), mp_context=ctx) as pool:
        for k, f in enumerate(as_completed([pool.submit(eval_ep, t) for t in tasks]), 1):
            results.append(f.result())
            if k % 25 == 0 or k == len(tasks):
                cp = sum(int(x["success"]) for x in results)
                print(f"  [{sid} {k:2d}/{len(tasks)}] PASS={cp}/{k} ({100*cp/k:.1f}%) ({time.perf_counter()-t_sim0:.1f}s)", flush=True)

    results.sort(key=lambda x: x["ep_idx"])
    n = len(results); n_pass = sum(int(x["success"]) for x in results)
    reasons = Counter(x["failure_reason"] for x in results if not x["success"])
    src_nm = cfg["objects"][0]["name"]; tgt_nm = cfg["objects"][1]["name"]
    iou_src = float(np.mean([x["metrics"][f"f0_{src_nm}_iou"] for x in results if x["metrics"][f"f0_{src_nm}_iou"] > 0] or [0.0]))
    iou_tgt = float(np.mean([x["metrics"][f"f0_{tgt_nm}_iou"] for x in results if x["metrics"][f"f0_{tgt_nm}_iou"] > 0] or [0.0]))

    summary = {
        "scene": cfg["title"], "scene_id": sid, "mujoco_version": mujoco.__version__,
        "timing_sec": {"total": round(time.perf_counter() - t0, 1), "sim": round(time.perf_counter() - t_sim0, 1)},
        "metrics": {
            "outcome_pass": n_pass, "outcome_pass_pct": round(100.0 * n_pass / n, 1),
            f"mean_f0_{src_nm}_iou": round(iou_src, 3), f"mean_f0_{tgt_nm}_iou": round(iou_tgt, 3),
        },
        "failure_reasons": dict(reasons), "episodes": results,
    }
    (asset_dir / f"{sid}_results.json").write_text(json.dumps(summary, indent=2))
    print("=" * 90, flush=True)
    print(f"[{sid}] COMPLETE in {time.perf_counter()-t0:.1f}s (Sim={time.perf_counter()-t_sim0:.1f}s)", flush=True)
    print(f"  Outcome PASS : {n_pass}/{n} ({100*n_pass/n:.1f}%) | IoU({src_nm}={iou_src:.3f}, {tgt_nm}={iou_tgt:.3f}) | Failures={dict(reasons)}", flush=True)
    print("=" * 90, flush=True)

    # Composite showcase H.264 video + keyframe audit strip directly on A100
    cands = sorted(
        [e for e in results if e["ep_idx"] < 16 and e["success"]],
        key=lambda e: -(e["metrics"][f"f0_{src_nm}_iou"] + e["metrics"][f"f0_{tgt_nm}_iou"]),
    )[:4]
    S = ROOT / "outputs" / "showcase_sim_frames"
    all_frames, strip_rows = [], []
    for e in cands:
        ei = e["ep_idx"]; od = S / f"{sid}_ep{ei:03d}"
        info = json.loads((od / "info.json").read_text())
        c1s = info["c1_serial"]
        cap = cv2.VideoCapture(str(bundle / f"episodes/ep_{ei:03d}" / f"{c1s}.mp4"))
        ep_fr = []
        for ti in info["ts"]:
            sim = cv2.imread(str(od / f"f_{ti:04d}.jpg"))
            if sim is None: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, ti); ok, rf = cap.read()
            if not ok: continue
            real = cv2.resize(rf, OUT_WH); ov = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
            bar = np.zeros((32, OUT_WH[0] * 3, 3), dtype=np.uint8); bar[:] = (22, 24, 28)
            cv2.putText(bar, f"{cfg['title']} (MuJoCo 4.0.1) | Ep#{ei:02d} t={ti:03d} | Outcome PASS {n_pass}/{n} ({100*n_pass/n:.1f}%)",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 245, 130), 1, cv2.LINE_AA)
            lbl = np.hstack([real, sim, ov])
            cv2.putText(lbl, "REAL CAMERA", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(lbl, "SIM (MuJoCo 4.0.1 + Hunyuan3D-2mv)", (OUT_WH[0] + 10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 245, 130), 1, cv2.LINE_AA)
            cv2.putText(lbl, "50/50 OVERLAY", (2 * OUT_WH[0] + 10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 255), 1, cv2.LINE_AA)
            ep_fr.append(np.vstack([bar, lbl]))
        cap.release()
        if ep_fr:
            all_frames.extend(ep_fr + [ep_fr[-1]] * 10)
            if len(strip_rows) < 2:
                strip_rows.append(np.vstack([ep_fr[k] for k in (0, len(ep_fr)//3, (2*len(ep_fr))//3, len(ep_fr)-1)]))
    if all_frames:
        raw_mp4 = asset_dir / f"raw_{sid}.mp4"; out_mp4 = asset_dir / f"{sid}_mj4_showcase.mp4"
        h, w = all_frames[0].shape[:2]
        vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
        for fr in all_frames: vw.write(fr)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-i", str(raw_mp4), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", str(out_mp4)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cv2.imwrite(str(asset_dir / f"inspect_{sid}.png"), np.vstack(strip_rows))
        print(f"[{sid} VIDEO] Saved {out_mp4} ({out_mp4.stat().st_size/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
