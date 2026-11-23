"""Official DROID robot assembly (Franka Emika Panda + Robotiq 2F-85) via MuJoCo MjSpec.

Also provides parametric tabletop scene builders (hollow thin-wall vs. naive solid)
for the Demo #1 benchmark case (`put the cup in the bowl`).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np

if os.environ.get("R2S_MENAGERIE_DIR"):
    MENAGERIE_DIR = Path(os.environ["R2S_MENAGERIE_DIR"])
else:
    from robot_descriptions import panda_mj_description
    MENAGERIE_DIR = Path(panda_mj_description.MJCF_PATH).parent.parent
PANDA_NOHAND_XML = MENAGERIE_DIR / "franka_emika_panda" / "panda_nohand.xml"
ROBOTIQ_2F85_XML = MENAGERIE_DIR / "robotiq_2f85" / "2f85.xml"


def build_droid_franka_robotiq_spec(
    offwidth: int = 1280,
    offheight: int = 720,
    gripper_yaw_deg: float = 45.0,
) -> mujoco.MjSpec:
    """Attach Menagerie Robotiq 2F-85 onto Menagerie Franka Panda link7 flange."""
    spec = mujoco.MjSpec.from_file(str(PANDA_NOHAND_XML))
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = offwidth
    spec.visual.global_.offheight = offheight

    gripper_spec = mujoco.MjSpec.from_file(str(ROBOTIQ_2F85_XML))
    link7 = spec.body("link7")
    if link7 is None:
        raise RuntimeError("link7 body not found in panda_nohand.xml")

    half = math.radians(gripper_yaw_deg) / 2.0
    frame = link7.add_frame(
        pos=[0.0, 0.0, 0.107],
        quat=[math.cos(half), 0.0, 0.0, math.sin(half)],
    )
    spec.attach(gripper_spec, prefix="robotiq_", frame=frame)

    # Add evaluation camera and overhead lighting
    spec.worldbody.add_light(pos=[0.5, -0.2, 2.0], dir=[0.0, 0.0, -1.0], diffuse=[0.9, 0.9, 0.9])
    spec.worldbody.add_camera(name="eval_cam", pos=[1.0, 0.0, 1.0], fovy=55.0)
    return spec


def _add_hollow_vessel(
    body: mujoco.MjsBody,
    name_prefix: str,
    radius_bottom: float,
    radius_top: float,
    height: float,
    wall_thickness: float,
    rgba: Tuple[float, float, float, float],
    total_mass: float,
    n_sides: int = 16,
    friction: Tuple[float, float, float] = (1.2, 0.01, 0.001),
) -> None:
    """Build a physically hollow thin-wall cup/bowl using a ring of box geoms + bottom disc."""
    r_mid = 0.5 * (radius_bottom + radius_top)
    tilt = math.atan2(radius_top - radius_bottom, height)
    seg_w = 2.0 * r_mid * math.tan(math.pi / n_sides) * 1.05
    seg_h = math.hypot(height, radius_top - radius_bottom)

    # Bottom disc
    bottom_mass = total_mass * 0.35
    body.add_geom(
        name=f"{name_prefix}_bottom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, wall_thickness / 2.0],
        size=[radius_bottom, wall_thickness / 2.0, 0.0],
        rgba=list(rgba),
        mass=bottom_mass,
        friction=list(friction),
    )

    wall_mass = (total_mass * 0.65) / n_sides
    for i in range(n_sides):
        theta = 2.0 * math.pi * i / n_sides
        cx = r_mid * math.cos(theta)
        cy = r_mid * math.sin(theta)
        cz = height / 2.0 + wall_thickness * 0.5
        # Rotate around Z by theta, then tilt slightly outward around local Y
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler("z", theta) * R.from_euler("y", tilt)
        q_xyzw = rot.as_quat()
        quat_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
        body.add_geom(
            name=f"{name_prefix}_wall_{i:02d}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[cx, cy, cz],
            quat=quat_wxyz,
            size=[wall_thickness / 2.0, seg_w / 2.0, seg_h / 2.0],
            rgba=list(rgba),
            mass=wall_mass,
            friction=list(friction),
        )


def build_cup_in_bowl_scene(
    cup_xy: Tuple[float, float],
    bowl_xy: Tuple[float, float],
    table_z: float = -0.02,
    hollow: bool = True,
    cup_friction: float = 1.8,
    cup_dims: Tuple[float, float, float] = (0.030, 0.040, 0.090),
    bowl_dims: Tuple[float, float, float] = (0.060, 0.085, 0.055),
    torsional_condim: bool = True,
    offwidth: int = 640,
    offheight: int = 360,
) -> mujoco.MjModel:
    """Compile a MuJoCo model for Demo #1 (`put the cup in the bowl`).

    Args:
        cup_xy: (x, y) initial position of orange cup on table in robot base frame.
        bowl_xy: (x, y) initial position of black bowl on table in robot base frame.
        table_z: table top surface z height in robot base frame.
        hollow: True = realistic hollow thin-wall cup & concave bowl;
                False = naive solid cylinder cup & solid bowl (baseline failure mode).
        cup_dims: (radius_bottom, radius_top, height) in meters.
        bowl_dims: (radius_bottom, radius_top, height) in meters.
    """
    spec = build_droid_franka_robotiq_spec(offwidth=offwidth, offheight=offheight)

    # Wooden tabletop surface
    spec.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.55, 0.0, table_z - 0.02],
        size=[0.55, 0.65, 0.02],
        rgba=[0.78, 0.58, 0.34, 1.0],
        friction=[0.9, 0.005, 0.0001],
    )

    cup_rb, cup_rt, cup_h = cup_dims
    bowl_rb, bowl_rt, bowl_h = bowl_dims

    # Orange cup
    cup_body = spec.worldbody.add_body(name="orange_cup", pos=[cup_xy[0], cup_xy[1], table_z + 0.001])
    cup_body.add_freejoint(name="orange_cup_jnt")
    if hollow:
        _add_hollow_vessel(
            cup_body,
            name_prefix="orange_cup",
            radius_bottom=cup_rb,
            radius_top=cup_rt,
            height=cup_h,
            wall_thickness=0.0035,
            rgba=(0.95, 0.35, 0.08, 1.0),
            total_mass=0.045,
            n_sides=16,
            friction=(cup_friction, 0.015, 0.001),
        )
    else:
        # Naive solid cylinder (cannot pinch the rim!)
        cup_body.add_geom(
            name="orange_cup_solid",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.0, 0.0, cup_h / 2.0],
            size=[cup_rt, cup_h / 2.0, 0.0],
            rgba=[0.95, 0.35, 0.08, 1.0],
            mass=0.045,
            friction=[cup_friction, 0.015, 0.001],
        )

    # Black bowl
    bowl_body = spec.worldbody.add_body(name="black_bowl", pos=[bowl_xy[0], bowl_xy[1], table_z + 0.001])
    bowl_body.add_freejoint(name="black_bowl_jnt")
    if hollow:
        _add_hollow_vessel(
            bowl_body,
            name_prefix="black_bowl",
            radius_bottom=bowl_rb,
            radius_top=bowl_rt,
            height=bowl_h,
            wall_thickness=0.005,
            rgba=(0.12, 0.12, 0.13, 1.0),
            total_mass=0.28,
            n_sides=16,
            friction=(0.9, 0.01, 0.001),
        )
    else:
        bowl_body.add_geom(
            name="black_bowl_solid",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.0, 0.0, bowl_h / 2.0],
            size=[bowl_rt, bowl_h / 2.0, 0.0],
            rgba=[0.12, 0.12, 0.13, 1.0],
            mass=0.28,
            friction=[0.9, 0.01, 0.001],
        )

    model = spec.compile()
    if torsional_condim:
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "orange_cup" in nm or "pad" in nm:
                model.geom_condim[g] = 4
    return model


def build_drawer_block_scene(
    block_xy: Tuple[float, float],
    table_z: float = 0.208,
    drawer_pos: Tuple[float, float, float] = (0.465, 0.055, 0.065),
    drawer_yaw_rad: float = 0.448,
    articulated: bool = True,
    offwidth: int = 480,
    offheight: int = 270,
) -> mujoco.MjModel:
    """Compile MuJoCo model for Demo #2 (`put the blue block in the drawer and close it`).

    Features a 1-DoF prismatic sliding drawer below a wooden countertop plus a freejoint
    light-blue Lego block. `articulated=False` welds the drawer static as the naive baseline.
    """
    spec = build_droid_franka_robotiq_spec(offwidth=offwidth, offheight=offheight)

    c_yaw, s_yaw = math.cos(drawer_yaw_rad), math.sin(drawer_yaw_rad)
    quat_yaw = [math.cos(drawer_yaw_rad / 2.0), 0.0, 0.0, math.sin(drawer_yaw_rad / 2.0)]

    # Countertop positioned behind the open drawer cavity with >= 1.8cm vertical clearance above drawer walls
    ct_cx = drawer_pos[0] + 0.35 * c_yaw
    ct_cy = drawer_pos[1] + 0.35 * s_yaw
    spec.worldbody.add_geom(
        name="counter_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[ct_cx, ct_cy, table_z - 0.012],
        quat=quat_yaw,
        size=[0.22, 0.55, 0.012],
        rgba=[0.82, 0.62, 0.32, 1.0],
        friction=[0.9, 0.005, 0.0001],
    )

    # Sliding drawer box (open at top; top rim strictly below countertop underside)
    drawer_z = min(drawer_pos[2], table_z - 0.130)
    drawer_body = spec.worldbody.add_body(
        name="drawer_box",
        pos=[drawer_pos[0], drawer_pos[1], drawer_z],
        quat=quat_yaw,
    )
    if articulated:
        drawer_body.add_joint(
            name="drawer_slide",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1.0, 0.0, 0.0],
            range=[0.0, 0.32],
            damping=1.5,
            frictionloss=0.15,
        )

    # Drawer geometry in drawer local frame (depth along X: 0.24m, width along Y: 0.36m, wall height: 0.092m)
    d_rgba = [0.22, 0.23, 0.25, 1.0]
    front_rgba = [0.35, 0.37, 0.40, 1.0]
    drawer_body.add_geom(name="drawer_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.0, 0.005], size=[0.12, 0.18, 0.005], rgba=d_rgba, mass=0.35)
    drawer_body.add_geom(name="drawer_front", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.115, 0.0, 0.048], size=[0.008, 0.185, 0.048], rgba=front_rgba, mass=0.18)
    drawer_body.add_geom(name="drawer_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.115, 0.0, 0.042], size=[0.006, 0.18, 0.042], rgba=d_rgba, mass=0.12)
    drawer_body.add_geom(name="drawer_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.175, 0.042], size=[0.12, 0.006, 0.042], rgba=d_rgba, mass=0.10)
    drawer_body.add_geom(name="drawer_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, -0.175, 0.042], size=[0.12, 0.006, 0.042], rgba=d_rgba, mass=0.10)

    # Light-blue Lego block on countertop (slightly taller 3.0cm height to match Lego Duplo brick)
    blk_body = spec.worldbody.add_body(
        name="blue_block",
        pos=[block_xy[0], block_xy[1], table_z + 0.016],
    )
    blk_body.add_freejoint(name="blue_block_jnt")
    blk_body.add_geom(
        name="blue_block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.0],
        size=[0.020, 0.015, 0.015],
        rgba=[0.18, 0.72, 0.95, 1.0],
        mass=0.025,
        friction=[1.8, 0.02, 0.002],
    )

    model = spec.compile()
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "blue_block" in nm or "pad" in nm:
            model.geom_condim[g] = 4
    return model


def build_pouring_cup_bowl_scene(
    cup_xy: Tuple[float, float],
    bowl_xy: Tuple[float, float],
    table_z: float = 0.048,
    hollow: bool = True,
    n_beans: int = 8,
    cup_dims: Tuple[float, float, float] = (0.030, 0.042, 0.105),
    bowl_dims: Tuple[float, float, float] = (0.062, 0.090, 0.060),
    offwidth: int = 480,
    offheight: int = 270,
) -> mujoco.MjModel:
    """Compile MuJoCo model for Demo #3 (`pour the contents of the cup into the bowl`).

    Features a thin-wall hollow red cup containing `n_beans` freejoint granular spheres
    and a concave receiving bowl. `hollow=False` replaces the cup with a solid cylinder.
    """
    spec = build_droid_franka_robotiq_spec(offwidth=offwidth, offheight=offheight)

    spec.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.55, 0.10, table_z - 0.02],
        size=[0.55, 0.70, 0.02],
        rgba=[0.80, 0.64, 0.40, 1.0],
        friction=[0.9, 0.005, 0.0001],
    )

    cup_rb, cup_rt, cup_h = cup_dims
    bowl_rb, bowl_rt, bowl_h = bowl_dims

    cup_body = spec.worldbody.add_body(name="red_cup", pos=[cup_xy[0], cup_xy[1], table_z + 0.001])
    cup_body.add_freejoint(name="red_cup_jnt")
    if hollow:
        _add_hollow_vessel(
            cup_body,
            name_prefix="red_cup",
            radius_bottom=cup_rb,
            radius_top=cup_rt,
            height=cup_h,
            wall_thickness=0.0035,
            rgba=(0.88, 0.14, 0.10, 1.0),
            total_mass=0.040,
            n_sides=16,
            friction=(1.8, 0.02, 0.002),
        )
    else:
        cup_body.add_geom(
            name="red_cup_solid",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.0, 0.0, cup_h / 2.0],
            size=[cup_rt, cup_h / 2.0, 0.0],
            rgba=(0.88, 0.14, 0.10, 1.0),
            mass=0.040,
            friction=(1.8, 0.02, 0.002),
        )

    bowl_body = spec.worldbody.add_body(name="pour_bowl", pos=[bowl_xy[0], bowl_xy[1], table_z + 0.001])
    bowl_body.add_freejoint(name="pour_bowl_jnt")
    if hollow:
        _add_hollow_vessel(
            bowl_body,
            name_prefix="pour_bowl",
            radius_bottom=bowl_rb,
            radius_top=bowl_rt,
            height=bowl_h,
            wall_thickness=0.005,
            rgba=(0.85, 0.87, 0.88, 1.0),
            total_mass=0.35,
            n_sides=16,
            friction=(1.0, 0.01, 0.001),
        )
    else:
        bowl_body.add_geom(
            name="pour_bowl_solid",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.0, 0.0, bowl_h / 2.0],
            size=[bowl_rt, bowl_h / 2.0, 0.0],
            rgba=(0.85, 0.87, 0.88, 1.0),
            mass=0.35,
            friction=(1.0, 0.01, 0.001),
        )

    # Granular beans placed inside the hollow cup (2 layers of 4 spheres)
    if hollow:
        for i in range(n_beans):
            layer = i // 4
            ang = (i % 4) * (math.pi / 2.0) + layer * 0.4
            rad = 0.011
            bx = cup_xy[0] + rad * math.cos(ang)
            by = cup_xy[1] + rad * math.sin(ang)
            bz = table_z + 0.013 + layer * 0.016
            bb = spec.worldbody.add_body(name=f"bean_{i}", pos=[bx, by, bz])
            bb.add_freejoint(name=f"bean_{i}_jnt")
            bb.add_geom(
                name=f"bean_{i}_geom",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.0075, 0.0, 0.0],
                rgba=[0.98, 0.58, 0.08, 1.0],
                mass=0.003,
                friction=[0.6, 0.005, 0.0005],
            )

    model = spec.compile()
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "red_cup" in nm or "pad" in nm:
            model.geom_condim[g] = 4
    return model
