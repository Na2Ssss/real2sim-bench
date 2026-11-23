"""Outcome-Based Task-Success Evaluator for Real-to-Sim MuJoCo Rollouts.

Design principle (matching SIMPLER / ManiSkill / LIBERO):
  Each task has ONE clean end-state goal predicate + basic numerical sanity (no NaN / physics explosion).
  Process telemetry (robot contact, max lift, uprightness, end speed) is still recorded in `metrics`
  for diagnostics, but does NOT impose brittle process gates that reject genuine human teleop successes.

Built-in atomic predicates (reusable across new scenes):
  - `evaluate_object_in_container`     (e.g., cup in bowl, can in bowl, object in bin)
  - `evaluate_object_out_of_container` (e.g., remove marker from cup)
  - `evaluate_joint_state`             (e.g., open/close drawer, door, lid, switch)
  - `evaluate_object_on_target`        (e.g., put lid on pot, place tool on tray)
  - `evaluate_particles_in_target`     (e.g., pour beans/granules into bowl)

Task-specific wrappers (backward compatible with Demo #1/#2/#3):
  - `evaluate_cup_in_bowl`
  - `evaluate_block_in_drawer`
  - `evaluate_pouring_beans`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Set

import mujoco
import numpy as np


@dataclass
class TaskEvalResult:
    """Standardized output of Task-Success Evaluation for a single rollout."""

    task: str
    success: bool
    failure_reason: str  # "OK", "GOAL_NOT_REACHED", "PHYSICS_EXPLOSION", or specific sub-reason
    goal_reached: bool
    robot_contacted: bool
    pre_contact_clean: bool
    stable_end_state: bool
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "success": bool(self.success),
            "failure_reason": self.failure_reason,
            "goal_reached": bool(self.goal_reached),
            "robot_contacted": bool(self.robot_contacted),
            "pre_contact_clean": bool(self.pre_contact_clean),
            "stable_end_state": bool(self.stable_end_state),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
        }


def _collect_descendant_geoms(model: mujoco.MjModel, root_body_names: Sequence[str]) -> Set[int]:
    """Return all geom IDs belonging to any body in `root_body_names` or its descendants."""
    root_ids = set()
    for name in root_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            root_ids.add(bid)

    selected_bodies = set(root_ids)
    changed = True
    while changed:
        changed = False
        for bid in range(1, model.nbody):
            if bid not in selected_bodies and int(model.body_parentid[bid]) in selected_bodies:
                selected_bodies.add(bid)
                changed = True

    return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in selected_bodies}


def _detect_robot_geoms(model: mujoco.MjModel) -> Set[int]:
    """Automatically collect all robot & gripper geom IDs from standard Franka/Robotiq/Panda names."""
    robot_roots = [
        "link0", "panda_link0", "base", "base_mount", "hand",
        "robotiq_base", "left_driver", "right_driver",
    ]
    gids = _collect_descendant_geoms(model, robot_roots)
    if gids:
        return gids
    keywords = ("panda", "link", "hand", "finger", "pad", "driver", "coupler", "follower")
    for g in range(model.ngeom):
        bname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").lower()
        if any(k in bname for k in keywords):
            gids.add(g)
    return gids


class RolloutMonitor:
    """Lightweight per-step MuJoCo numerical-sanity & telemetry monitor (<1ms overhead per episode)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        target_body_name: str,
        robot_geoms: Optional[Set[int]] = None,
    ) -> None:
        self.model = model
        self.target_body_name = target_body_name
        self.target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
        if self.target_bid < 0:
            raise ValueError(f"Body '{target_body_name}' not found in MuJoCo model.")

        self.target_geoms = _collect_descendant_geoms(model, [target_body_name])
        self.robot_geoms = robot_geoms if robot_geoms is not None else _detect_robot_geoms(model)

        self.step_count: int = 0
        self.init_pos: Optional[np.ndarray] = None
        self.first_contact_step: Optional[int] = None
        self.contact_steps: int = 0
        self.max_pre_contact_disp: float = 0.0
        self.max_lift: float = 0.0
        self.max_speed: float = 0.0
        self.has_nan: bool = False

    def reset(self, data: mujoco.MjData) -> None:
        """Call right after initial settling (before policy/trajectory execution)."""
        self.step_count = 0
        self.init_pos = data.xpos[self.target_bid].copy()
        self.first_contact_step = None
        self.contact_steps = 0
        self.max_pre_contact_disp = 0.0
        self.max_lift = 0.0
        self.max_speed = 0.0
        self.has_nan = False

    def observe(self, data: mujoco.MjData) -> None:
        """Call once per control step (or every simulation substep) during rollout."""
        if self.init_pos is None:
            self.reset(data)

        pos = data.xpos[self.target_bid]
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(data.qpos)):
            self.has_nan = True
            return

        in_contact = False
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.robot_geoms and g2 in self.target_geoms) or (
                g2 in self.robot_geoms and g1 in self.target_geoms
            ):
                in_contact = True
                break

        if in_contact:
            self.contact_steps += 1
            if self.first_contact_step is None:
                self.first_contact_step = self.step_count

        disp = float(np.linalg.norm(pos - self.init_pos))
        if self.first_contact_step is None:
            self.max_pre_contact_disp = max(self.max_pre_contact_disp, disp)

        self.max_lift = max(self.max_lift, float(pos[2] - self.init_pos[2]))

        jnt_adr = int(self.model.body_jntadr[self.target_bid])
        if jnt_adr >= 0:
            dof_adr = int(self.model.jnt_dofadr[jnt_adr])
            speed = float(np.linalg.norm(data.qvel[dof_adr : dof_adr + 3]))
            self.max_speed = max(self.max_speed, speed)

        self.step_count += 1

    def end_speed(self, data: mujoco.MjData) -> float:
        jnt_adr = int(self.model.body_jntadr[self.target_bid])
        if jnt_adr < 0:
            return 0.0
        dof_adr = int(self.model.jnt_dofadr[jnt_adr])
        return float(np.linalg.norm(data.qvel[dof_adr : dof_adr + 3]))


def _finalize(
    task: str,
    monitor: Optional[RolloutMonitor],
    data: mujoco.MjData,
    goal_reached: bool,
    extra_metrics: Dict[str, float],
    max_speed_explosion: float = 25.0,
) -> TaskEvalResult:
    """Outcome-only finalizer: PASS iff `goal_reached` and simulation did not explode/NaN."""
    has_nan = (not np.all(np.isfinite(data.qpos))) or (monitor.has_nan if monitor is not None else False)
    max_spd = monitor.max_speed if monitor is not None else 0.0
    exploded = has_nan or (max_spd >= max_speed_explosion)

    if exploded:
        reason = "PHYSICS_EXPLOSION"
    elif not goal_reached:
        reason = "GOAL_NOT_REACHED"
    else:
        reason = "OK"

    # Diagnostic telemetry (logged only — never gates success)
    robot_contacted = (monitor.contact_steps >= 1) if monitor is not None else True
    pre_contact_clean = (monitor.max_pre_contact_disp <= 0.05) if monitor is not None else True
    end_spd = monitor.end_speed(data) if monitor is not None else 0.0
    stable_end = (not exploded) and (end_spd <= 0.25)

    metrics = {
        "contact_steps": float(monitor.contact_steps) if monitor is not None else 0.0,
        "first_contact_step": float(monitor.first_contact_step if (monitor is not None and monitor.first_contact_step is not None) else -1),
        "pre_contact_disp_cm": float(monitor.max_pre_contact_disp * 100.0) if monitor is not None else 0.0,
        "max_lift_cm": float(monitor.max_lift * 100.0) if monitor is not None else 0.0,
        "end_speed_mps": float(end_spd),
        **extra_metrics,
    }
    return TaskEvalResult(
        task=task,
        success=(reason == "OK"),
        failure_reason=reason,
        goal_reached=goal_reached,
        robot_contacted=robot_contacted,
        pre_contact_clean=pre_contact_clean,
        stable_end_state=stable_end,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Generic Atomic Outcome Predicates (reusable across any new Real2Sim scene)
# ---------------------------------------------------------------------------


def evaluate_object_in_container(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    monitor: Optional[RolloutMonitor],
    object_body: str,
    container_body: str,
    max_xy_dist_m: float = 0.085,
    z_range_rel: Sequence[float] = (-0.015, 0.085),
    table_z: float = -0.04,
    task_name: str = "object_in_container",
) -> TaskEvalResult:
    """Atomic predicate: target object ends inside container XY rim & Z bounds."""
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    cnt_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, container_body)
    obj_pos = data.xpos[obj_bid]
    cnt_pos = data.xpos[cnt_bid]

    xy_dist = float(np.linalg.norm(obj_pos[:2] - cnt_pos[:2]))
    dz = float(obj_pos[2] - cnt_pos[2])
    above_table = float(obj_pos[2]) >= (table_z - 0.015)
    seated_z = (float(z_range_rel[0]) <= dz <= float(z_range_rel[1])) and above_table
    upright_cos = float(data.xmat[obj_bid].reshape(3, 3)[2, 2])

    goal_reached = (xy_dist <= max_xy_dist_m) and seated_z
    return _finalize(
        task=task_name,
        monitor=monitor,
        data=data,
        goal_reached=goal_reached,
        extra_metrics={
            "end_xy_dist_cm": xy_dist * 100.0,
            "end_dz_cm": dz * 100.0,
            "cup_upright_cos": upright_cos,
        },
    )


def evaluate_object_out_of_container(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    monitor: Optional[RolloutMonitor],
    object_body: str,
    container_body: str,
    min_xy_dist_m: float = 0.095,
    min_z_above_container_m: float = 0.12,
    task_name: str = "object_out_of_container",
) -> TaskEvalResult:
    """Atomic predicate: object has been removed outside container (XY outside rim OR lifted clear above)."""
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    cnt_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, container_body)
    obj_pos = data.xpos[obj_bid]
    cnt_pos = data.xpos[cnt_bid]

    xy_dist = float(np.linalg.norm(obj_pos[:2] - cnt_pos[:2]))
    dz = float(obj_pos[2] - cnt_pos[2])
    goal_reached = (xy_dist >= min_xy_dist_m) or (dz >= min_z_above_container_m)
    return _finalize(
        task=task_name,
        monitor=monitor,
        data=data,
        goal_reached=goal_reached,
        extra_metrics={"end_xy_dist_cm": xy_dist * 100.0, "end_dz_cm": dz * 100.0},
    )


def evaluate_joint_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    monitor: Optional[RolloutMonitor],
    joint_name: str,
    q_init: float,
    q_target: float,
    max_residual: float = 0.035,
    min_progress_ratio: float = 0.80,
    task_name: str = "joint_state",
) -> TaskEvalResult:
    """Atomic predicate: 1-DoF articulated joint (drawer/door/lid/switch) reached `q_target`."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    q_end = float(data.qpos[int(model.jnt_qposadr[jid])]) if jid >= 0 else q_init
    full_stroke = abs(q_target - q_init)
    sign = 1.0 if q_target >= q_init else -1.0
    moved = sign * (q_end - q_init)
    residual = max(0.0, full_stroke - moved)
    ratio = float(np.clip(moved / max(full_stroke, 1e-4), 0.0, 1.0))
    goal_reached = (residual <= max_residual) or (ratio >= min_progress_ratio)
    return _finalize(
        task=task_name,
        monitor=monitor,
        data=data,
        goal_reached=goal_reached,
        extra_metrics={
            "joint_q_end": q_end,
            "joint_moved_cm": moved * 100.0,
            "joint_residual_cm": residual * 100.0,
            "joint_progress_ratio": ratio,
        },
    )


def evaluate_object_on_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    monitor: Optional[RolloutMonitor],
    object_body: str,
    target_body: str,
    max_xy_dist_m: float = 0.09,
    z_range_rel: Sequence[float] = (0.0, 0.12),
    task_name: str = "object_on_target",
) -> TaskEvalResult:
    """Atomic predicate: object rests on top of target (e.g., lid on pot, tool on tray/plate)."""
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    tgt_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
    obj_pos = data.xpos[obj_bid]
    tgt_pos = data.xpos[tgt_bid]

    xy_dist = float(np.linalg.norm(obj_pos[:2] - tgt_pos[:2]))
    dz = float(obj_pos[2] - tgt_pos[2])
    goal_reached = (xy_dist <= max_xy_dist_m) and (float(z_range_rel[0]) <= dz <= float(z_range_rel[1]))
    return _finalize(
        task=task_name,
        monitor=monitor,
        data=data,
        goal_reached=goal_reached,
        extra_metrics={"end_xy_dist_cm": xy_dist * 100.0, "end_dz_cm": dz * 100.0},
    )


def evaluate_particles_in_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    monitor: Optional[RolloutMonitor],
    target_body: str = "target_bowl",
    particle_prefix: str = "bean_",
    target_radius_m: float = 0.095,
    min_particles_in: int = 1,
    min_ratio_in: float = 0.0,
    table_z: float = -0.04,
    task_name: str = "particles_in_target",
) -> TaskEvalResult:
    """Atomic predicate: granular particles poured into target container."""
    tgt_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
    tgt_pos = data.xpos[tgt_bid]

    p_bids = [
        b for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(particle_prefix)
    ]
    n_total = len(p_bids)
    n_in = 0
    for bid in p_bids:
        p = data.xpos[bid]
        d_xy = float(np.linalg.norm(p[:2] - tgt_pos[:2]))
        if d_xy <= target_radius_m and (table_z - 0.01 <= float(p[2]) <= float(tgt_pos[2]) + 0.08):
            n_in += 1

    ratio = (n_in / n_total) if n_total > 0 else 0.0
    goal_reached = (n_in >= min_particles_in) and (ratio >= min_ratio_in)
    return _finalize(
        task=task_name,
        monitor=monitor,
        data=data,
        goal_reached=goal_reached,
        extra_metrics={
            "beans_in_bowl": float(n_in),
            "beans_total": float(n_total),
            "beans_ratio": float(ratio),
        },
    )


# ---------------------------------------------------------------------------
# Task Wrappers for Demo #1 / #2 / #3 (backward compatible signatures)
# ---------------------------------------------------------------------------


def evaluate_cup_in_bowl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    monitor: Optional[RolloutMonitor] = None,
    cup_body: str = "orange_cup",
    bowl_body: str = "black_bowl",
    max_xy_dist_m: float = 0.085,
    min_lift_m: float = 0.0,  # kept for signature compatibility (not enforced)
    table_z: float = -0.04,
) -> TaskEvalResult:
    """Demo #1 Outcome Criterion: orange cup ends inside black bowl."""
    del min_lift_m
    return evaluate_object_in_container(
        model=model,
        data=data,
        monitor=monitor,
        object_body=cup_body,
        container_body=bowl_body,
        max_xy_dist_m=max_xy_dist_m,
        z_range_rel=(-0.015, 0.085),
        table_z=table_z,
        task_name="cup_in_bowl",
    )


def evaluate_block_in_drawer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    monitor: Optional[RolloutMonitor] = None,
    block_body: str = "blue_block",
    drawer_body: str = "drawer_box",
    drawer_joint: str = "drawer_slide",
    drawer_yaw_rad: float = 0.0,
    drawer_q_init: float = 0.0,
    drawer_q_closed: float = 0.20,
    max_residual_open_m: float = 0.035,
    min_close_ratio: float = 0.80,
    min_lift_m: float = 0.0,  # kept for signature compatibility (not enforced)
    drawer_half_extents: Sequence[float] = (0.15, 0.20),
    table_z: float = -0.04,
) -> TaskEvalResult:
    """Demo #2 Outcome Criterion: blue block is inside drawer AND drawer is closed."""
    del min_lift_m
    block_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block_body)
    drawer_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, drawer_body)
    bpos = data.xpos[block_bid]
    dpos = data.xpos[drawer_bid]

    rel = bpos - dpos
    c_y, s_y = float(np.cos(-drawer_yaw_rad)), float(np.sin(-drawer_yaw_rad))
    lx = c_y * float(rel[0]) - s_y * float(rel[1])
    ly = s_y * float(rel[0]) + c_y * float(rel[1])
    inside_xy = (abs(lx) <= drawer_half_extents[0]) and (abs(ly) <= drawer_half_extents[1])
    inside_z = (float(bpos[2]) < table_z - 0.015) and (float(bpos[2]) > dpos[2] - 0.05)
    block_in_drawer = inside_xy and inside_z

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, drawer_joint)
    drawer_q_end = float(data.qpos[int(model.jnt_qposadr[jid])]) if jid >= 0 else drawer_q_init
    full_stroke = abs(drawer_q_closed - drawer_q_init)
    sign = 1.0 if drawer_q_closed >= drawer_q_init else -1.0
    pushed = sign * (drawer_q_end - drawer_q_init)
    residual_open = max(0.0, full_stroke - pushed)
    close_ratio = float(np.clip(pushed / max(full_stroke, 1e-4), 0.0, 1.0))
    drawer_closed = (residual_open <= max_residual_open_m) or (close_ratio >= min_close_ratio)

    goal_reached = block_in_drawer and drawer_closed
    res = _finalize(
        task="block_in_drawer",
        monitor=monitor,
        data=data,
        goal_reached=goal_reached,
        extra_metrics={
            "block_in_drawer": float(block_in_drawer),
            "drawer_closed": float(drawer_closed),
            "drawer_pushed_cm": pushed * 100.0,
            "drawer_residual_open_cm": residual_open * 100.0,
            "drawer_close_ratio": close_ratio,
            "rel_x_cm": float(lx * 100.0),
            "rel_y_cm": float(ly * 100.0),
            "rel_z_cm": float(rel[2] * 100.0),
        },
    )
    if res.failure_reason == "GOAL_NOT_REACHED":
        if block_in_drawer and not drawer_closed:
            res.failure_reason = "DRAWER_NOT_CLOSED"
        elif not block_in_drawer and drawer_closed:
            res.failure_reason = "BLOCK_NOT_IN_DRAWER"
    return res


def evaluate_pouring_beans(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    monitor: Optional[RolloutMonitor] = None,
    bowl_body: str = "target_bowl",
    bean_prefix: str = "bean_",
    bowl_radius_m: float = 0.095,
    min_ratio_in_bowl: float = 0.0,
    min_beans_in_bowl: int = 1,
    min_lift_m: float = 0.0,  # kept for signature compatibility (not enforced)
    table_z: float = -0.04,
) -> TaskEvalResult:
    """Demo #3 Outcome Criterion: beans poured into target bowl (>= min_beans_in_bowl)."""
    del min_lift_m
    return evaluate_particles_in_target(
        model=model,
        data=data,
        monitor=monitor,
        target_body=bowl_body,
        particle_prefix=bean_prefix,
        target_radius_m=bowl_radius_m,
        min_particles_in=min_beans_in_bowl,
        min_ratio_in=min_ratio_in_bowl,
        table_z=table_z,
        task_name="pouring_beans",
    )
