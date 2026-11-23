"""End-to-end tests for Real2Sim-Bench v0.2: anti-cheat gates, exact intrinsics, and Grader."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from robot_descriptions import panda_mj_description

from r2s_bench.backends.mujoco_backend import MuJoCoBackend
from r2s_bench.grading.grader import Real2SimGrader
from r2s_bench.schema import AgentSubmission, CameraCalib


def _make_xml(obj_xyz=(0.52, -0.08, 0.03), dynamic=True, secret_actuator=False) -> str:
    panda_path = Path(panda_mj_description.MJCF_PATH)
    raw = panda_path.read_text().replace('meshdir="assets"', f'meshdir="{panda_path.parent / "assets"}"')
    ox, oy, oz = obj_xyz
    joint_xml = '<freejoint name="cube_jnt"/>' if dynamic else ""
    if secret_actuator:
        joint_xml = '<joint name="cube_slide_x" type="slide" axis="1 0 0"/>'
    extra = f"""
    <light pos="0.5 0.0 1.8" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="eval_cam" pos="1.15 -0.65 0.72" fovy="45"/>
    <geom name="table_top" type="box" pos="0.5 0 -0.02" size="0.45 0.55 0.02" rgba="0.82 0.78 0.72 1"/>
    <body name="target_cube" pos="{ox} {oy} {oz}">
      {joint_xml}
      <geom name="target_cube_geom" type="box" size="0.03 0.03 0.03" mass="0.12" rgba="0.88 0.18 0.18 1"/>
    </body>
  </worldbody>"""
    raw = raw.replace("</worldbody>", extra, 1)
    if secret_actuator:
        raw = raw.replace("</actuator>", '<position name="cheat_act" joint="cube_slide_x" kp="500"/>\n  </actuator>', 1)
    if "<visual>" not in raw:
        raw = raw.replace("<worldbody>", '<visual><global offwidth="640" offheight="480"/></visual>\n  <worldbody>', 1)
    return raw


def _make_calib(pos_offset=(0.0, 0.0, 0.0)) -> CameraCalib:
    cam_pos = np.array([1.15, -0.65, 0.72]) + np.array(pos_offset)
    target = np.array([0.45, 0.0, 0.12])
    z = (target - cam_pos) / np.linalg.norm(target - cam_pos)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = cam_pos
    return CameraCalib(
        name="eval_cam", serial="sim0", width=320, height=240,
        fx=280.0, fy=280.0, cx=168.0, cy=118.0, cam_to_base=T.tolist(),
    )


def _make_traj(n=25):
    q0 = np.array([-0.25, 0.35, 0.0, -2.05, 0.0, 2.40, 0.78])
    q1 = np.array([-0.28, 0.62, 0.0, -2.18, 0.0, 2.80, 0.78])
    q2 = np.array([0.28, 0.62, 0.0, -2.18, 0.0, 2.80, 0.78])
    qs = []
    for i in range(n):
        if i < n // 3:
            a = i / max(1, n // 3 - 1)
            qs.append((1 - a) * q0 + a * q1)
        else:
            a = (i - n // 3) / max(1, n - n // 3 - 1)
            qs.append((1 - a) * q1 + a * q2)
    return np.stack(qs, axis=0), np.zeros(n, dtype=np.float32)


def test_anti_cheat_and_grader_pipeline() -> None:
    calib = _make_calib()
    qpos_seq, grip_seq = _make_traj(n=24)

    # 1. Generate GT reference frames
    gt_backend = MuJoCoBackend(xml_string=_make_xml(), object_bodies=["target_cube"])
    gt_backend.reset(robot_qpos=qpos_seq[0], gripper=0.0)
    gt_rgbs, gt_rmasks, gt_omasks = [], [], []
    for q, g in zip(qpos_seq, grip_seq):
        gt_backend.step_robot_target(q, float(g), dt=1.0 / 15.0)
        rgb, rm, om = gt_backend.render_camera(calib)
        gt_rgbs.append(rgb)
        gt_rmasks.append(rm)
        gt_omasks.append(om)
    gt_backend.close()

    grader = Real2SimGrader()
    sub = AgentSubmission(case_id="unit_test_case", backend="mujoco", scene_file="inline", object_bodies=["target_cube"])

    # 2. Cheat test A: secretly actuated object must be caught
    cheat_b = MuJoCoBackend(xml_string=_make_xml(secret_actuator=True), object_bodies=["target_cube"])
    m_cheat = grader.evaluate_episode(cheat_b, sub, "ep0", calib, qpos_seq, grip_seq, gt_rgbs, gt_rmasks, gt_omasks)
    rep_cheat = grader.summarize("unit_test_case", [m_cheat])
    cheat_b.close()
    assert rep_cheat.failure_category == "cheat_detected", rep_cheat
    assert rep_cheat.aggregate_score == 0.0

    # 3. Cheat test B: static non-dynamic object must be caught
    static_b = MuJoCoBackend(xml_string=_make_xml(dynamic=False), object_bodies=["target_cube"])
    m_static = grader.evaluate_episode(static_b, sub, "ep0", calib, qpos_seq, grip_seq, gt_rgbs, gt_rmasks, gt_omasks)
    rep_static = grader.summarize("unit_test_case", [m_static])
    static_b.close()
    assert rep_static.failure_category == "cheat_detected", rep_static

    # 4. Oracle pass test
    ok_b = MuJoCoBackend(xml_string=_make_xml(), object_bodies=["target_cube"])
    m_ok = grader.evaluate_episode(ok_b, sub, "ep0", calib, qpos_seq, grip_seq, gt_rgbs, gt_rmasks, gt_omasks)
    rep_ok = grader.summarize("unit_test_case", [m_ok])
    ok_b.close()
    assert rep_ok.failure_category == "pass" and rep_ok.aggregate_score > 0.95, rep_ok

    # 5. Misaligned object -> dynamics_fail with much lower score than Oracle
    bad_obj_b = MuJoCoBackend(xml_string=_make_xml(obj_xyz=(0.59, -0.08, 0.03)), object_bodies=["target_cube"])
    m_bad_obj = grader.evaluate_episode(bad_obj_b, sub, "ep0", calib, qpos_seq, grip_seq, gt_rgbs, gt_rmasks, gt_omasks)
    rep_bad_obj = grader.summarize("unit_test_case", [m_bad_obj])
    bad_obj_b.close()
    assert rep_bad_obj.failure_category == "dynamics_fail", rep_bad_obj
    assert rep_bad_obj.aggregate_score < rep_ok.aggregate_score - 0.3

    # 6. Misaligned camera -> calib_fail
    bad_cam_b = MuJoCoBackend(xml_string=_make_xml(), object_bodies=["target_cube"])
    m_bad_cam = grader.evaluate_episode(bad_cam_b, sub, "ep0", _make_calib((0.18, 0.12, 0.06)), qpos_seq, grip_seq, gt_rgbs, gt_rmasks, gt_omasks)
    rep_bad_cam = grader.summarize("unit_test_case", [m_bad_cam])
    bad_cam_b.close()
    assert rep_bad_cam.failure_category == "calib_fail", rep_bad_cam

    print("ALL TESTS PASSED:")
    print(f"  cheat_actuated -> {rep_cheat.failure_category} ({rep_cheat.aggregate_score})")
    print(f"  cheat_static   -> {rep_static.failure_category} ({rep_static.aggregate_score})")
    print(f"  oracle         -> {rep_ok.failure_category} ({rep_ok.aggregate_score})")
    print(f"  bad_object     -> {rep_bad_obj.failure_category} ({rep_bad_obj.aggregate_score})")
    print(f"  bad_camera     -> {rep_bad_cam.failure_category} ({rep_bad_cam.aggregate_score})")


if __name__ == "__main__":
    test_anti_cheat_and_grader_pipeline()
