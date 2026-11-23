"""Unit tests for clean outcome-based task evaluation and cross-scene transfer primitives."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from r2s_bench.eval.task_success import (
    RolloutMonitor,
    evaluate_block_in_drawer,
    evaluate_cup_in_bowl,
    evaluate_joint_state,
    evaluate_object_on_target,
    evaluate_object_out_of_container,
    evaluate_pouring_beans,
)
from r2s_bench.pipeline.scene_transfer import (
    PRESET_SCENES,
    SceneTransferConfig,
    evaluate_scene_outcome,
    select_camera_extrinsics,
    triangulate_two_views,
)
from r2s_bench.schema import CameraCalib


def _make_test_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml = """
    <mujoco>
      <worldbody>
        <body name="black_bowl" pos="0.5 0.0 0.0"><geom type="box" size="0.06 0.06 0.02"/></body>
        <body name="orange_cup" pos="0.52 0.01 0.02"><freejoint/><geom type="cylinder" size="0.03 0.04"/></body>
        <body name="drawer_box" pos="0.6 0.0 -0.10">
          <joint name="drawer_slide" type="slide" axis="1 0 0" range="-0.25 0.05"/>
          <geom type="box" size="0.15 0.20 0.02"/>
        </body>
        <body name="blue_block" pos="0.6 0.0 -0.07"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>
        <body name="target_bowl" pos="0.4 0.2 0.0"><geom type="box" size="0.06 0.06 0.02"/></body>
        <body name="bean_0" pos="0.41 0.21 0.02"><freejoint/><geom type="sphere" size="0.008"/></body>
        <body name="bean_1" pos="0.10 0.20 0.02"><freejoint/><geom type="sphere" size="0.008"/></body>
        <body name="marker" pos="0.8 0.0 0.02"><freejoint/><geom type="cylinder" size="0.008 0.06"/></body>
        <body name="pot_lid" pos="0.5 0.0 0.05"><freejoint/><geom type="cylinder" size="0.06 0.01"/></body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_outcome_predicates_and_explosion_guard() -> None:
    m, d = _make_test_model()
    assert evaluate_cup_in_bowl(m, d, table_z=-0.02).success
    assert evaluate_block_in_drawer(m, d, drawer_q_init=-0.20, drawer_q_closed=0.0, table_z=-0.02).success
    assert evaluate_pouring_beans(m, d, table_z=-0.02).success
    assert evaluate_object_out_of_container(m, d, None, "marker", "black_bowl").success
    assert evaluate_joint_state(m, d, None, "drawer_slide", q_init=-0.20, q_target=0.0).success
    assert evaluate_object_on_target(m, d, None, "pot_lid", "black_bowl").success

    # Half-open drawer fails with DRAWER_NOT_CLOSED
    sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    d.qpos[int(m.jnt_qposadr[sj])] = -0.14
    mujoco.mj_forward(m, d)
    r_open = evaluate_block_in_drawer(m, d, drawer_q_init=-0.20, drawer_q_closed=0.0, table_z=-0.02)
    assert not r_open.success and r_open.failure_reason == "DRAWER_NOT_CLOSED"

    # Physics explosion (NaN / extreme velocity) must be rejected
    mon = RolloutMonitor(m, "orange_cup")
    mon.reset(d)
    mon.max_speed = 50.0
    r_exp = evaluate_cup_in_bowl(m, d, mon, table_z=-0.02)
    assert not r_exp.success and r_exp.failure_reason == "PHYSICS_EXPLOSION"


def test_scene_transfer_primitives_and_configs() -> None:
    m, d = _make_test_model()
    assert evaluate_scene_outcome(PRESET_SCENES["demo1_cup_in_bowl"], m, d, overrides={"table_z": -0.02}).success

    # Tripod-shift guard (>4cm shift rejects global refined extrinsic)
    t_ref = np.eye(4)
    t_moved = np.eye(4)
    t_moved[:3, 3] = [0.20, 0.0, 0.0]
    _, used_ref = select_camera_extrinsics(t_moved, t_ref, max_shift_m=0.04)
    assert not used_ref

    # Exact dual-view triangulation
    t1, t2 = np.eye(4), np.eye(4)
    t2[:3, 3] = [1.0, 0.0, 0.0]
    c1 = CameraCalib("c1", "1", 640, 480, 500, 500, 320, 240, t1.tolist())
    c2 = CameraCalib("c2", "2", 640, 480, 500, 500, 320, 240, t2.tolist())
    pt_gt = np.array([0.4, 0.2, 2.0])
    uv1 = (500 * pt_gt[0] / pt_gt[2] + 320, 500 * pt_gt[1] / pt_gt[2] + 240)
    uv2 = (500 * (pt_gt[0] - 1.0) / pt_gt[2] + 320, 500 * pt_gt[1] / pt_gt[2] + 240)
    pt_est, miss = triangulate_two_views(c1, uv1, c2, uv2)
    assert np.linalg.norm(pt_est - pt_gt) < 1e-6 and miss < 1e-6

    # Preset JSON round-trip & declarative scene configs valid JSON
    with tempfile.TemporaryDirectory() as td:
        for sid, cfg in PRESET_SCENES.items():
            p = Path(td) / f"{sid}.json"
            cfg.to_json(p)
            assert SceneTransferConfig.from_json(p).scene_id == sid

    root = Path(__file__).resolve().parents[1]
    scene_jsons = sorted((root / "configs" / "scenes").glob("*.json"))
    assert len(scene_jsons) >= 4
    for sp in scene_jsons:
        raw = json.loads(sp.read_text())
        assert "scene_id" in raw and len(raw["objects"]) == 2 and "outcome" in raw


if __name__ == "__main__":
    test_outcome_predicates_and_explosion_guard()
    test_scene_transfer_primitives_and_configs()
    print("ALL TESTS PASSED!")
