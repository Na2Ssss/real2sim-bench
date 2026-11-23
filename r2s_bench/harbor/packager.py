"""Harbor closed-book task packager for Real2Sim-Bench.

Generates a Harbor-compatible task directory with strict separation between:
- task/public_input/  (mounted into Agent container)
- verifier/hidden/    (never mounted into Agent container; accessed only by Grader)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from r2s_bench.schema import CaseBundleSpec


INSTRUCTION_TEMPLATE = """# Real2Sim-Bench Task: `{case_id}`

## Objective
Reconstruct an executable, physically accurate simulation environment for the real-world robot workspace shown in `/workspace/input/`.

- **Track**: `{track}`
- **Difficulty Tier**: `{tier}`
- **Robot Platform**: `{robot_type}`
- **Camera Extrinsics Provided**: `{provide_extrinsics}`

## Public Inputs (`/workspace/input/`)
1. `task_input.json`: Camera intrinsics{ext_note}, SysID reference demonstrations (Track B), and initial frames for held-out evaluation episodes.
2. `episodes/`: Reference real-robot 720p videos and measured joint trajectories (`trajectory.npz`).
3. `init_frames/`: Frame-0 RGB images for each held-out evaluation episode.

## Required Submission (`/workspace/submission/`)
Create `submission.json` conforming to `AgentSubmission`:
```json
{{
  "case_id": "{case_id}",
  "backend": "mujoco",
  "scene_file": "/workspace/submission/scene.xml",
  "object_bodies": ["<interactive_body_name_1>"]
}}
```

## Anti-Cheat Rules (Strictly Enforced by Independent Verifier)
1. Do NOT modify the official robot arm URDF/MJCF collision geometry.
2. Every interactive object in `object_bodies` MUST have genuine dynamic joints (`<freejoint>` or `<hinge>`/`<slide>`) and positive mass; secretly actuating or kinematically baking object trajectories results in immediate disqualification (`cheat_detected`).
3. Your scene will be evaluated on **held-out real-robot teleoperation replays** across resting stability, L1 camera/robot mask alignment, and L2 dynamic object tracking.
"""


def package_harbor_task(case_dir: Path, harbor_out_dir: Path) -> Path:
    """Package a built CaseBundle into a closed-book Harbor benchmark task."""
    spec = CaseBundleSpec.from_json(case_dir / "hidden" / "case_bundle.json")
    task_root = harbor_out_dir / spec.case_id
    pub_dst = task_root / "task" / "public_input"
    ver_dst = task_root / "verifier" / "hidden"

    if pub_dst.exists():
        shutil.rmtree(pub_dst)
    if ver_dst.exists():
        shutil.rmtree(ver_dst)

    shutil.copytree(case_dir / "public", pub_dst)
    shutil.copytree(case_dir / "hidden", ver_dst)

    ext_note = " and calibrated `cam_to_base` extrinsics" if spec.provide_extrinsics else " (UNCALIBRATED: you must estimate `cam_to_base`)"
    instruction = INSTRUCTION_TEMPLATE.format(
        case_id=spec.case_id,
        track=spec.track,
        tier=spec.tier,
        robot_type=spec.robot_type,
        provide_extrinsics=spec.provide_extrinsics,
        ext_note=ext_note,
    )
    (task_root / "task" / "instruction.md").write_text(instruction)

    task_toml = f'''version = "1.0"

[metadata]
case_id = "{spec.case_id}"
tier = "{spec.tier}"
track = "{spec.track}"
source_dataset = "{spec.source_dataset}"
robot_type = "{spec.robot_type}"

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 1800.0
'''
    (task_root / "task" / "task.toml").write_text(task_toml)

    # Closed-book Harbor launcher stub (mirrors g1d-real2sim-bench-v1/harbor_run.py)
    launcher = '''"""Closed-book Harbor launcher ensuring /logs/verifier and hidden/ are unmounted from Agent."""
import os
from pathlib import Path

def filter_agent_mounts(original_mounts):
    blocked = {"/logs/verifier", "/verifier"}
    return [m for m in original_mounts if m.get("target") not in blocked]
'''
    (task_root / "harbor_closed_book.py").write_text(launcher)

    manifest = {
        "case_id": spec.case_id,
        "protocol": "r2s-bench-v0.2",
        "num_public_episodes": len(spec.public_episodes),
        "num_heldout_episodes": len(spec.heldout_episodes),
        "provide_extrinsics": spec.provide_extrinsics,
    }
    (task_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return task_root
