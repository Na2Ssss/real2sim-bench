"""DROID dataset adapter with strict GT-IoU filtering and fixed-camera clustering.

Pulls individual 720p MP4s and trajectory.h5 files directly from public GCS
(gresearch/robotics/droid_raw/1.0.1) without downloading the 1.7 TB full dataset.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from r2s_bench.adapters.base import BaseDatasetAdapter
from r2s_bench.schema import CameraCalib, CaseBundleSpec, EpisodeSpec


def droid_ext6_to_matrix(ext6: List[float]) -> np.ndarray:
    """Convert DROID [x, y, z, roll, pitch, yaw] into 4x4 T_base_cam (OpenCV frame)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", ext6[3:6]).as_matrix()
    T[:3, 3] = ext6[0:3]
    return T


def _gcs_download(gcs_obj_path: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1024:
        return dst
    url = f"https://storage.googleapis.com/gresearch/{urllib.parse.quote(gcs_obj_path)}"
    urllib.request.urlretrieve(url, dst)
    return dst


class DroidAdapter(BaseDatasetAdapter):
    """Adapter for DROID using KarlP/droid calibration metadata + public GCS raw episodes."""

    def __init__(self, karlp_meta_dir: Path):
        self.meta_dir = Path(karlp_meta_dir)
        self.cam2base: Dict[str, dict] = json.loads((self.meta_dir / "cam2base_extrinsics.json").read_text())
        self.intrinsics: Dict[str, dict] = json.loads((self.meta_dir / "intrinsics.json").read_text())
        lang_path = self.meta_dir / "droid_language_annotations.json"
        self.lang: Dict[str, dict] = json.loads(lang_path.read_text()) if lang_path.exists() else {}

    def discover_clusters(
        self,
        min_iou: float = 0.85,
        pos_bucket_m: float = 0.05,
        rot_bucket_deg: float = 3.0,
        min_episodes: int = 10,
        only_gt_source: bool = True,
    ) -> List[Dict[str, Any]]:
        """Cluster high-IoU success episodes by (lab/date, cam_serial, quantized pose)."""
        buckets: Dict[Tuple[str, str, tuple, tuple], List[Tuple[str, dict]]] = defaultdict(list)

        for eid, entry in self.cam2base.items():
            rpath = entry.get("relative_path", "")
            if "/success/" not in rpath:
                continue
            if only_gt_source and entry.get("source") != "GT":
                continue
            q = float(entry.get("quality_metric") or 0.0)
            if q < min_iou:
                continue
            cs = next((k for k in entry if k.isdigit()), None)
            if not cs or eid not in self.intrinsics or cs not in self.intrinsics[eid]:
                continue

            ext6 = entry[cs]
            pos = np.asarray(ext6[:3], dtype=np.float64)
            rot_deg = np.degrees(np.asarray(ext6[3:6], dtype=np.float64))
            p_bin = tuple(np.round(pos / pos_bucket_m).astype(int).tolist())
            r_bin = tuple(np.round(rot_deg / rot_bucket_deg).astype(int).tolist())
            parts = rpath.split("/")
            lab_date = f"{parts[0]}/{parts[2]}"
            buckets[(lab_date, cs, p_bin, r_bin)].append((eid, entry))

        clusters: List[Dict[str, Any]] = []
        for idx, ((ld, cs, p_bin, r_bin), items) in enumerate(
            sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
        ):
            if len(items) < min_episodes:
                continue
            items.sort(key=lambda x: x[0])
            eids = [x[0] for x in items]
            ious = [float(x[1].get("quality_metric", 0.0)) for x in items]
            sample_instrs = [
                self.lang[e].get("language_instruction1", "")
                for e in eids[:5]
                if e in self.lang and self.lang[e].get("language_instruction1")
            ]
            cluster_id = f"droid__{ld.replace('/', '_')}__cam{cs}__b{idx:03d}"
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "lab_date": ld,
                    "cam_serial": cs,
                    "num_episodes": len(eids),
                    "mean_iou": round(float(np.mean(ious)), 4),
                    "sample_instructions": sample_instrs[:3],
                    "episode_ids": eids,
                }
            )
        return clusters

    def _export_episode(
        self,
        eid: str,
        cam_serial: str,
        split: str,
        bundle_dir: Path,
        provide_extrinsics: bool,
    ) -> EpisodeSpec:
        entry = self.cam2base[eid]
        rpath = entry["relative_path"]
        intr = self.intrinsics[eid][cam_serial]
        T_base_cam = droid_ext6_to_matrix(entry[cam_serial])
        fx, cx, fy, cy = intr["cameraMatrix"]

        sub_dir = "public" if split == "sysid_public" else "hidden"
        ep_rel_dir = Path(sub_dir) / "episodes" / eid
        ep_abs_dir = bundle_dir / ep_rel_dir
        ep_abs_dir.mkdir(parents=True, exist_ok=True)

        # Download 720p calibrated camera MP4 and trajectory.h5
        mp4_rel = ep_rel_dir / f"cam_{cam_serial}.mp4"
        _gcs_download(f"robotics/droid_raw/1.0.1/{rpath}/recordings/MP4/{cam_serial}.mp4", bundle_dir / mp4_rel)

        h5_tmp = ep_abs_dir / "raw_trajectory.h5"
        _gcs_download(f"robotics/droid_raw/1.0.1/{rpath}/trajectory.h5", h5_tmp)

        with h5py.File(h5_tmp, "r") as f:
            qpos = f["observation/robot_state/joint_positions"][:].astype(np.float32)
            gripper = f["observation/robot_state/gripper_position"][:].astype(np.float32)
            cartesian = f["observation/robot_state/cartesian_position"][:].astype(np.float32)
            torques = f["observation/robot_state/motor_torques_measured"][:].astype(np.float32)

        npz_rel = ep_rel_dir / "trajectory.npz"
        np.savez_compressed(
            bundle_dir / npz_rel,
            qpos=qpos,
            gripper=gripper,
            cartesian=cartesian,
            motor_torques=torques,
            fps=np.array(15.0, dtype=np.float32),
        )
        h5_tmp.unlink(missing_ok=True)

        # Always save frame 0 into public/init_frames/ so Agent knows object initial pose for Track B reset
        cap = cv2.VideoCapture(str(bundle_dir / mp4_rel))
        ok, frame0 = cap.read()
        cap.release()
        if ok:
            init_dir = bundle_dir / "public" / "init_frames"
            init_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(init_dir / f"{eid}_cam_{cam_serial}_frame0.png"), frame0)

        cam_calib = CameraCalib(
            name=f"ext_{cam_serial}",
            serial=cam_serial,
            width=int(intr["width"]),
            height=int(intr["height"]),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            cam_to_base=T_base_cam.tolist(),
            quality_iou=float(entry.get("quality_metric", 0.0)),
        )

        lang_str = self.lang.get(eid, {}).get("language_instruction1", "")
        return EpisodeSpec(
            episode_id=eid,
            source_dataset="droid",
            robot_type="franka_robotiq_2f85",
            fps=15.0,
            num_steps=int(len(qpos)),
            language_instruction=lang_str,
            split=split,
            cameras=[cam_calib],
            video_paths={cam_calib.name: str(mp4_rel)},
            trajectory_path=str(npz_rel),
        )

    def build_case_bundle(
        self,
        cluster_id: str,
        out_dir: Path,
        n_public: int = 1,
        n_heldout: int = 2,
        tier: str = "tier1_rigid",
        provide_extrinsics: bool = True,
        cluster_record: Optional[Dict[str, Any]] = None,
    ) -> CaseBundleSpec:
        if cluster_record is None:
            clusters = self.discover_clusters(min_episodes=n_public + n_heldout)
            match = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
            if match is None:
                raise ValueError(f"Cluster {cluster_id} not found.")
            cluster_record = match

        eids = cluster_record["episode_ids"]
        cs = cluster_record["cam_serial"]
        total_need = n_public + n_heldout
        if len(eids) < total_need:
            raise ValueError(f"Cluster {cluster_id} has {len(eids)} episodes, need {total_need}.")

        # Evenly sample across the session so object poses vary cleanly
        idxs = np.linspace(0, len(eids) - 1, total_need, dtype=int).tolist()
        chosen = [eids[i] for i in idxs]
        pub_eids = chosen[:n_public]
        hid_eids = chosen[n_public:]

        case_dir = Path(out_dir) / cluster_id
        case_dir.mkdir(parents=True, exist_ok=True)

        public_specs = [
            self._export_episode(e, cs, "sysid_public", case_dir, provide_extrinsics) for e in pub_eids
        ]
        heldout_specs = [
            self._export_episode(e, cs, "heldout_eval", case_dir, provide_extrinsics) for e in hid_eids
        ]

        bundle_spec = CaseBundleSpec(
            case_id=cluster_id,
            tier=tier,
            track="track_b_sysid_multipose" if n_public > 0 else "track_a_static",
            source_dataset="droid",
            robot_type="franka_robotiq_2f85",
            scene_cluster_id=cluster_id,
            public_episodes=public_specs,
            heldout_episodes=heldout_specs,
            provide_extrinsics=provide_extrinsics,
        )
        # Save full spec in hidden/ for the Grader, and sanitized public spec in public/
        bundle_spec.to_json(case_dir / "hidden" / "case_bundle.json")

        public_view = {
            "case_id": bundle_spec.case_id,
            "tier": bundle_spec.tier,
            "track": bundle_spec.track,
            "robot_type": bundle_spec.robot_type,
            "provide_extrinsics": provide_extrinsics,
            "sysid_episodes": [
                {
                    "episode_id": ep.episode_id,
                    "fps": ep.fps,
                    "num_steps": ep.num_steps,
                    "language_instruction": ep.language_instruction,
                    "video_paths": ep.video_paths,
                    "trajectory_path": ep.trajectory_path,
                    "cameras": [
                        {
                            "name": c.name,
                            "width": c.width,
                            "height": c.height,
                            "fx": c.fx,
                            "fy": c.fy,
                            "cx": c.cx,
                            "cy": c.cy,
                            **({"cam_to_base": c.cam_to_base} if provide_extrinsics else {}),
                        }
                        for c in ep.cameras
                    ],
                }
                for ep in public_specs
            ],
            "eval_init_frames": [
                {
                    "episode_id": ep.episode_id,
                    "language_instruction": ep.language_instruction,
                    "init_frame": f"public/init_frames/{ep.episode_id}_cam_{cs}_frame0.png",
                }
                for ep in heldout_specs
            ],
        }
        (case_dir / "public" / "task_input.json").write_text(
            json.dumps(public_view, indent=2, ensure_ascii=False)
        )
        return bundle_spec
