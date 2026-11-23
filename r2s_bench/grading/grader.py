"""Closed-book Real2Sim Grader engine with anti-cheat gates and failure taxonomy."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from r2s_bench.backends.base import BaseSimBackend
from r2s_bench.grading.l1_visual import evaluate_l1_frame
from r2s_bench.grading.l2_physics import evaluate_l2_sequence
from r2s_bench.schema import AgentSubmission, CameraCalib, EpisodeEvalMetrics, EvalReport


class Real2SimGrader:
    """Evaluates an AgentSubmission against hidden real-robot teleoperation rollouts."""

    def __init__(
        self,
        max_resting_drift_m: float = 0.015,
        max_penetration_m: float = 0.01,
        min_l1_robot_iou: float = 0.70,
    ):
        self.max_resting_drift_m = max_resting_drift_m
        self.max_penetration_m = max_penetration_m
        self.min_l1_robot_iou = min_l1_robot_iou

    def evaluate_episode(
        self,
        backend: BaseSimBackend,
        submission: AgentSubmission,
        episode_id: str,
        calib: CameraCalib,
        qpos_seq: np.ndarray,
        gripper_seq: np.ndarray,
        real_rgbs: List[np.ndarray],
        real_robot_masks: List[np.ndarray],
        real_obj_masks: Optional[List[np.ndarray]] = None,
        fps: float = 15.0,
    ) -> EpisodeEvalMetrics:
        # Gate 0: Static anti-cheat checks (no baked animation / no secretly actuated objects)
        violations = backend.inspect_anti_cheat(submission.object_bodies)
        if violations:
            return EpisodeEvalMetrics(
                episode_id=episode_id,
                anti_cheat_passed=False,
                anti_cheat_violations=violations,
                resting_max_drift_m=float("inf"),
                resting_max_penetration_m=float("inf"),
                l1_robot_mask_iou=0.0,
                l1_fg_psnr_db=0.0,
            )

        # Gate 1: Reset & resting stability check (no floating, exploding, or severe penetration)
        backend.reset(robot_qpos=qpos_seq[0], gripper=float(gripper_seq[0]))
        stab = backend.check_resting_stability(duration_sec=0.5, object_bodies=submission.object_bodies)

        # Clean reset back to exact t=0 before starting teleop trajectory replay
        backend.reset(robot_qpos=qpos_seq[0], gripper=float(gripper_seq[0]))

        # Open-loop replay of measured real-robot joint trajectory
        dt = 1.0 / fps
        l1_ious, l1_psnrs, sim_obj_masks = [], [], []
        n_steps = min(len(qpos_seq), len(real_rgbs))
        for i in range(n_steps):
            backend.step_robot_target(qpos_seq[i], float(gripper_seq[i]), dt=dt)
            sim_rgb, sim_rmask, sim_omask = backend.render_camera(calib)
            fg = real_robot_masks[i] if real_obj_masks is None else np.logical_or(real_robot_masks[i], real_obj_masks[i])
            l1 = evaluate_l1_frame(real_rgbs[i], sim_rgb, real_robot_masks[i], sim_rmask, fg_mask=fg)
            l1_ious.append(l1["l1_robot_mask_iou"])
            l1_psnrs.append(l1["l1_fg_psnr_db"])
            sim_obj_masks.append(sim_omask)

        l2_mean_iou = l2_final_iou = l2_err_px = l2_onset = None
        if real_obj_masks is not None:
            l2 = evaluate_l2_sequence(real_obj_masks[:n_steps], sim_obj_masks, fps=fps)
            l2_mean_iou = l2["l2_obj_track_mean_iou"]
            l2_final_iou = l2["l2_obj_track_final_iou"]
            l2_err_px = l2["l2_centroid_mean_err_px"]
            l2_onset = l2["l2_contact_onset_err_sec"]

        return EpisodeEvalMetrics(
            episode_id=episode_id,
            anti_cheat_passed=True,
            anti_cheat_violations=[],
            resting_max_drift_m=stab["max_resting_drift_m"],
            resting_max_penetration_m=stab["max_penetration_m"],
            l1_robot_mask_iou=float(np.mean(l1_ious)),
            l1_fg_psnr_db=float(np.mean(l1_psnrs)),
            l2_obj_track_mean_iou=l2_mean_iou,
            l2_obj_track_final_iou=l2_final_iou,
            l2_centroid_mean_err_px=l2_err_px,
            l2_contact_onset_err_sec=l2_onset,
        )

    def summarize(self, case_id: str, ep_results: List[EpisodeEvalMetrics]) -> EvalReport:
        if not ep_results:
            return EvalReport(case_id=case_id, submission_valid=False, aggregate_score=0.0,
                              per_episode=[], failure_category="schema_error", notes="No episodes evaluated.")

        if any(not r.anti_cheat_passed for r in ep_results):
            return EvalReport(case_id=case_id, submission_valid=False, aggregate_score=0.0,
                              per_episode=ep_results, failure_category="cheat_detected",
                              notes="Static anti-cheat gate failed.")

        mean_l1 = float(np.mean([r.l1_robot_mask_iou for r in ep_results]))
        max_drift = max(r.resting_max_drift_m for r in ep_results)
        l2_vals = [r.l2_obj_track_mean_iou for r in ep_results if r.l2_obj_track_mean_iou is not None]
        mean_l2 = float(np.mean(l2_vals)) if l2_vals else None

        if mean_l1 < self.min_l1_robot_iou:
            cat = "calib_fail"
            score = 0.25 * mean_l1
        elif max_drift > self.max_resting_drift_m:
            cat = "physics_unstable"
            score = 0.25 * mean_l1
        else:
            if mean_l2 is not None:
                score = 0.30 * mean_l1 + 0.70 * mean_l2
                cat = "pass" if mean_l2 >= 0.50 else "dynamics_fail"
            else:
                score = mean_l1
                cat = "pass"

        return EvalReport(
            case_id=case_id,
            submission_valid=True,
            aggregate_score=round(float(np.clip(score, 0.0, 1.0)), 4),
            per_episode=ep_results,
            failure_category=cat,
        )
