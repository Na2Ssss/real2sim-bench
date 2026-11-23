"""Task-Success Evaluation (A) and Sim-Real Consistency Evaluation (B)."""

from r2s_bench.eval.task_success import (
    RolloutMonitor,
    TaskEvalResult,
    evaluate_block_in_drawer,
    evaluate_cup_in_bowl,
    evaluate_pouring_beans,
)
from r2s_bench.eval.sim_real_eval import (
    EpisodeLabel,
    SimRealReport,
    compute_policy_correlation,
    compute_sim_real_report,
)

__all__ = [
    "RolloutMonitor",
    "TaskEvalResult",
    "evaluate_cup_in_bowl",
    "evaluate_block_in_drawer",
    "evaluate_pouring_beans",
    "EpisodeLabel",
    "SimRealReport",
    "compute_sim_real_report",
    "compute_policy_correlation",
]
