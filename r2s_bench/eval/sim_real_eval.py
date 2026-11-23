"""Part B: Sim-Real Task-Success Consistency Evaluator.

Measures how accurately a reconstructed simulation environment predicts REAL
robot task success / failure:
  1. Single-trajectory confusion matrix on Real Success (y_real=1) + Real Failure (y_real=0):
       - TPR (Sim=PASS | Real=PASS): does the sim faithfully reproduce real successes?
       - TNR (Sim=FAIL | Real=FAIL): does the sim reject real failures (not too lenient)?
       - Balanced Accuracy = 0.5 * (TPR + TNR)
       - Overall Agreement = (TP + TN) / N
  2. Multi-policy ranking correlation (SIMPLER benchmark standard):
       - Pearson r, Spearman rho, and MMRV (Mean Maximum Rank Violation) across policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from r2s_bench.eval.task_success import TaskEvalResult


@dataclass
class EpisodeLabel:
    episode_id: str
    real_success: bool
    sim_result: TaskEvalResult
    policy_name: str = "teleop"


@dataclass
class SimRealReport:
    n_total: int
    n_real_pos: int
    n_real_neg: int
    tp: int
    fn: int
    tn: int
    fp: int
    tpr: float
    tnr: float
    balanced_accuracy: float
    agreement: float
    f1: float
    failure_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_total": self.n_total,
            "n_real_pos": self.n_real_pos,
            "n_real_neg": self.n_real_neg,
            "confusion_matrix": {"TP": self.tp, "FN": self.fn, "TN": self.tn, "FP": self.fp},
            "tpr": round(self.tpr, 4),
            "tnr": round(self.tnr, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "agreement": round(self.agreement, 4),
            "f1": round(self.f1, 4),
            "failure_breakdown": self.failure_breakdown,
        }

    def format_summary(self) -> str:
        lines = [
            "=== Sim-Real Task-Success Consistency Report ===",
            f"Total Episodes : {self.n_total} (Real Success={self.n_real_pos}, Real Failure={self.n_real_neg})",
            f"Confusion Mat  : TP={self.tp} | FN={self.fn} | TN={self.tn} | FP={self.fp}",
            f"TPR (Real=1)   : {self.tpr*100:.1f}% ({self.tp}/{self.n_real_pos})",
            f"TNR (Real=0)   : {self.tnr*100:.1f}% ({self.tn}/{self.n_real_neg})" if self.n_real_neg > 0 else "TNR (Real=0)   : N/A (no real negatives)",
            f"Balanced Acc   : {self.balanced_accuracy*100:.1f}%",
            f"Agreement      : {self.agreement*100:.1f}% ({self.tp + self.tn}/{self.n_total})",
            f"F1 Score       : {self.f1:.3f}",
        ]
        if self.failure_breakdown:
            lines.append(f"Sim Fail Modes : {self.failure_breakdown}")
        return "\n".join(lines)


def compute_sim_real_report(episodes: Sequence[EpisodeLabel]) -> SimRealReport:
    """Compute confusion matrix and Sim-Real task-success consistency metrics."""
    tp = fn = tn = fp = 0
    fail_counts: Dict[str, int] = {}
    for ep in episodes:
        y_real = bool(ep.real_success)
        y_sim = bool(ep.sim_result.success)
        if y_real and y_sim:
            tp += 1
        elif y_real and not y_sim:
            fn += 1
        elif not y_real and not y_sim:
            tn += 1
        else:
            fp += 1
        if not y_sim:
            r = ep.sim_result.failure_reason
            fail_counts[r] = fail_counts.get(r, 0) + 1

    n_pos = tp + fn
    n_neg = tn + fp
    n_tot = n_pos + n_neg
    tpr = (tp / n_pos) if n_pos > 0 else 0.0
    tnr = (tn / n_neg) if n_neg > 0 else 0.0
    bal_acc = (0.5 * (tpr + tnr)) if (n_pos > 0 and n_neg > 0) else (tpr if n_pos > 0 else tnr)
    agreement = ((tp + tn) / n_tot) if n_tot > 0 else 0.0
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0

    return SimRealReport(
        n_total=n_tot,
        n_real_pos=n_pos,
        n_real_neg=n_neg,
        tp=tp,
        fn=fn,
        tn=tn,
        fp=fp,
        tpr=tpr,
        tnr=tnr,
        balanced_accuracy=bal_acc,
        agreement=agreement,
        f1=f1,
        failure_breakdown=fail_counts,
    )


def compute_policy_correlation(
    real_rates: Sequence[float],
    sim_rates: Sequence[float],
) -> Dict[str, float]:
    """Compute SIMPLER-style multi-policy Sim-Real correlation: Pearson r, Spearman rho, MMRV."""
    r_arr = np.asarray(real_rates, dtype=np.float64)
    s_arr = np.asarray(sim_rates, dtype=np.float64)
    n = len(r_arr)
    if n < 2:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan"), "mmrv": float("nan")}

    pr = float(pearsonr(r_arr, s_arr).statistic) if (np.std(r_arr) > 0 and np.std(s_arr) > 0) else 0.0
    sr = float(spearmanr(r_arr, s_arr).statistic) if (np.std(r_arr) > 0 and np.std(s_arr) > 0) else 0.0

    # Mean Maximum Rank Violation (SIMPLER, Li et al. 2024)
    rank_violations: List[float] = []
    for i in range(n):
        max_viol = 0.0
        for j in range(n):
            if i == j:
                continue
            # Violation occurs when sim orders (i, j) opposite to real
            if (s_arr[i] > s_arr[j] and r_arr[i] < r_arr[j]) or (
                s_arr[i] < s_arr[j] and r_arr[i] > r_arr[j]
            ):
                max_viol = max(max_viol, abs(float(r_arr[i] - r_arr[j])))
        rank_violations.append(max_viol)

    return {
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4),
        "mmrv": round(float(np.mean(rank_violations)), 4),
    }
