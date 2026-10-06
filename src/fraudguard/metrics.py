"""Ranking, top-K capacity, probability calibration, and simulated cost metrics."""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from fraudguard.data import TARGET_FIELD


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Average Precision (AP) score using standard step integration."""
    y_t = np.asarray(y_true)
    y_s = np.asarray(y_score)
    if len(np.unique(y_t)) < 2:
        return float("nan")
    return float(average_precision_score(y_t, y_s))


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC score, handling single-class edge cases gracefully."""
    y_t = np.asarray(y_true)
    y_s = np.asarray(y_score)
    if len(np.unique(y_t)) < 2:
        return float("nan")
    return float(roc_auc_score(y_t, y_s))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score loss."""
    y_t = np.asarray(y_true)
    y_p = np.clip(np.asarray(y_prob), 0.0, 1.0)
    return float(brier_score_loss(y_t, y_p))


def compute_log_loss_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Log Loss score with probability clipping."""
    y_t = np.asarray(y_true)
    y_p = np.clip(np.asarray(y_prob), 1e-15, 1.0 - 1e-15)
    if len(np.unique(y_t)) < 2:
        return float("nan")
    return float(log_loss(y_t, y_p))


def compute_top_k_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ids: np.ndarray,
    fractions: Optional[List[float]] = None,
    amounts: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute unconditional Top-K ranking metrics with deterministic tie-breaking.

    Sort order: descending score, ascending ID.
    """
    if fractions is None:
        fractions = [0.005, 0.01, 0.02, 0.05]

    y_t = np.asarray(y_true).astype(int)
    y_s = np.asarray(y_score).astype(float)
    id_arr = np.asarray(ids)
    amt_arr = np.asarray(amounts).astype(float) if amounts is not None else None

    n = len(y_t)
    if n == 0:
        return {}

    total_positives = int(np.sum(y_t))
    prevalence = float(total_positives / n)

    # Sort indices: primary = -score (descending), secondary = id (ascending)
    # Using lexsort: last key is primary sort key
    order = np.lexsort((id_arr, -y_s))
    sorted_y = y_t[order]
    sorted_amt = amt_arr[order] if amt_arr is not None else None

    results: Dict[str, Any] = {
        "total_rows": n,
        "total_fraud": total_positives,
        "prevalence": round(prevalence, 6),
        "by_fraction": {},
    }

    for q in fractions:
        k = max(1, min(n, math.ceil(q * n)))
        top_y = sorted_y[:k]
        fraud_in_top_k = int(np.sum(top_y))
        precision_k = float(fraud_in_top_k / k)
        recall_k = float(fraud_in_top_k / total_positives) if total_positives > 0 else float("nan")
        lift_k = float(precision_k / prevalence) if prevalence > 0 else float("nan")

        amt_captured = None
        if sorted_amt is not None:
            amt_captured = float(np.sum(sorted_amt[:k][top_y == 1]))

        results["by_fraction"][str(q)] = {
            "fraction": q,
            "k": k,
            "selected_fraud": fraud_in_top_k,
            "precision": round(precision_k, 6),
            "recall": round(recall_k, 6) if not math.isnan(recall_k) else None,
            "lift": round(lift_k, 4) if not math.isnan(lift_k) else None,
            "fraud_amount_captured": round(amt_captured, 2) if amt_captured is not None else None,
        }

    return results


def compute_simulated_cost(
    y_true: np.ndarray,
    selected_mask: np.ndarray,
    missed_fraud_cost: float = 100.0,
    review_cost: float = 1.0,
    legitimate_friction_cost: float = 2.0,
) -> Dict[str, Any]:
    """Compute financial/operational simulated cost given a binary review queue decision."""
    y_t = np.asarray(y_true).astype(int)
    mask = np.asarray(selected_mask).astype(bool)

    total_fraud = int(np.sum(y_t))
    selected_count = int(np.sum(mask))

    selected_fraud = int(np.sum(y_t[mask]))
    fn = total_fraud - selected_fraud
    fp = selected_count - selected_fraud
    tp = selected_fraud
    tn = len(y_t) - total_fraud - fp

    cost = fn * missed_fraud_cost + selected_count * review_cost + fp * legitimate_friction_cost

    # Review-none baseline cost (all fraud missed, 0 reviews, 0 friction)
    cost_review_none = total_fraud * missed_fraud_cost
    savings = cost_review_none - cost

    return {
        "missed_fraud_cost": missed_fraud_cost,
        "review_cost": review_cost,
        "legitimate_friction_cost": legitimate_friction_cost,
        "total_fraud": total_fraud,
        "selected_count": selected_count,
        "selected_fraud (TP)": tp,
        "missed_fraud (FN)": fn,
        "false_alarms (FP)": fp,
        "true_negatives (TN)": tn,
        "simulated_cost": round(float(cost), 2),
        "cost_review_none": round(float(cost_review_none), 2),
        "cost_savings": round(float(savings), 2),
    }


def compute_cost_sensitivity_sweep(
    y_true: np.ndarray,
    selected_mask: np.ndarray,
    missed_costs: Optional[List[float]] = None,
    review_costs: Optional[List[float]] = None,
    friction_costs: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Run parameter sweep across cost combinations."""
    if missed_costs is None:
        missed_costs = [20.0, 100.0, 500.0]
    if review_costs is None:
        review_costs = [0.5, 1.0, 2.0]
    if friction_costs is None:
        friction_costs = [0.0, 2.0, 5.0]

    grid_results = []
    for m in missed_costs:
        for r in review_costs:
            for f in friction_costs:
                res = compute_simulated_cost(
                    y_true,
                    selected_mask,
                    missed_fraud_cost=m,
                    review_cost=r,
                    legitimate_friction_cost=f,
                )
                grid_results.append(res)
    return grid_results


def compute_block_bootstrap_intervals(
    df: pd.DataFrame,
    scores: np.ndarray,
    model_name: str = "model",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute dependence-aware 95% paired block bootstrap intervals using elapsed-day bins."""
    rng = np.random.default_rng(seed)
    df_eval = df.copy().reset_index(drop=True)
    df_eval["score"] = scores
    df_eval["day_block"] = np.floor(df_eval["TransactionDT"] / 86400.0).astype(int)

    unique_blocks = df_eval["day_block"].unique()
    n_blocks = len(unique_blocks)

    if n_blocks < 2:
        return {
            "status": "insufficient_blocks",
            "block_count": n_blocks,
            "message": "Fewer than 2 day blocks available for block bootstrap.",
        }

    ap_list = []
    recall_1pct_list = []
    valid_replicates = 0

    for _ in range(n_bootstraps):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        # Concatenate selected blocks
        sample_df = pd.concat(
            [df_eval[df_eval["day_block"] == b] for b in sampled_blocks], ignore_index=True
        )
        y_t = sample_df[TARGET_FIELD].to_numpy()
        y_s = sample_df["score"].to_numpy()
        ids = sample_df["TransactionID"].to_numpy()

        if len(np.unique(y_t)) < 2 or np.sum(y_t) == 0:
            continue

        ap = compute_average_precision(y_t, y_s)
        top_res = compute_top_k_metrics(y_t, y_s, ids, fractions=[0.01])
        rec1 = top_res["by_fraction"]["0.01"]["recall"]

        if not math.isnan(ap) and rec1 is not None:
            ap_list.append(ap)
            recall_1pct_list.append(rec1)
            valid_replicates += 1

    if valid_replicates < 50:
        return {
            "status": "insufficient_support",
            "valid_replicates": valid_replicates,
            "message": "Too few valid replicates with positive classes.",
        }

    ap_arr = np.array(ap_list)
    rec_arr = np.array(recall_1pct_list)

    return {
        "status": "success",
        "n_bootstraps": n_bootstraps,
        "valid_replicates": valid_replicates,
        "average_precision": {
            "mean": round(float(np.mean(ap_arr)), 5),
            "ci_95_lower": round(float(np.percentile(ap_arr, 2.5)), 5),
            "ci_95_upper": round(float(np.percentile(ap_arr, 97.5)), 5),
        },
        "recall_at_1pct": {
            "mean": round(float(np.mean(rec_arr)), 5),
            "ci_95_lower": round(float(np.percentile(rec_arr, 2.5)), 5),
            "ci_95_upper": round(float(np.percentile(rec_arr, 97.5)), 5),
        },
    }
