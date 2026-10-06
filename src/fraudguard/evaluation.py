"""Frozen final holdout evaluation, uncertainty quantification, and segment diagnostics."""

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from fraudguard.artifacts import compute_file_sha256, load_bundle
from fraudguard.data import TARGET_FIELD, validate_dataframe
from fraudguard.metrics import (
    compute_average_precision,
    compute_block_bootstrap_intervals,
    compute_brier_score,
    compute_cost_sensitivity_sweep,
    compute_log_loss_score,
    compute_roc_auc,
    compute_simulated_cost,
    compute_top_k_metrics,
)
from fraudguard.policy import apply_review_policy, calculate_review_capacity, compute_policy_metrics


class EvaluationError(Exception):
    """Raised when freeze verification fails or evaluation constraints are violated."""

    pass


def verify_freeze_integrity(freeze_manifest_path: str, champion_dir: str) -> Dict[str, Any]:
    """Verify that champion artifacts have not been modified since the policy was frozen."""
    fpath = Path(freeze_manifest_path)
    if not fpath.exists():
        raise EvaluationError(f"Freeze manifest not found: {freeze_manifest_path}")

    with open(fpath, "r") as f:
        freeze_data = json.load(f)

    champ_info = freeze_data.get("champion", {})
    expected_hashes = champ_info.get("file_hashes", {})

    champ_path = Path(champion_dir)
    for fname, expected_h in expected_hashes.items():
        file_p = champ_path / fname
        if not file_p.exists():
            raise EvaluationError(f"Frozen champion file missing: {fname}")
        actual_h = compute_file_sha256(file_p)
        if actual_h != expected_h:
            raise EvaluationError(
                f"Freeze integrity violation for {fname}: expected {expected_h}, got {actual_h}"
            )

    return freeze_data


def evaluate_final_holdout(
    split_manifest_path: str = "data/processed/split_manifest.json",
    freeze_manifest_path: str = "reports/freeze_manifest.json",
    artifact_dir: str = "artifacts/champion",
    baseline_dir: str = "artifacts/baselines",
    partition_name: str = "final_test",
    output_dir: str = "reports/final",
) -> Dict[str, Any]:
    """Run frozen holdout evaluation on complete un-truncated partition."""
    # 1. Verify freeze manifest
    verify_freeze_integrity(freeze_manifest_path, artifact_dir)

    with open(split_manifest_path, "r") as f:
        split_manifest = json.load(f)

    raw_path = split_manifest["raw_data_path"]
    df = pd.read_csv(raw_path, low_memory=False)
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    filtered_df = filtered_df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )

    test_info = split_manifest["partitions"][partition_name]
    test_df = filtered_df.iloc[test_info["indices"]].copy().reset_index(drop=True)

    y_test = test_df[TARGET_FIELD].to_numpy().astype(int)
    ids_test = test_df["TransactionID"].to_numpy()
    amounts_test = test_df["TransactionAmt"].to_numpy().astype(float)
    n_test = len(test_df)
    total_fraud = int(np.sum(y_test))
    prevalence = float(total_fraud / n_test)

    # 2. Load Champion Model
    champ_bundle = load_bundle(artifact_dir, verify_hashes=True)
    X_test_champ = champ_bundle.preprocessor.transform(test_df)
    raw_champ_probs = champ_bundle.model.predict_proba(X_test_champ)[:, 1]
    if champ_bundle.calibrator is not None:
        champ_probs = champ_bundle.calibrator.predict_proba(raw_champ_probs)[:, 1]
    else:
        champ_probs = raw_champ_probs

    champ_tau = float(champ_bundle.policy.get("cutoff", 0.5))
    champ_capacity = calculate_review_capacity(n_test, fraction=0.01)
    champ_mask, _ = apply_review_policy(
        champ_probs, ids_test, cutoff=champ_tau, capacity=champ_capacity
    )

    # 3. Load LR Reference Model
    lr_bundle_dir = Path(baseline_dir) / "logistic_reference"
    lr_bundle = load_bundle(str(lr_bundle_dir), verify_hashes=True)
    X_test_lr = lr_bundle.preprocessor.transform(test_df)
    raw_lr_probs = lr_bundle.model.predict_proba(X_test_lr)[:, 1]
    if lr_bundle.calibrator is not None:
        lr_probs = lr_bundle.calibrator.predict_proba(raw_lr_probs)[:, 1]
    else:
        lr_probs = raw_lr_probs

    lr_tau = float(lr_bundle.policy.get("cutoff", 0.5))
    lr_capacity = calculate_review_capacity(n_test, fraction=0.01)
    lr_mask, _ = apply_review_policy(lr_probs, ids_test, cutoff=lr_tau, capacity=lr_capacity)

    # 4. Amount baseline
    amt_scores = amounts_test / (np.max(amounts_test) if np.max(amounts_test) > 0 else 1.0)
    amt_capacity = calculate_review_capacity(n_test, fraction=0.01)
    amt_order = np.lexsort((ids_test, -amounts_test))
    amt_mask = np.zeros(n_test, dtype=bool)
    amt_mask[amt_order[:amt_capacity]] = True

    # 5. Prior baseline
    prior_val = float(split_manifest["partitions"]["development"]["fraud_prevalence"])
    prior_probs = np.full(n_test, prior_val)

    # Compute metrics for comparison table
    models_eval = {
        "Prior Reference": {
            "model_family": "Prior",
            "scores": prior_probs,
            "mask": np.zeros(n_test, dtype=bool),
            "cutoff": prior_val,
        },
        "Amount Baseline": {
            "model_family": "Amount",
            "scores": amt_scores,
            "mask": amt_mask,
            "cutoff": float(np.min(amounts_test[amt_mask])) if np.sum(amt_mask) > 0 else 0.0,
        },
        "Logistic Regression Reference": {
            "model_family": "LogisticRegression",
            "scores": lr_probs,
            "mask": lr_mask,
            "cutoff": lr_tau,
        },
        "Champion (LightGBM)": {
            "model_family": champ_bundle.manifest.get("model_family", "LightGBM"),
            "scores": champ_probs,
            "mask": champ_mask,
            "cutoff": champ_tau,
        },
    }

    comparison_rows = []
    for m_name, m_data in models_eval.items():
        s = m_data["scores"]
        mask = m_data["mask"]
        ap = compute_average_precision(y_test, s)
        roc = compute_roc_auc(y_test, s)
        brier = compute_brier_score(y_test, s)
        top1 = compute_top_k_metrics(y_test, s, ids_test, fractions=[0.01], amounts=amounts_test)
        f1 = top1["by_fraction"]["0.01"]
        pol = compute_policy_metrics(y_test, mask, amounts=amounts_test)
        cost_res = compute_simulated_cost(y_test, mask)

        comparison_rows.append(
            {
                "model_name": m_name,
                "model_family": m_data["model_family"],
                "frozen_cutoff": m_data["cutoff"],
                "AP": round(ap, 5) if not math.isnan(ap) else None,
                "ROC_AUC": round(roc, 5) if not math.isnan(roc) else None,
                "Brier": round(brier, 5),
                "Top1pct_Precision": f1["precision"],
                "Top1pct_Recall": f1["recall"],
                "Top1pct_Lift": f1["lift"],
                "Policy_Selected": pol["selected_for_review"],
                "Policy_Precision": pol["policy_precision"],
                "Policy_Recall": pol["policy_recall"],
                "Simulated_Cost": cost_res["simulated_cost"],
                "Cost_Savings_vs_None": cost_res["cost_savings"],
            }
        )

    comp_df = pd.DataFrame(comparison_rows)

    # 6. Champion Full Top-K Spectrum
    champ_top_k = compute_top_k_metrics(
        y_test, champ_probs, ids_test, fractions=[0.005, 0.01, 0.02, 0.05], amounts=amounts_test
    )
    champ_policy_metrics = compute_policy_metrics(y_test, champ_mask, amounts=amounts_test)
    champ_cost = compute_simulated_cost(y_test, champ_mask)
    cost_sensitivity = compute_cost_sensitivity_sweep(y_test, champ_mask)

    # 7. Block Bootstrap Uncertainty (1,000 replicates)
    bootstrap_res = compute_block_bootstrap_intervals(
        df=test_df,
        scores=champ_probs,
        model_name="Champion",
        n_bootstraps=1000,
        seed=42,
    )

    # 8. Segment Analysis
    segment_rows = []
    # By ProductCD
    for prod in test_df["ProductCD"].unique():
        p_mask = (test_df["ProductCD"] == prod).to_numpy()
        sub_n = int(np.sum(p_mask))
        sub_fraud = int(np.sum(y_test[p_mask]))
        sub_prev = float(sub_fraud / sub_n) if sub_n > 0 else 0.0
        sub_sel = int(np.sum(champ_mask & p_mask))
        sub_tp = int(np.sum(champ_mask & p_mask & (y_test == 1)))
        rec = float(sub_tp / sub_fraud) if sub_fraud > 0 else None
        prec = float(sub_tp / sub_sel) if sub_sel > 0 else None

        has_support = sub_n >= 200 and sub_fraud >= 20
        segment_rows.append(
            {
                "segment_type": "ProductCD",
                "segment_value": str(prod),
                "row_count": sub_n,
                "fraud_count": sub_fraud,
                "fraud_prevalence": round(sub_prev, 5),
                "selected_count": sub_sel,
                "selected_fraud": sub_tp,
                "recall": round(rec, 5) if rec is not None else None,
                "precision": round(prec, 5) if prec is not None else None,
                "adequate_support": has_support,
            }
        )

    # By Amount Bands (using dev quantiles: <=50, (50, 150], (150, 500], >500)
    amt_bands = [
        ("<= 50", amounts_test <= 50.0),
        ("50 - 150", (amounts_test > 50.0) & (amounts_test <= 150.0)),
        ("150 - 500", (amounts_test > 150.0) & (amounts_test <= 500.0)),
        ("> 500", amounts_test > 500.0),
    ]
    for b_name, b_mask in amt_bands:
        sub_n = int(np.sum(b_mask))
        sub_fraud = int(np.sum(y_test[b_mask]))
        sub_prev = float(sub_fraud / sub_n) if sub_n > 0 else 0.0
        sub_sel = int(np.sum(champ_mask & b_mask))
        sub_tp = int(np.sum(champ_mask & b_mask & (y_test == 1)))
        rec = float(sub_tp / sub_fraud) if sub_fraud > 0 else None
        prec = float(sub_tp / sub_sel) if sub_sel > 0 else None
        has_support = sub_n >= 200 and sub_fraud >= 20
        segment_rows.append(
            {
                "segment_type": "AmountBand",
                "segment_value": b_name,
                "row_count": sub_n,
                "fraud_count": sub_fraud,
                "fraud_prevalence": round(sub_prev, 5),
                "selected_count": sub_sel,
                "selected_fraud": sub_tp,
                "recall": round(rec, 5) if rec is not None else None,
                "precision": round(prec, 5) if prec is not None else None,
                "adequate_support": has_support,
            }
        )

    # By Elapsed Time Quintiles
    quintiles = pd.qcut(test_df["TransactionDT"], q=4, labels=["T_Q1", "T_Q2", "T_Q3", "T_Q4"])
    for q_name in ["T_Q1", "T_Q2", "T_Q3", "T_Q4"]:
        q_mask = (quintiles == q_name).to_numpy()
        sub_n = int(np.sum(q_mask))
        sub_fraud = int(np.sum(y_test[q_mask]))
        sub_prev = float(sub_fraud / sub_n) if sub_n > 0 else 0.0
        sub_sel = int(np.sum(champ_mask & q_mask))
        sub_tp = int(np.sum(champ_mask & q_mask & (y_test == 1)))
        rec = float(sub_tp / sub_fraud) if sub_fraud > 0 else None
        prec = float(sub_tp / sub_sel) if sub_sel > 0 else None
        has_support = sub_n >= 200 and sub_fraud >= 20
        segment_rows.append(
            {
                "segment_type": "TimeBlock",
                "segment_value": q_name,
                "row_count": sub_n,
                "fraud_count": sub_fraud,
                "fraud_prevalence": round(sub_prev, 5),
                "selected_count": sub_sel,
                "selected_fraud": sub_tp,
                "recall": round(rec, 5) if rec is not None else None,
                "precision": round(prec, 5) if prec is not None else None,
                "adequate_support": has_support,
            }
        )

    seg_df = pd.DataFrame(segment_rows)
    cost_df = pd.DataFrame(cost_sensitivity)

    # Save output reports
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_df.to_csv(out_dir / "model_comparison.csv", index=False)
    seg_df.to_csv(out_dir / "segment_analysis.csv", index=False)
    cost_df.to_csv(out_dir / "cost_sensitivity.csv", index=False)

    final_report = {
        "evaluation_partition": partition_name,
        "total_rows": n_test,
        "total_fraud": total_fraud,
        "prevalence": round(prevalence, 6),
        "freeze_manifest_verified": True,
        "champion_model": {
            "model_version": champ_bundle.manifest.get("model_version"),
            "model_family": champ_bundle.manifest.get("model_family"),
            "score_type": champ_bundle.policy.get("score_type", "calibrated_probability"),
            "frozen_cutoff": champ_tau,
            "average_precision": round(compute_average_precision(y_test, champ_probs), 5),
            "roc_auc": round(compute_roc_auc(y_test, champ_probs), 5),
            "brier_score": round(compute_brier_score(y_test, champ_probs), 5),
            "log_loss": round(compute_log_loss_score(y_test, champ_probs), 5),
            "top_k_ranking": champ_top_k,
            "policy_metrics": champ_policy_metrics,
            "simulated_cost": champ_cost,
            "uncertainty_block_bootstrap": bootstrap_res,
        },
        "model_comparison": comparison_rows,
        "usefulness_checks": {
            "minimum_AP_greater_than_prevalence": bool(
                compute_average_precision(y_test, champ_probs) > prevalence
            ),
            "minimum_Lift_at_1pct_greater_than_1": bool(
                champ_top_k["by_fraction"]["0.01"]["lift"] > 1.0
            ),
            "stretch_AP_improvement_over_LR": bool(
                compute_average_precision(y_test, champ_probs)
                > compute_average_precision(y_test, lr_probs)
            ),
        },
    }

    with open(out_dir / "final_evaluation.json", "w") as f:
        json.dump(final_report, f, indent=2)

    return final_report
