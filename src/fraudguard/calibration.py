"""Independent probability calibration, reliability diagnostics, and mapping decision."""

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from fraudguard.artifacts import load_bundle, save_bundle
from fraudguard.data import TARGET_FIELD, validate_dataframe
from fraudguard.metrics import (
    compute_average_precision,
    compute_brier_score,
    compute_log_loss_score,
    compute_top_k_metrics,
)


class SigmoidCalibrator:
    """1D Sigmoid (Platt scaling) calibrator fitted on raw model probabilities."""

    def __init__(self):
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        p = np.asarray(raw_probs).reshape(-1, 1)
        # Logit transform or direct probability input
        eps = 1e-7
        p_clipped = np.clip(p, eps, 1.0 - eps)
        logit_p = np.log(p_clipped / (1.0 - p_clipped))
        self.calibrator.fit(logit_p, y)
        self.is_fitted = True
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        p = np.asarray(raw_probs).reshape(-1, 1)
        eps = 1e-7
        p_clipped = np.clip(p, eps, 1.0 - eps)
        logit_p = np.log(p_clipped / (1.0 - p_clipped))
        return self.calibrator.predict_proba(logit_p)


def evaluate_calibration_decision(
    y_policy: np.ndarray,
    ids_policy: np.ndarray,
    raw_probs: np.ndarray,
    calib_probs: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate whether sigmoid calibration meets the protocol decision criteria on policy rows."""
    brier_raw = compute_brier_score(y_policy, raw_probs)
    brier_calib = compute_brier_score(y_policy, calib_probs)
    brier_diff = brier_raw - brier_calib  # Positive means calibration improved Brier (lower error)

    logloss_raw = compute_log_loss_score(y_policy, raw_probs)
    logloss_calib = compute_log_loss_score(y_policy, calib_probs)

    ap_raw = compute_average_precision(y_policy, raw_probs)
    ap_calib = compute_average_precision(y_policy, calib_probs)
    ap_drop = ap_raw - ap_calib  # Should be <= 0.005

    top_raw = compute_top_k_metrics(y_policy, raw_probs, ids_policy, fractions=[0.01])
    top_calib = compute_top_k_metrics(y_policy, calib_probs, ids_policy, fractions=[0.01])

    rec1_raw = top_raw["by_fraction"]["0.01"]["recall"] or 0.0
    rec1_calib = top_calib["by_fraction"]["0.01"]["recall"] or 0.0
    rec_drop = rec1_raw - rec1_calib  # Should be <= 0.02

    # Decision rule from Section 6 of 05_EXPERIMENT_PLAN.md:
    # Retain sigmoid mapping only if Brier improves by >= 0.001, AP decreases by <= 0.005, Recall@1% decreases by <= 0.02
    brier_improved = bool(brier_diff >= 0.001)
    ap_acceptable = bool(ap_drop <= 0.005)
    recall_acceptable = bool(rec_drop <= 0.02)

    use_calibration = bool(brier_improved and ap_acceptable and recall_acceptable)

    return {
        "decision": "calibrated_probability" if use_calibration else "raw_probability",
        "use_calibration": use_calibration,
        "criteria": {
            "brier_improved_ge_0_001": {
                "passed": brier_improved,
                "diff": round(float(brier_diff), 6),
            },
            "ap_drop_le_0_005": {"passed": ap_acceptable, "diff": round(float(ap_drop), 6)},
            "recall_drop_le_0_02": {"passed": recall_acceptable, "diff": round(float(rec_drop), 6)},
        },
        "raw_metrics": {
            "brier_score": round(float(brier_raw), 6),
            "log_loss": round(float(logloss_raw), 6),
            "average_precision": round(float(ap_raw), 6),
            "recall_at_1pct": round(float(rec1_raw), 6),
        },
        "calibrated_metrics": {
            "brier_score": round(float(brier_calib), 6),
            "log_loss": round(float(logloss_calib), 6),
            "average_precision": round(float(ap_calib), 6),
            "recall_at_1pct": round(float(rec1_calib), 6),
        },
    }


def plot_reliability_curves(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    calib_probs: np.ndarray,
    output_path: str = "reports/calibration_reliability.png",
) -> None:
    """Generate reliability curve plot comparing raw and calibrated probabilities."""
    prob_true_raw, prob_pred_raw = calibration_curve(
        y_true, raw_probs, n_bins=10, strategy="quantile"
    )
    prob_true_calib, prob_pred_calib = calibration_curve(
        y_true, calib_probs, n_bins=10, strategy="quantile"
    )

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(prob_pred_raw, prob_true_raw, "s-", color="#1f77b4", label="Raw Probabilities")
    plt.plot(prob_pred_calib, prob_true_calib, "o-", color="#2ca02c", label="Sigmoid Calibrated")

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Fraud", fontsize=11)
    plt.title("Calibration Reliability Curve (Policy Partition)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close()


def run_calibration_pipeline(
    selection_manifest_path: str = "artifacts/selection/selected.json",
    candidate_bundle_dir: str = "artifacts/selection/candidate_bundle",
    baseline_dir: str = "artifacts/baselines",
    split_manifest_path: str = "data/processed/split_manifest.json",
    config_path: str = "configs/project.json",
    output_dir: str = "artifacts/champion",
) -> Dict[str, Any]:
    """Execute calibration fitting on calibration partition and decision on policy partition."""
    with open(split_manifest_path, "r") as f:
        split_manifest = json.load(f)

    with open(config_path, "r") as f:
        config = json.load(f)

    raw_path = split_manifest["raw_data_path"]
    df = pd.read_csv(raw_path, low_memory=False)
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    filtered_df = filtered_df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )

    calib_info = split_manifest["partitions"]["calibration"]
    policy_info = split_manifest["partitions"]["policy"]

    calib_df = filtered_df.iloc[calib_info["indices"]].copy().reset_index(drop=True)
    policy_df = filtered_df.iloc[policy_info["indices"]].copy().reset_index(drop=True)

    # 1. Process candidate model
    candidate_bundle = load_bundle(candidate_bundle_dir, verify_hashes=True)
    preprocessor = candidate_bundle.preprocessor
    model = candidate_bundle.model

    X_calib = preprocessor.transform(calib_df)
    y_calib = calib_df[TARGET_FIELD].to_numpy().astype(int)

    X_policy = preprocessor.transform(policy_df)
    y_policy = policy_df[TARGET_FIELD].to_numpy().astype(int)
    ids_policy = policy_df["TransactionID"].to_numpy()

    # Raw probabilities
    raw_probs_calib = model.predict_proba(X_calib)[:, 1]
    raw_probs_policy = model.predict_proba(X_policy)[:, 1]

    # Fit Sigmoid calibrator on CALIBRATION rows only
    calibrator = SigmoidCalibrator()
    calibrator.fit(raw_probs_calib, y_calib)

    calib_probs_policy = calibrator.predict_proba(raw_probs_policy)[:, 1]

    # Evaluate decision on POLICY rows
    decision_info = evaluate_calibration_decision(
        y_policy=y_policy,
        ids_policy=ids_policy,
        raw_probs=raw_probs_policy,
        calib_probs=calib_probs_policy,
    )

    # Plot reliability curves
    plot_reliability_curves(
        y_true=y_policy,
        raw_probs=raw_probs_policy,
        calib_probs=calib_probs_policy,
        output_path="reports/calibration_reliability.png",
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "calibration_comparison.json", "w") as f:
        json.dump(decision_info, f, indent=2)

    # Save to artifacts/champion
    final_calibrator = calibrator if decision_info["use_calibration"] else None
    policy_dict = {
        "policy_version": "review-v1",
        "cutoff": 0.5,  # Provisional cutoff, frozen in T12
        "default_review_fraction": config.get("policy", {}).get("default_review_fraction", 0.01),
        "score_type": decision_info["decision"],
    }

    champion_dir = Path(output_dir)
    save_bundle(
        output_dir=str(champion_dir),
        preprocessor=preprocessor,
        model=model,
        feature_names=preprocessor.get_feature_names(),
        raw_mapping=preprocessor.get_raw_mapping(),
        policy=policy_dict,
        reference_summary=candidate_bundle.reference_summary,
        calibrator=final_calibrator,
        metadata={
            "model_version": "fraudguard-champion-v1",
            "model_family": candidate_bundle.manifest.get("model_family", "LightGBM"),
            "calibration_decision": decision_info,
        },
    )

    # Also apply calibration protocol to the fixed LR reference model in artifacts/baselines/logistic_reference
    lr_ref_dir = Path(baseline_dir) / "logistic_reference"
    if lr_ref_dir.exists():
        lr_bundle = load_bundle(str(lr_ref_dir), verify_hashes=True)
        lr_prep = lr_bundle.preprocessor
        lr_model = lr_bundle.model

        lr_raw_calib = lr_model.predict_proba(lr_prep.transform(calib_df))[:, 1]
        lr_raw_policy = lr_model.predict_proba(lr_prep.transform(policy_df))[:, 1]

        lr_calib = SigmoidCalibrator()
        lr_calib.fit(lr_raw_calib, y_calib)
        lr_calib_policy = lr_calib.predict_proba(lr_raw_policy)[:, 1]

        lr_decision = evaluate_calibration_decision(
            y_policy, ids_policy, lr_raw_policy, lr_calib_policy
        )
        lr_final_calib = lr_calib if lr_decision["use_calibration"] else None

        save_bundle(
            output_dir=str(lr_ref_dir),
            preprocessor=lr_prep,
            model=lr_model,
            feature_names=lr_prep.get_feature_names(),
            raw_mapping=lr_prep.get_raw_mapping(),
            policy=lr_bundle.policy,
            reference_summary=lr_bundle.reference_summary,
            calibrator=lr_final_calib,
            metadata={
                "model_version": "lr-reference-v1",
                "model_family": "LogisticRegression",
                "calibration_decision": lr_decision,
            },
        )

    return decision_info
