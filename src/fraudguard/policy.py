"""Review capacity, score thresholding, tie-breaking, and policy evaluation."""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def calculate_review_capacity(row_count: int, fraction: float) -> int:
    """Compute maximum number of transactions that can be reviewed given a batch size and capacity fraction."""
    if row_count < 1:
        raise ValueError(f"Expected a nonempty batch, got row_count={row_count}")
    if not (0 < fraction <= 1.0):
        raise ValueError(f"Expected review fraction in (0, 1], got {fraction}")
    return min(row_count, math.ceil(row_count * fraction))


def determine_policy_cutoff(
    scores: np.ndarray,
    ids: np.ndarray,
    fraction: float = 0.01,
) -> float:
    """Determine score cutoff tau on policy partition scores such that exactly top K candidates qualify."""
    s_arr = np.asarray(scores).astype(float)
    id_arr = np.asarray(ids)
    n = len(s_arr)
    if n == 0:
        raise ValueError("Cannot compute cutoff on empty scores array.")

    k = calculate_review_capacity(n, fraction)
    # Sort order: descending score, ascending ID
    order = np.lexsort((id_arr, -s_arr))
    # The score at rank K (1-indexed, so index k-1)
    tau = float(s_arr[order[k - 1]])
    return tau


def apply_review_policy(
    scores: np.ndarray,
    ids: np.ndarray,
    cutoff: float,
    capacity: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply frozen cutoff tau and capacity constraint with deterministic tie-breaking.

    Returns:
        (selected_mask, ranks) where selected_mask is a boolean array of length N,
        and ranks is an integer array from 1 to N indicating sorted rank position.
    """
    s_arr = np.asarray(scores).astype(float)
    id_arr = np.asarray(ids)
    n = len(s_arr)

    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=int)

    # Sort descending score, ascending ID
    order = np.lexsort((id_arr, -s_arr))
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    # Threshold condition
    qualifies = s_arr >= cutoff

    selected_mask = np.zeros(n, dtype=bool)

    if capacity is None:
        selected_mask = qualifies
    else:
        # Take at most `capacity` qualifying items in rank order
        selected_count = 0
        for idx in order:
            if qualifies[idx] and selected_count < capacity:
                selected_mask[idx] = True
                selected_count += 1

    return selected_mask, ranks


def compute_policy_metrics(
    y_true: np.ndarray,
    selected_mask: np.ndarray,
    amounts: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute review queue performance metrics under the active policy."""
    y_t = np.asarray(y_true).astype(int)
    mask = np.asarray(selected_mask).astype(bool)
    n = len(y_t)
    total_fraud = int(np.sum(y_t))
    selected_count = int(np.sum(mask))

    selected_fraud = int(np.sum(y_t[mask]))
    review_rate = float(selected_count / n) if n > 0 else 0.0
    precision = float(selected_fraud / selected_count) if selected_count > 0 else None
    recall = float(selected_fraud / total_fraud) if total_fraud > 0 else None

    amt_captured = None
    if amounts is not None:
        amt_arr = np.asarray(amounts).astype(float)
        amt_captured = float(np.sum(amt_arr[mask][y_t[mask] == 1]))

    return {
        "total_rows": n,
        "total_fraud": total_fraud,
        "selected_for_review": selected_count,
        "review_rate": round(review_rate, 6),
        "selected_fraud": selected_fraud,
        "policy_precision": round(precision, 6) if precision is not None else None,
        "policy_recall": round(recall, 6) if recall is not None else None,
        "observed_fraud_amount_captured": round(amt_captured, 2)
        if amt_captured is not None
        else None,
        "queue_underfilled": selected_count == 0,
    }


def freeze_policy_pipeline(
    artifact_dir: str = "artifacts/champion",
    baseline_dir: str = "artifacts/baselines",
    split_manifest_path: str = "data/processed/split_manifest.json",
    review_fraction: float = 0.01,
    output_path: str = "reports/freeze_manifest.json",
) -> Dict[str, Any]:
    """Determine policy cutoff on policy partition and freeze champion and reference bundles."""
    import hashlib
    import json
    import time
    from pathlib import Path

    from fraudguard.artifacts import load_bundle, save_bundle
    from fraudguard.data import validate_dataframe

    with open(split_manifest_path, "r") as f:
        split_manifest = json.load(f)

    with open(split_manifest_path, "rb") as f:
        split_manifest_hash = hashlib.sha256(f.read()).hexdigest()

    raw_path = split_manifest["raw_data_path"]
    with open(raw_path, "rb") as f:
        raw_data_hash = hashlib.sha256(f.read()).hexdigest()

    df = pd.read_csv(raw_path, low_memory=False)
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    filtered_df = filtered_df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )

    policy_info = split_manifest["partitions"]["policy"]
    policy_df = filtered_df.iloc[policy_info["indices"]].copy().reset_index(drop=True)
    policy_ids = policy_df["TransactionID"].to_numpy()

    # 1. Champion bundle freeze
    champ_bundle = load_bundle(artifact_dir, verify_hashes=True)
    X_policy = champ_bundle.preprocessor.transform(policy_df)
    raw_champ_probs = champ_bundle.model.predict_proba(X_policy)[:, 1]

    if champ_bundle.calibrator is not None:
        champ_probs = champ_bundle.calibrator.predict_proba(raw_champ_probs)[:, 1]
    else:
        champ_probs = raw_champ_probs

    champ_tau = determine_policy_cutoff(champ_probs, policy_ids, fraction=review_fraction)

    champ_policy = {
        "policy_version": "review-v1-frozen",
        "cutoff": round(float(champ_tau), 6),
        "default_review_fraction": review_fraction,
        "score_type": "calibrated_probability"
        if champ_bundle.calibrator is not None
        else "raw_probability",
        "status": "frozen",
    }

    save_bundle(
        output_dir=artifact_dir,
        preprocessor=champ_bundle.preprocessor,
        model=champ_bundle.model,
        feature_names=champ_bundle.preprocessor.get_feature_names(),
        raw_mapping=champ_bundle.preprocessor.get_raw_mapping(),
        policy=champ_policy,
        reference_summary=champ_bundle.reference_summary,
        calibrator=champ_bundle.calibrator,
        metadata={
            **champ_bundle.manifest.get("metadata", {}),
            "policy_freeze": champ_policy,
            "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    reloaded_champ = load_bundle(artifact_dir, verify_hashes=True)

    # 2. Baseline LR reference freeze
    lr_ref_dir = Path(baseline_dir) / "logistic_reference"
    lr_info = {}
    if lr_ref_dir.exists():
        lr_bundle = load_bundle(str(lr_ref_dir), verify_hashes=True)
        X_lr = lr_bundle.preprocessor.transform(policy_df)
        raw_lr_probs = lr_bundle.model.predict_proba(X_lr)[:, 1]
        if lr_bundle.calibrator is not None:
            lr_probs = lr_bundle.calibrator.predict_proba(raw_lr_probs)[:, 1]
        else:
            lr_probs = raw_lr_probs

        lr_tau = determine_policy_cutoff(lr_probs, policy_ids, fraction=review_fraction)
        lr_policy = {
            "policy_version": "review-v1-frozen",
            "cutoff": round(float(lr_tau), 6),
            "default_review_fraction": review_fraction,
            "score_type": "calibrated_probability"
            if lr_bundle.calibrator is not None
            else "raw_probability",
            "status": "frozen",
        }

        save_bundle(
            output_dir=str(lr_ref_dir),
            preprocessor=lr_bundle.preprocessor,
            model=lr_bundle.model,
            feature_names=lr_bundle.preprocessor.get_feature_names(),
            raw_mapping=lr_bundle.preprocessor.get_raw_mapping(),
            policy=lr_policy,
            reference_summary=lr_bundle.reference_summary,
            calibrator=lr_bundle.calibrator,
            metadata={
                **lr_bundle.manifest.get("metadata", {}),
                "policy_freeze": lr_policy,
                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        reloaded_lr = load_bundle(str(lr_ref_dir), verify_hashes=True)
        lr_info = {
            "artifact_dir": str(lr_ref_dir),
            "model_version": reloaded_lr.manifest.get("model_version"),
            "cutoff": float(lr_tau),
            "file_hashes": reloaded_lr.manifest.get("file_hashes", {}),
        }

    freeze_record = {
        "freeze_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "review_fraction": review_fraction,
        "split_manifest_sha256": split_manifest_hash,
        "raw_data_sha256": raw_data_hash,
        "champion": {
            "artifact_dir": artifact_dir,
            "model_version": reloaded_champ.manifest.get("model_version"),
            "cutoff": float(champ_tau),
            "score_type": champ_policy["score_type"],
            "file_hashes": reloaded_champ.manifest.get("file_hashes", {}),
        },
        "baseline_lr_reference": lr_info,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(freeze_record, f, indent=2)

    return freeze_record
