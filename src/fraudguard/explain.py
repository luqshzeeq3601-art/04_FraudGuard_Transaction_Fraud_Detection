"""Global permutation importance and local additive explanations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from fraudguard.artifacts import load_bundle
from fraudguard.data import TARGET_FIELD, validate_dataframe
from fraudguard.metrics import compute_average_precision


def compute_global_permutation_importance(
    preprocessor: Any,
    model: Any,
    df: pd.DataFrame,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance on policy rows using Average Precision."""
    rng = np.random.default_rng(seed)
    y_true = df[TARGET_FIELD].to_numpy().astype(int)
    raw_mapping = preprocessor.get_raw_mapping()

    # Baseline AP
    X_base = preprocessor.transform(df)
    base_probs = model.predict_proba(X_base)[:, 1]
    base_ap = compute_average_precision(y_true, base_probs)

    raw_features = list(raw_mapping.keys())
    importance_records = []

    for raw_col in raw_features:
        drops = []
        for _ in range(n_repeats):
            df_perm = df.copy()
            # Permute raw column
            df_perm[raw_col] = rng.permutation(df_perm[raw_col].to_numpy())
            X_perm = preprocessor.transform(df_perm)
            perm_probs = model.predict_proba(X_perm)[:, 1]
            perm_ap = compute_average_precision(y_true, perm_probs)
            drop = base_ap - perm_ap
            drops.append(drop)

        importance_records.append(
            {
                "raw_feature": raw_col,
                "mean_ap_importance": round(float(np.mean(drops)), 6),
                "std_ap_importance": round(float(np.std(drops)), 6),
            }
        )

    imp_df = pd.DataFrame(importance_records)
    imp_df = imp_df.sort_values(by="mean_ap_importance", ascending=False).reset_index(drop=True)
    return imp_df


def compute_local_shap_explanations(
    preprocessor: Any,
    model: Any,
    calibrator: Optional[Any],
    df_selected: pd.DataFrame,
    max_rows: int = 200,
) -> Tuple[List[Dict[str, Any]], float]:
    """Compute local SHAP contributions and verify additivity in raw model space."""
    rows_to_explain = df_selected.head(max_rows).copy()
    X = preprocessor.transform(rows_to_explain)
    raw_mapping = preprocessor.get_raw_mapping()
    encoded_names = preprocessor.get_feature_names()

    # Compute SHAP values
    if hasattr(model, "predict_proba") and "LGBM" in type(model).__name__:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_matrix = shap_values[1]
            base_val = float(explainer.expected_value[1])
        else:
            shap_matrix = shap_values
            base_val = float(
                explainer.expected_value
                if not isinstance(explainer.expected_value, np.ndarray)
                else explainer.expected_value[0]
            )
    else:
        # Linear or general model
        explainer = shap.LinearExplainer(model, X)
        shap_matrix = explainer.shap_values(X)
        base_val = float(explainer.expected_value)

    # Raw model outputs (margin/logit or probability)
    raw_probs = model.predict_proba(X)[:, 1]
    calib_probs = calibrator.predict_proba(raw_probs)[:, 1] if calibrator is not None else raw_probs

    explanations = []
    max_additivity_error = 0.0

    for i in range(len(rows_to_explain)):
        row_shap = shap_matrix[i]
        row_id = int(rows_to_explain["TransactionID"].iloc[i])
        row_amt = float(rows_to_explain["TransactionAmt"].iloc[i])

        # Aggregate transformed column contributions back to raw 12 features
        raw_contribs: Dict[str, float] = {}
        for raw_col, mapped_cols in raw_mapping.items():
            col_indices = [encoded_names.index(c) for c in mapped_cols if c in encoded_names]
            val = float(np.sum(row_shap[col_indices])) if col_indices else 0.0
            raw_contribs[raw_col] = round(val, 6)

        # Check additivity in raw model output space
        sum_contribs = base_val + float(np.sum(row_shap))
        # Compare with model's margin or raw probability output
        # For tree models with margin output, sum matches raw margin
        if hasattr(model, "predict"):
            try:
                raw_pred = float(model.predict(X[i : i + 1], raw_score=True)[0])
            except Exception:
                raw_pred = float(raw_probs[i])
        else:
            raw_pred = float(raw_probs[i])

        additivity_error = abs(sum_contribs - raw_pred)
        max_additivity_error = max(max_additivity_error, additivity_error)

        explanations.append(
            {
                "TransactionID": row_id,
                "TransactionAmt": row_amt,
                "fraud_score": round(float(calib_probs[i]), 4),
                "score_type": "calibrated_probability"
                if calibrator is not None
                else "raw_probability",
                "base_value_raw_space": round(base_val, 6),
                "raw_feature_contributions": raw_contribs,
                "sum_of_contributions_raw_space": round(sum_contribs, 6),
                "raw_model_output": round(raw_pred, 6),
                "additivity_error": round(additivity_error, 8),
                "note": "Contributions represent effects in model raw output space, not direct calibrated probability differences.",
            }
        )

    return explanations, max_additivity_error


def run_explanation_pipeline(
    artifact_dir: str = "artifacts/champion",
    split_manifest_path: str = "data/processed/split_manifest.json",
    partition_name: str = "policy",
    max_rows: int = 200,
    output_dir: str = "reports/explanations",
) -> Dict[str, Any]:
    """Generate global permutation importance and local contributions for selected policy transactions."""
    with open(split_manifest_path, "r") as f:
        split_manifest = json.load(f)

    raw_path = split_manifest["raw_data_path"]
    df = pd.read_csv(raw_path, low_memory=False)
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    filtered_df = filtered_df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )

    part_info = split_manifest["partitions"][partition_name]
    part_df = filtered_df.iloc[part_info["indices"]].copy().reset_index(drop=True)

    bundle = load_bundle(artifact_dir, verify_hashes=True)
    preprocessor = bundle.preprocessor
    model = bundle.model
    calibrator = bundle.calibrator

    # 1. Global permutation importance on policy rows
    global_imp_df = compute_global_permutation_importance(
        preprocessor=preprocessor,
        model=model,
        df=part_df,
        n_repeats=5,
        seed=42,
    )

    # 2. Local explanations for top qualifying/selected rows
    X_part = preprocessor.transform(part_df)
    raw_probs = model.predict_proba(X_part)[:, 1]
    if calibrator is not None:
        probs = calibrator.predict_proba(raw_probs)[:, 1]
    else:
        probs = raw_probs

    # Select rows with highest fraud scores
    top_indices = np.argsort(-probs)[:max_rows]
    selected_df = part_df.iloc[top_indices].copy().reset_index(drop=True)

    local_explanations, max_error = compute_local_shap_explanations(
        preprocessor=preprocessor,
        model=model,
        calibrator=calibrator,
        df_selected=selected_df,
        max_rows=max_rows,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_imp_df.to_csv(out_dir / "global_importance.csv", index=False)

    explanation_payload = {
        "partition": partition_name,
        "model_version": bundle.manifest.get("model_version"),
        "total_explained": len(local_explanations),
        "max_additivity_error": round(max_error, 8),
        "additivity_verified_le_1e4": bool(max_error <= 1e-4),
        "explanations": local_explanations,
    }

    with open(out_dir / "local_contributions.json", "w") as f:
        json.dump(explanation_payload, f, indent=2)

    return {
        "global_importance": global_imp_df.to_dict(orient="records"),
        "max_additivity_error": max_error,
        "total_local_explained": len(local_explanations),
    }
