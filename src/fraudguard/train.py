"""Training pipelines, baseline models, LightGBM challengers, and cross-validation."""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from fraudguard.artifacts import save_bundle
from fraudguard.data import TARGET_FIELD, validate_dataframe
from fraudguard.features import FeaturePreprocessor
from fraudguard.metrics import (
    compute_average_precision,
    compute_brier_score,
    compute_roc_auc,
    compute_top_k_metrics,
)


class PriorClassifier(BaseEstimator, ClassifierMixin):
    """Predicts the constant training set prior fraud prevalence."""

    def __init__(self):
        self.prior_: float = 0.0
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PriorClassifier":
        y_arr = np.asarray(y)
        self.prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        p1 = np.full(n, self.prior_)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AmountClassifier(BaseEstimator, ClassifierMixin):
    """Ranks transactions strictly by TransactionAmt descending (reference ranking)."""

    def __init__(self):
        self.min_amt_: float = 0.0
        self.max_amt_: float = 1.0
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AmountClassifier":
        X_arr = np.asarray(X)
        amt = X_arr[:, 0]
        self.min_amt_ = float(np.min(amt)) if len(amt) > 0 else 0.0
        self.max_amt_ = float(np.max(amt)) if len(amt) > 0 else 1.0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        amt = X_arr[:, 0]
        denom = (self.max_amt_ - self.min_amt_) if (self.max_amt_ > self.min_amt_) else 1.0
        p1 = np.clip((amt - self.min_amt_) / denom, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def create_model(
    model_type: str,
    params: Optional[Dict[str, Any]] = None,
    train_pos_neg_ratio: Optional[float] = None,
) -> BaseEstimator:
    """Instantiate model by family and parameters."""
    p = params or {}
    seed = p.get("seed", 42)

    if model_type == "prior":
        return PriorClassifier()
    elif model_type == "amount":
        return AmountClassifier()
    elif model_type == "logistic":
        C = p.get("C", 1.0)
        class_weight = p.get("class_weight", None)
        max_iter = p.get("max_iter", 1000)
        return LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        )
    elif model_type == "lightgbm":
        n_estimators = p.get("n_estimators", 200)
        learning_rate = p.get("learning_rate", 0.05)
        num_leaves = p.get("num_leaves", 31)
        min_child_samples = p.get("min_child_samples", 50)
        weight_mode = p.get("weight_mode", "unweighted")

        scale_pos_weight = 1.0
        if weight_mode == "sqrt_imbalance" and train_pos_neg_ratio is not None:
            scale_pos_weight = math.sqrt(train_pos_neg_ratio)
        elif "scale_pos_weight" in p:
            scale_pos_weight = p["scale_pos_weight"]

        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def run_fold_cross_validation(
    df: pd.DataFrame,
    folds_info: List[Dict[str, Any]],
    model_type: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute temporal fold cross-validation with fold-local preprocessing."""
    p = params or {}
    fold_results = []

    for fold in folds_info:
        t_idx = fold["train_indices"]
        v_idx = fold["val_indices"]

        train_df = df.iloc[t_idx].copy().reset_index(drop=True)
        val_df = df.iloc[v_idx].copy().reset_index(drop=True)

        # Fold-local preprocessor fit
        preprocessor = FeaturePreprocessor(scale_numeric=True)
        X_train = preprocessor.fit_transform(train_df)
        y_train = train_df[TARGET_FIELD].to_numpy().astype(int)

        X_val = preprocessor.transform(val_df)
        y_val = val_df[TARGET_FIELD].to_numpy().astype(int)
        val_ids = val_df["TransactionID"].to_numpy()

        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        ratio = (neg_count / pos_count) if pos_count > 0 else 1.0

        model = create_model(model_type, params=p, train_pos_neg_ratio=ratio)

        start_time = time.time()
        model.fit(X_train, y_train)
        fit_duration = time.time() - start_time

        val_probs = model.predict_proba(X_val)[:, 1]

        ap = compute_average_precision(y_val, val_probs)
        roc = compute_roc_auc(y_val, val_probs)
        brier = compute_brier_score(y_val, val_probs)
        top_k = compute_top_k_metrics(y_val, val_probs, val_ids, fractions=[0.01])
        rec1 = top_k["by_fraction"]["0.01"]["recall"]
        prec1 = top_k["by_fraction"]["0.01"]["precision"]
        lift1 = top_k["by_fraction"]["0.01"]["lift"]

        fold_results.append(
            {
                "fold_id": fold["fold_id"],
                "fit_time_seconds": round(fit_duration, 4),
                "average_precision": ap,
                "roc_auc": roc,
                "brier_score": brier,
                "recall_at_1pct": rec1 if rec1 is not None else 0.0,
                "precision_at_1pct": prec1,
                "lift_at_1pct": lift1 if lift1 is not None else 0.0,
            }
        )

    # Summary averages
    mean_ap = float(np.mean([r["average_precision"] for r in fold_results]))
    std_ap = float(np.std([r["average_precision"] for r in fold_results]))
    mean_rec1 = float(np.mean([r["recall_at_1pct"] for r in fold_results]))
    std_rec1 = float(np.std([r["recall_at_1pct"] for r in fold_results]))
    mean_prec1 = float(np.mean([r["precision_at_1pct"] for r in fold_results]))
    mean_lift1 = float(np.mean([r["lift_at_1pct"] for r in fold_results]))
    mean_brier = float(np.mean([r["brier_score"] for r in fold_results]))

    return {
        "model_type": model_type,
        "params": p,
        "mean_average_precision": round(mean_ap, 5),
        "std_average_precision": round(std_ap, 5),
        "mean_recall_at_1pct": round(mean_rec1, 5),
        "std_recall_at_1pct": round(std_rec1, 5),
        "mean_precision_at_1pct": round(mean_prec1, 5),
        "mean_lift_at_1pct": round(mean_lift1, 4),
        "mean_brier_score": round(mean_brier, 5),
        "folds": fold_results,
    }


def compute_reference_monitoring_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute reference missingness, category frequencies, and numeric quantiles for monitoring."""
    summary: Dict[str, Any] = {
        "reference_row_count": len(df),
        "numeric": {},
        "categorical": {},
    }

    for col in ["TransactionAmt", "dist1", "dist2"]:
        valid = df[col].dropna()
        if len(valid) > 0:
            quantiles = [float(q) for q in np.quantile(valid, np.linspace(0.1, 0.9, 9))]
            summary["numeric"][col] = {
                "missing_rate": round(float(df[col].isna().mean()), 5),
                "quantiles": quantiles,
                "min": float(valid.min()),
                "max": float(valid.max()),
                "is_constant": bool(valid.nunique() <= 1),
            }
        else:
            summary["numeric"][col] = {
                "missing_rate": 1.0,
                "quantiles": [],
                "min": None,
                "max": None,
                "is_constant": True,
            }

    for col in [
        "ProductCD",
        "card4",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "M4",
        "M6",
    ]:
        vc = df[col].fillna("__MISSING__").value_counts(normalize=True).to_dict()
        summary["categorical"][col] = {
            "missing_rate": round(float(df[col].isna().mean()), 5),
            "frequencies": {str(k): round(float(v), 5) for k, v in vc.items()},
        }

    return summary


def train_and_evaluate_baselines(
    manifest_path: str = "data/processed/split_manifest.json",
    config_path: str = "configs/project.json",
    output_dir: str = "artifacts/baselines",
    selected_models: Optional[List[str]] = None,
    report_path: Optional[str] = None,
) -> pd.DataFrame:
    """Train baseline models on development folds and save comparisons and reference bundles."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    with open(config_path, "r") as f:
        config = json.load(f)

    raw_path = manifest["raw_data_path"]
    df = pd.read_csv(raw_path, low_memory=False)
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    filtered_df = filtered_df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )

    dev_info = manifest["partitions"]["development"]
    dev_df = filtered_df.iloc[dev_info["indices"]].copy().reset_index(drop=True)
    folds_info = manifest["folds"]

    models_to_run = selected_models or ["prior", "amount", "logistic"]
    experiment_runs = []

    # Local MLflow setup
    tracking_uri = config.get("tracking", {}).get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get("tracking", {}).get("experiment_name", "FraudGuard"))

    # 1. Prior
    if "prior" in models_to_run:
        with mlflow.start_run(run_name="Baseline_Prior"):
            res = run_fold_cross_validation(filtered_df, folds_info, "prior")
            mlflow.log_params({"model_type": "prior"})
            mlflow.log_metrics(
                {
                    "cv_mean_AP": res["mean_average_precision"],
                    "cv_mean_Recall_at_1pct": res["mean_recall_at_1pct"],
                    "cv_mean_Brier": res["mean_brier_score"],
                }
            )
            experiment_runs.append(
                {
                    "experiment_id": "E00",
                    "model_family": "Prior",
                    "variant": "Constant prevalence",
                    "mean_AP": res["mean_average_precision"],
                    "std_AP": res["std_average_precision"],
                    "mean_Recall@1%": res["mean_recall_at_1pct"],
                    "mean_Precision@1%": res["mean_precision_at_1pct"],
                    "mean_Lift@1%": res["mean_lift_at_1pct"],
                    "mean_Brier": res["mean_brier_score"],
                }
            )

    # 2. Amount
    if "amount" in models_to_run:
        with mlflow.start_run(run_name="Baseline_Amount"):
            res = run_fold_cross_validation(filtered_df, folds_info, "amount")
            mlflow.log_params({"model_type": "amount"})
            mlflow.log_metrics(
                {
                    "cv_mean_AP": res["mean_average_precision"],
                    "cv_mean_Recall_at_1pct": res["mean_recall_at_1pct"],
                }
            )
            experiment_runs.append(
                {
                    "experiment_id": "E01",
                    "model_family": "Amount",
                    "variant": "TransactionAmt descending",
                    "mean_AP": res["mean_average_precision"],
                    "std_AP": res["std_average_precision"],
                    "mean_Recall@1%": res["mean_recall_at_1pct"],
                    "mean_Precision@1%": res["mean_precision_at_1pct"],
                    "mean_Lift@1%": res["mean_lift_at_1pct"],
                    "mean_Brier": res["mean_brier_score"],
                }
            )

    # 3. Logistic Regression (unweighted C=1.0 and balanced)
    if "logistic" in models_to_run:
        for exp_id, c_val, cw, variant_name in [
            ("E02", 1.0, None, "Unweighted C=1.0 (Fixed Reference)"),
            ("E03", 1.0, "balanced", "Balanced weights C=1.0"),
        ]:
            with mlflow.start_run(run_name=f"Baseline_LR_{exp_id}"):
                p = {"C": c_val, "class_weight": cw, "seed": config.get("seed", 42)}
                res = run_fold_cross_validation(filtered_df, folds_info, "logistic", params=p)
                mlflow.log_params(p)
                mlflow.log_metrics(
                    {
                        "cv_mean_AP": res["mean_average_precision"],
                        "cv_mean_Recall_at_1pct": res["mean_recall_at_1pct"],
                        "cv_mean_Brier": res["mean_brier_score"],
                    }
                )
                experiment_runs.append(
                    {
                        "experiment_id": exp_id,
                        "model_family": "LogisticRegression",
                        "variant": variant_name,
                        "mean_AP": res["mean_average_precision"],
                        "std_AP": res["std_average_precision"],
                        "mean_Recall@1%": res["mean_recall_at_1pct"],
                        "mean_Precision@1%": res["mean_precision_at_1pct"],
                        "mean_Lift@1%": res["mean_lift_at_1pct"],
                        "mean_Brier": res["mean_brier_score"],
                    }
                )

    # 4. LightGBM challenger if requested
    if "lightgbm" in models_to_run:
        for exp_id, weight_mode, v_name in [
            ("E04", "unweighted", "LightGBM Unweighted default"),
            ("E05", "sqrt_imbalance", "LightGBM sqrt(imbalance) weight"),
        ]:
            with mlflow.start_run(run_name=f"Challenger_{exp_id}"):
                p = {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_child_samples": 50,
                    "weight_mode": weight_mode,
                    "seed": config.get("seed", 42),
                }
                res = run_fold_cross_validation(filtered_df, folds_info, "lightgbm", params=p)
                mlflow.log_params(p)
                mlflow.log_metrics(
                    {
                        "cv_mean_AP": res["mean_average_precision"],
                        "cv_mean_Recall_at_1pct": res["mean_recall_at_1pct"],
                        "cv_mean_Brier": res["mean_brier_score"],
                    }
                )
                experiment_runs.append(
                    {
                        "experiment_id": exp_id,
                        "model_family": "LightGBM",
                        "variant": v_name,
                        "mean_AP": res["mean_average_precision"],
                        "std_AP": res["std_average_precision"],
                        "mean_Recall@1%": res["mean_recall_at_1pct"],
                        "mean_Precision@1%": res["mean_precision_at_1pct"],
                        "mean_Lift@1%": res["mean_lift_at_1pct"],
                        "mean_Brier": res["mean_brier_score"],
                    }
                )

    comp_df = pd.DataFrame(experiment_runs)
    if report_path:
        comp_file = Path(report_path)
    else:
        out_p = Path(output_dir)
        fname = (
            "challenger_comparison.csv"
            if "lightgbm" in models_to_run
            else "baseline_comparison.csv"
        )
        comp_file = out_p / fname
    comp_file.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(comp_file, index=False)

    # Train fixed unweighted C=1.0 LR reference and provisional baseline bundle on all development data
    preprocessor = FeaturePreprocessor(scale_numeric=True)
    X_dev = preprocessor.fit_transform(dev_df)
    y_dev = dev_df[TARGET_FIELD].to_numpy().astype(int)

    lr_ref = LogisticRegression(C=1.0, class_weight=None, max_iter=1000, random_state=42)
    lr_ref.fit(X_dev, y_dev)

    ref_summary = compute_reference_monitoring_summary(dev_df)
    policy_dict = {
        "policy_version": "review-v1-provisional",
        "cutoff": 0.5,
        "default_review_fraction": 0.01,
        "status": "provisional",
    }

    # Save to output_dir / logistic_reference
    out_base = Path(output_dir)
    lr_ref_dir = out_base / "logistic_reference"
    save_bundle(
        output_dir=str(lr_ref_dir),
        preprocessor=preprocessor,
        model=lr_ref,
        feature_names=preprocessor.get_feature_names(),
        raw_mapping=preprocessor.get_raw_mapping(),
        policy=policy_dict,
        reference_summary=ref_summary,
        metadata={
            "model_version": "lr-reference-v1",
            "model_family": "LogisticRegression",
            "experiment_id": "E02",
        },
    )

    # Save provisional bundle to output_dir / provisional
    prov_dir = out_base / "provisional"
    save_bundle(
        output_dir=str(prov_dir),
        preprocessor=preprocessor,
        model=lr_ref,
        feature_names=preprocessor.get_feature_names(),
        raw_mapping=preprocessor.get_raw_mapping(),
        policy=policy_dict,
        reference_summary=ref_summary,
        metadata={
            "model_version": "provisional-baseline-v1",
            "model_family": "LogisticRegression",
            "experiment_id": "E02",
        },
    )

    return comp_df


def tune_and_select_model(
    manifest_path: str = "data/processed/split_manifest.json",
    config_path: str = "configs/project.json",
    output_dir: str = "artifacts/selection",
    comparison_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute bounded tuning search across LR and LightGBM and apply champion selection rule."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    with open(config_path, "r") as f:
        config = json.load(f)

    raw_path = manifest["raw_data_path"]
    df = pd.read_csv(raw_path, low_memory=False)
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    filtered_df = filtered_df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )

    dev_info = manifest["partitions"]["development"]
    dev_df = filtered_df.iloc[dev_info["indices"]].copy().reset_index(drop=True)
    folds_info = manifest["folds"]

    tracking_uri = config.get("tracking", {}).get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get("tracking", {}).get("experiment_name", "FraudGuard"))

    all_candidates: List[Dict[str, Any]] = []

    # 1. LR grid (6 configurations)
    lr_c_vals = [0.1, 1.0, 10.0]
    lr_weights = [None, "balanced"]
    for c_val in lr_c_vals:
        for cw in lr_weights:
            cfg_name = f"LR_C{c_val}_{cw or 'unweighted'}"
            params = {"C": c_val, "class_weight": cw, "seed": config.get("seed", 42)}
            with mlflow.start_run(run_name=f"Tune_{cfg_name}"):
                res = run_fold_cross_validation(filtered_df, folds_info, "logistic", params=params)
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        "cv_mean_AP": res["mean_average_precision"],
                        "cv_mean_Recall_at_1pct": res["mean_recall_at_1pct"],
                        "cv_mean_Brier": res["mean_brier_score"],
                    }
                )
                all_candidates.append(
                    {
                        "config_name": cfg_name,
                        "model_family": "LogisticRegression",
                        "params": params,
                        "mean_AP": res["mean_average_precision"],
                        "std_AP": res["std_average_precision"],
                        "mean_Recall@1%": res["mean_recall_at_1pct"],
                        "std_Recall@1%": res["std_recall_at_1pct"],
                        "mean_Precision@1%": res["mean_precision_at_1pct"],
                        "mean_Lift@1%": res["mean_lift_at_1pct"],
                        "mean_Brier": res["mean_brier_score"],
                    }
                )

    # 2. LightGBM bounded grid (12 seeded configurations)
    lgb_grid = [
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 50,
            "weight_mode": "unweighted",
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "weight_mode": "unweighted",
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 50,
            "weight_mode": "unweighted",
        },
        {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 100,
            "weight_mode": "unweighted",
        },
        {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 100,
            "weight_mode": "unweighted",
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 100,
            "weight_mode": "unweighted",
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 50,
            "weight_mode": "sqrt_imbalance",
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "weight_mode": "sqrt_imbalance",
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 50,
            "weight_mode": "sqrt_imbalance",
        },
        {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 100,
            "weight_mode": "sqrt_imbalance",
        },
        {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 100,
            "weight_mode": "sqrt_imbalance",
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 100,
            "weight_mode": "sqrt_imbalance",
        },
    ]

    for idx, g in enumerate(lgb_grid, start=1):
        cfg_name = f"LGBM_{g['weight_mode']}_n{g['n_estimators']}_lr{g['learning_rate']}_l{g['num_leaves']}"
        params = dict(g)
        params["seed"] = config.get("seed", 42)
        with mlflow.start_run(run_name=f"Tune_{cfg_name}"):
            res = run_fold_cross_validation(filtered_df, folds_info, "lightgbm", params=params)
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "cv_mean_AP": res["mean_average_precision"],
                    "cv_mean_Recall_at_1pct": res["mean_recall_at_1pct"],
                    "cv_mean_Brier": res["mean_brier_score"],
                }
            )
            all_candidates.append(
                {
                    "config_name": cfg_name,
                    "model_family": "LightGBM",
                    "params": params,
                    "mean_AP": res["mean_average_precision"],
                    "std_AP": res["std_average_precision"],
                    "mean_Recall@1%": res["mean_recall_at_1pct"],
                    "std_Recall@1%": res["std_recall_at_1pct"],
                    "mean_Precision@1%": res["mean_precision_at_1pct"],
                    "mean_Lift@1%": res["mean_lift_at_1pct"],
                    "mean_Brier": res["mean_brier_score"],
                }
            )

    # Convert to DataFrame
    comparison_df = pd.DataFrame(all_candidates)
    comparison_df = comparison_df.sort_values(by="mean_AP", ascending=False).reset_index(drop=True)

    if comparison_report_path:
        comp_file = Path(comparison_report_path)
    else:
        out_p = Path(output_dir)
        comp_file = out_p / "development_model_comparison.csv"
    comp_file.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comp_file, index=False)

    # -------------------------------------------------------------
    # Apply Champion Selection Rule (Section 5 of 05_EXPERIMENT_PLAN.md)
    # 1. Find highest mean AP
    best_ap = comparison_df["mean_AP"].max()
    # 2. Filter within 0.005 absolute of best AP
    ap_filtered = comparison_df[comparison_df["mean_AP"] >= best_ap - 0.005].copy()
    # 3. Find highest mean Recall@1% in this subset
    best_rec = ap_filtered["mean_Recall@1%"].max()
    # Filter within 0.02 of best recall
    rec_filtered = ap_filtered[ap_filtered["mean_Recall@1%"] >= best_rec - 0.02].copy()

    # 4. Tie-breaking preference:
    # Prefer LogisticRegression over LightGBM, unweighted over weighted, lower complexity
    def model_rank_key(row):
        is_lr = 1 if row["model_family"] == "LogisticRegression" else 0
        p = row["params"]
        is_unweighted = (
            1
            if (p.get("class_weight") is None and p.get("weight_mode") in [None, "unweighted"])
            else 0
        )
        complexity = (
            p.get("C", 1.0) if is_lr else (p.get("n_estimators", 200) * p.get("num_leaves", 31))
        )
        # We want to maximize is_lr, maximize is_unweighted, minimize complexity, maximize mean_AP
        return (is_lr, is_unweighted, -complexity, row["mean_AP"])

    rec_filtered["rank_key"] = rec_filtered.apply(model_rank_key, axis=1)
    selected_row = rec_filtered.sort_values(by="rank_key", ascending=False).iloc[0]

    selected_config = {
        "selected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_rule": "AP within 0.005 of max, Recall@1% within 0.02 of max, prefer LR, prefer unweighted, lower complexity",
        "config_name": selected_row["config_name"],
        "model_family": selected_row["model_family"],
        "params": selected_row["params"],
        "mean_AP": float(selected_row["mean_AP"]),
        "std_AP": float(selected_row["std_AP"]),
        "mean_Recall@1%": float(selected_row["mean_Recall@1%"]),
        "std_Recall@1%": float(selected_row["std_Recall@1%"]),
        "mean_Precision@1%": float(selected_row["mean_Precision@1%"]),
        "mean_Lift@1%": float(selected_row["mean_Lift@1%"]),
        "mean_Brier": float(selected_row["mean_Brier"]),
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "selected.json", "w") as f:
        json.dump(selected_config, f, indent=2)

    # Fit selected candidate on ALL development rows
    preprocessor = FeaturePreprocessor(scale_numeric=True)
    X_dev = preprocessor.fit_transform(dev_df)
    y_dev = dev_df[TARGET_FIELD].to_numpy().astype(int)

    pos_count = int(np.sum(y_dev == 1))
    neg_count = int(np.sum(y_dev == 0))
    ratio = (neg_count / pos_count) if pos_count > 0 else 1.0

    selected_model = create_model(
        "logistic" if selected_row["model_family"] == "LogisticRegression" else "lightgbm",
        params=selected_row["params"],
        train_pos_neg_ratio=ratio,
    )
    selected_model.fit(X_dev, y_dev)

    ref_summary = compute_reference_monitoring_summary(dev_df)
    policy_dict = {
        "policy_version": "review-v1-uncalibrated",
        "cutoff": 0.5,
        "default_review_fraction": 0.01,
        "status": "candidate_uncalibrated",
    }

    candidate_dir = out_path / "candidate_bundle"
    save_bundle(
        output_dir=str(candidate_dir),
        preprocessor=preprocessor,
        model=selected_model,
        feature_names=preprocessor.get_feature_names(),
        raw_mapping=preprocessor.get_raw_mapping(),
        policy=policy_dict,
        reference_summary=ref_summary,
        metadata={
            "model_version": f"{selected_row['model_family'].lower()}-candidate-v1",
            "model_family": selected_row["model_family"],
            "config_name": selected_row["config_name"],
        },
    )

    return selected_config
