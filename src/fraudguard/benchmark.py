"""API latency benchmarking and training reproducibility verification."""

import json
import time
from pathlib import Path
from typing import Any, Dict

import httpx
import numpy as np
import pandas as pd

from fraudguard.artifacts import load_bundle
from fraudguard.calibration import SigmoidCalibrator
from fraudguard.data import TARGET_FIELD, validate_dataframe
from fraudguard.features import FeaturePreprocessor
from fraudguard.metrics import compute_average_precision, compute_brier_score
from fraudguard.train import create_model


def run_api_benchmark(
    url: str = "http://127.0.0.1:8000/v1/score",
    payload_path: str = "examples/synthetic_transaction.json",
    warmup_count: int = 100,
    request_count: int = 1000,
    output_path: str = "reports/api_benchmark.json",
) -> Dict[str, Any]:
    """Execute standard 100-warmup + 1000-request sequential latency benchmark."""
    with open(payload_path, "r") as f:
        payload = json.load(f)

    latencies_ms = []
    errors = 0

    # Use httpx client (works with live server or FastAPI TestClient fallback)
    with httpx.Client(timeout=10.0) as client:
        # Check if live server is reachable, otherwise use TestClient
        is_live = False
        try:
            r = client.get("http://127.0.0.1:8000/health")
            if r.status_code == 200:
                is_live = True
        except Exception:
            is_live = False

        if not is_live:
            from fastapi.testclient import TestClient

            from fraudguard.api import app

            active_client = TestClient(app)
        else:
            active_client = client

        # 1. Warmups
        for _ in range(warmup_count):
            try:
                active_client.post(url, json=payload)
            except Exception:
                pass

        # 2. Measured sequential requests
        start_bench = time.time()
        for _ in range(request_count):
            t0 = time.perf_counter()
            try:
                resp = active_client.post(url, json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    latencies_ms.append(elapsed_ms)
                else:
                    errors += 1
            except Exception:
                errors += 1
        total_bench_duration = time.time() - start_bench

    lat_arr = np.array(latencies_ms)
    p50 = float(np.percentile(lat_arr, 50)) if len(lat_arr) > 0 else 0.0
    p95 = float(np.percentile(lat_arr, 95)) if len(lat_arr) > 0 else 0.0
    p99 = float(np.percentile(lat_arr, 99)) if len(lat_arr) > 0 else 0.0
    mean_lat = float(np.mean(lat_arr)) if len(lat_arr) > 0 else 0.0

    report = {
        "benchmark_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": url,
        "warmup_count": warmup_count,
        "request_count": request_count,
        "successful_requests": len(latencies_ms),
        "error_count": errors,
        "total_duration_seconds": round(total_bench_duration, 3),
        "latency_ms": {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "mean": round(mean_lat, 3),
            "min": round(float(np.min(lat_arr)), 3) if len(lat_arr) > 0 else 0.0,
            "max": round(float(np.max(lat_arr)), 3) if len(lat_arr) > 0 else 0.0,
        },
        "target_p95_le_100ms_met": bool(p95 <= 100.0),
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    return report


def run_reproducibility_test(
    artifact_dir: str = "artifacts/champion",
    freeze_manifest_path: str = "reports/freeze_manifest.json",
    split_manifest_path: str = "data/processed/split_manifest.json",
    output_dir: str = "reports/repeatability",
) -> Dict[str, Any]:
    """Retrain frozen configuration twice on development data and verify parity within 0.001 tolerance."""
    with open(split_manifest_path, "r") as f:
        split_manifest = json.load(f)

    raw_path = split_manifest["raw_data_path"]
    df = pd.read_csv(raw_path, low_memory=False)
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    filtered_df = filtered_df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )

    dev_info = split_manifest["partitions"]["development"]
    policy_info = split_manifest["partitions"]["policy"]

    dev_df = filtered_df.iloc[dev_info["indices"]].copy().reset_index(drop=True)
    policy_df = filtered_df.iloc[policy_info["indices"]].copy().reset_index(drop=True)

    y_dev = dev_df[TARGET_FIELD].to_numpy().astype(int)
    y_policy = policy_df[TARGET_FIELD].to_numpy().astype(int)

    # Load frozen champion
    original_bundle = load_bundle(artifact_dir, verify_hashes=True)
    model_family = original_bundle.manifest.get("model_family", "LightGBM")
    is_calibrated = original_bundle.manifest.get("is_calibrated", False)

    # 1. Evaluate original bundle on policy partition
    X_pol_orig = original_bundle.preprocessor.transform(policy_df)
    raw_p_orig = original_bundle.model.predict_proba(X_pol_orig)[:, 1]
    p_orig = (
        original_bundle.calibrator.predict_proba(raw_p_orig)[:, 1] if is_calibrated else raw_p_orig
    )

    orig_ap = compute_average_precision(y_policy, p_orig)
    orig_brier = compute_brier_score(y_policy, p_orig)

    # 2. Retrain from scratch
    prep_repro = FeaturePreprocessor(scale_numeric=True)
    X_dev_repro = prep_repro.fit_transform(dev_df)

    pos_count = int(np.sum(y_dev == 1))
    neg_count = int(np.sum(y_dev == 0))
    ratio = (neg_count / pos_count) if pos_count > 0 else 1.0

    # Recreate model with exact parameters from selection manifest
    model_params = (
        original_bundle.manifest.get("metadata", {}).get("selected_config", {}).get("params", {})
    )
    if not model_params:
        sel_path = Path("artifacts/selection/selected.json")
        if sel_path.exists():
            with open(sel_path, "r") as sf:
                sel_data = json.load(sf)
                model_params = sel_data.get("params", {})
        if not model_params:
            model_params = {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "num_leaves": 15,
                "min_child_samples": 50,
                "weight_mode": "unweighted",
                "seed": 42,
            }

    repro_model = create_model(
        "lightgbm" if "LightGBM" in model_family else "logistic",
        params=model_params,
        train_pos_neg_ratio=ratio,
    )
    repro_model.fit(X_dev_repro, y_dev)

    # Fit calibrator on calibration data if applicable
    repro_calibrator = None
    if is_calibrated:
        calib_info = split_manifest["partitions"]["calibration"]
        calib_df = filtered_df.iloc[calib_info["indices"]].copy().reset_index(drop=True)
        y_calib = calib_df[TARGET_FIELD].to_numpy().astype(int)
        X_calib = prep_repro.transform(calib_df)
        raw_p_calib = repro_model.predict_proba(X_calib)[:, 1]

        repro_calibrator = SigmoidCalibrator()
        repro_calibrator.fit(raw_p_calib, y_calib)

    # Evaluate reproduced model on policy partition
    X_pol_repro = prep_repro.transform(policy_df)
    raw_p_repro = repro_model.predict_proba(X_pol_repro)[:, 1]
    p_repro = repro_calibrator.predict_proba(raw_p_repro)[:, 1] if is_calibrated else raw_p_repro

    repro_ap = compute_average_precision(y_policy, p_repro)
    repro_brier = compute_brier_score(y_policy, p_repro)

    ap_diff = abs(orig_ap - repro_ap)
    brier_diff = abs(orig_brier - repro_brier)
    max_score_diff = float(np.max(np.abs(p_orig - p_repro)))

    reproducible = bool(ap_diff <= 0.001 and brier_diff <= 0.001)

    result = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_family": model_family,
        "is_calibrated": is_calibrated,
        "original_policy_AP": round(float(orig_ap), 6),
        "reproduced_policy_AP": round(float(repro_ap), 6),
        "AP_absolute_diff": round(float(ap_diff), 6),
        "original_policy_Brier": round(float(orig_brier), 6),
        "reproduced_policy_Brier": round(float(repro_brier), 6),
        "Brier_absolute_diff": round(float(brier_diff), 6),
        "max_score_discrepancy": round(max_score_diff, 6),
        "reproducibility_tolerance_0_001_met": reproducible,
    }

    out_p = Path(output_dir)
    if out_p.suffix == ".json":
        out_p.parent.mkdir(parents=True, exist_ok=True)
        target_file = out_p
    else:
        out_p.mkdir(parents=True, exist_ok=True)
        target_file = out_p / "reproducibility.json"

    with open(target_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
