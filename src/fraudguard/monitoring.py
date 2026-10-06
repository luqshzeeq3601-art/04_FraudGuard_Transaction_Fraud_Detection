"""Input and score drift monitoring with PSI, missingness checks, and HTML reporting."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fraudguard.artifacts import load_bundle
from fraudguard.data import (
    CATEGORICAL_PREDICTORS,
    NUMERIC_PREDICTORS,
    RAW_PREDICTORS,
    validate_dataframe,
)
from fraudguard.scoring import FraudScorer


def calculate_psi(
    ref_fractions: np.ndarray,
    batch_fractions: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) with epsilon smoothing and renormalization."""
    p = np.asarray(ref_fractions).astype(float) + epsilon
    q = np.asarray(batch_fractions).astype(float) + epsilon

    p = p / np.sum(p)
    q = q / np.sum(q)

    psi_val = np.sum((q - p) * np.log(q / p))
    return float(psi_val)


def compute_numeric_psi(
    ref_quantiles: List[float],
    batch_values: np.ndarray,
    psi_threshold: float = 0.20,
) -> Dict[str, Any]:
    """Compute PSI for numeric feature using reference quantile bin edges with open bounds."""
    vals = np.asarray(batch_values).astype(float)
    valid_vals = vals[~np.isnan(vals)]

    if len(valid_vals) == 0:
        return {
            "status": "all_missing",
            "psi": None,
            "warning": True,
            "bins": [],
        }

    # Unique sorted edges from reference quantiles
    edges = sorted(list(set(ref_quantiles)))
    if len(edges) < 2:
        return {
            "status": "insufficient_edges",
            "psi": 0.0,
            "warning": False,
            "bins": [],
        }

    # Bins: (-inf, e0], (e0, e1], ..., (e_{k-1}, inf)
    bin_counts = []
    # First bin: (-inf, edges[0]]
    bin_counts.append(int(np.sum(valid_vals <= edges[0])))
    for i in range(len(edges) - 1):
        bin_counts.append(int(np.sum((valid_vals > edges[i]) & (valid_vals <= edges[i + 1]))))
    # Last bin: (edges[-1], inf)
    bin_counts.append(int(np.sum(valid_vals > edges[-1])))

    total_valid = len(valid_vals)
    batch_fractions = np.array(bin_counts, dtype=float) / total_valid

    # Theoretical uniform reference fractions across quantile bins
    n_bins = len(bin_counts)
    ref_fractions = np.full(n_bins, 1.0 / n_bins)

    psi = calculate_psi(ref_fractions, batch_fractions)
    warning = bool(psi >= psi_threshold)

    # Human-readable bin descriptions with open outer bounds (JSON nulls for infinity)
    bin_specs = []
    bin_specs.append(
        {
            "lower_bound": None,
            "upper_bound": edges[0],
            "batch_fraction": round(float(batch_fractions[0]), 5),
        }
    )
    for i in range(len(edges) - 1):
        bin_specs.append(
            {
                "lower_bound": edges[i],
                "upper_bound": edges[i + 1],
                "batch_fraction": round(float(batch_fractions[i + 1]), 5),
            }
        )
    bin_specs.append(
        {
            "lower_bound": edges[-1],
            "upper_bound": None,
            "batch_fraction": round(float(batch_fractions[-1]), 5),
        }
    )

    return {
        "status": "computed",
        "psi": round(psi, 5),
        "warning": warning,
        "bin_count": n_bins,
        "bins": bin_specs,
    }


def generate_monitoring_report(
    input_batch_path: str,
    artifact_dir: str = "artifacts/champion",
    output_dir: str = "reports/monitoring",
    missing_rate_threshold: float = 0.05,
    unseen_category_threshold: float = 0.05,
    psi_threshold: float = 0.20,
) -> Dict[str, Any]:
    """Analyze batch drift against frozen reference summary and generate JSON and HTML reports."""
    bundle = load_bundle(artifact_dir, verify_hashes=True)
    ref_summary = bundle.reference_summary
    scorer = FraudScorer(bundle)

    df = pd.read_csv(input_batch_path, low_memory=False)
    filtered_df, _ = validate_dataframe(df, is_labelled=False, allow_extra_columns=False)
    n = len(filtered_df)

    has_support = n >= 100
    warnings: List[Dict[str, Any]] = []

    # 1. Missingness checks
    missing_results: Dict[str, Any] = {}
    for col in RAW_PREDICTORS:
        batch_missing = float(filtered_df[col].isna().mean())
        if col in ref_summary.get("numeric", {}):
            ref_missing = ref_summary["numeric"][col].get("missing_rate", 0.0)
        else:
            ref_missing = ref_summary.get("categorical", {}).get(col, {}).get("missing_rate", 0.0)

        diff = abs(batch_missing - ref_missing)
        is_warn = bool(diff >= missing_rate_threshold)
        if is_warn:
            warnings.append(
                {
                    "type": "missingness_shift",
                    "feature": col,
                    "ref_missing": round(ref_missing, 4),
                    "batch_missing": round(batch_missing, 4),
                    "diff": round(diff, 4),
                }
            )

        missing_results[col] = {
            "ref_missing_rate": round(ref_missing, 4),
            "batch_missing_rate": round(batch_missing, 4),
            "diff": round(diff, 4),
            "warning": is_warn,
        }

    # 2. Categorical unseen checks
    cat_results: Dict[str, Any] = {}
    for col in CATEGORICAL_PREDICTORS:
        ref_cat_data = ref_summary.get("categorical", {}).get(col, {})
        known_cats = set(ref_cat_data.get("frequencies", {}).keys())

        vals = filtered_df[col].fillna("__MISSING__").astype(str)
        unseen_mask = ~vals.isin(known_cats)
        unseen_rate = float(unseen_mask.mean())
        is_warn = bool(unseen_rate >= unseen_category_threshold)

        if is_warn:
            warnings.append(
                {
                    "type": "unseen_category",
                    "feature": col,
                    "unseen_rate": round(unseen_rate, 4),
                    "threshold": unseen_category_threshold,
                }
            )

        cat_results[col] = {
            "known_categories_count": len(known_cats),
            "unseen_rate": round(unseen_rate, 4),
            "warning": is_warn,
        }

    # 3. Numeric PSI checks
    num_results: Dict[str, Any] = {}
    for col in NUMERIC_PREDICTORS:
        ref_num_data = ref_summary.get("numeric", {}).get(col, {})
        if ref_num_data.get("is_constant", False):
            # Constant reference handling
            ref_min = ref_num_data.get("min", 0.0)
            valid_vals = filtered_df[col].dropna().to_numpy()
            diff_frac = float(np.mean(valid_vals != ref_min)) if len(valid_vals) > 0 else 0.0
            is_warn = bool(diff_frac >= 0.05)
            if is_warn:
                warnings.append(
                    {
                        "type": "constant_feature_shift",
                        "feature": col,
                        "different_fraction": round(diff_frac, 4),
                    }
                )
            num_results[col] = {
                "status": "constant_reference",
                "different_fraction": round(diff_frac, 4),
                "warning": is_warn,
            }
        else:
            q_edges = ref_num_data.get("quantiles", [])
            psi_res = compute_numeric_psi(
                q_edges, filtered_df[col].to_numpy(), psi_threshold=psi_threshold
            )
            if psi_res["warning"]:
                warnings.append(
                    {
                        "type": "numeric_psi_warning",
                        "feature": col,
                        "psi": psi_res["psi"],
                    }
                )
            num_results[col] = psi_res

    # 4. Score Distribution PSI
    probs = scorer.predict_probabilities(filtered_df)
    # Generate 10 decile bins for scores: [0, 0.1, 0.2, ..., 1.0]
    score_edges = [round(float(x), 2) for x in np.linspace(0.1, 0.9, 9)]
    score_psi_res = compute_numeric_psi(score_edges, probs, psi_threshold=psi_threshold)
    if score_psi_res["warning"]:
        warnings.append(
            {
                "type": "score_psi_warning",
                "psi": score_psi_res["psi"],
            }
        )

    report_payload = {
        "batch_row_count": n,
        "adequate_sample_support": has_support,
        "total_warnings_count": len(warnings),
        "warnings": warnings,
        "missingness": missing_results,
        "categoricals": cat_results,
        "numerics": num_results,
        "fraud_score_monitoring": {
            "score_type": scorer.score_type,
            "mean_score": round(float(np.mean(probs)), 4),
            "median_score": round(float(np.median(probs)), 4),
            "cutoff": scorer.cutoff,
            "review_rate": round(float(np.mean(probs >= scorer.cutoff)), 4),
            "score_psi": score_psi_res["psi"],
            "score_psi_warning": score_psi_res["warning"],
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report_payload, f, indent=2)

    # Generate standalone HTML report
    html_content = generate_html_monitoring_report(report_payload)
    with open(out_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_payload


def generate_html_monitoring_report(report: Dict[str, Any]) -> str:
    """Render clean, responsive HTML monitoring report."""
    warnings_rows = ""
    for w in report["warnings"]:
        warnings_rows += f"<tr><td><code>{w.get('type')}</code></td><td>{w.get('feature', 'Score')}</td><td>{json.dumps(w)}</td></tr>"

    if not warnings_rows:
        warnings_rows = (
            "<tr><td colspan='3' style='color: green;'>No drift warnings detected.</td></tr>"
        )

    num_rows = ""
    for col, data in report["numerics"].items():
        psi_val = data.get("psi")
        warn_style = "color: red; font-weight: bold;" if data.get("warning") else "color: green;"
        num_rows += f"<tr><td>{col}</td><td>{data.get('status')}</td><td style='{warn_style}'>{psi_val if psi_val is not None else 'N/A'}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FraudGuard Monitoring Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 30px; background: #f8f9fa; color: #212529; }}
        .card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #1a202c; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #edf2f7; font-weight: 600; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .badge-success {{ background: #c6f6d5; color: #22543d; }}
        .badge-warn {{ background: #fed7d7; color: #742a2a; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>FraudGuard Drift & Quality Report</h1>
        <p><strong>Batch Sample Size:</strong> {report["batch_row_count"]} rows | <strong>Adequate Support (&ge;100):</strong> {"Yes" if report["adequate_sample_support"] else "No"}</p>
        <p><strong>Total Warnings:</strong> <span class="badge {"badge-warn" if report["total_warnings_count"] > 0 else "badge-success"}">{report["total_warnings_count"]} Warnings</span></p>
    </div>

    <div class="card">
        <h2>Active Diagnostics & Warnings</h2>
        <table>
            <thead><tr><th>Warning Type</th><th>Target</th><th>Details</th></tr></thead>
            <tbody>{warnings_rows}</tbody>
        </table>
    </div>

    <div class="card">
        <h2>Numeric Feature Stability (PSI)</h2>
        <table>
            <thead><tr><th>Feature</th><th>Status</th><th>PSI Value</th></tr></thead>
            <tbody>{num_rows}</tbody>
        </table>
    </div>

    <div class="card">
        <h2>Score Distribution Summary</h2>
        <p><strong>Mean Fraud Score:</strong> {report["fraud_score_monitoring"]["mean_score"]}</p>
        <p><strong>Review Rate at Frozen Cutoff ({report["fraud_score_monitoring"]["cutoff"]}):</strong> {report["fraud_score_monitoring"]["review_rate"]:.4%}</p>
        <p><strong>Score PSI:</strong> {report["fraud_score_monitoring"]["score_psi"]}</p>
    </div>
</body>
</html>
"""
    return html
