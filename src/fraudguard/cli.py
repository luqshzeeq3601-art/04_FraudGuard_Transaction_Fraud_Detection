"""Command-line interface for FraudGuard."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraudguard",
        description="FraudGuard: Transaction Fraud Detection and Capped Review System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate raw dataset against schema")
    validate_parser.add_argument("--input", required=True, help="Path to input CSV file")
    validate_parser.add_argument(
        "--config", default="configs/project.json", help="Path to config file"
    )
    validate_parser.add_argument(
        "--report", default="reports/data_quality.json", help="Path to output report"
    )

    # split
    split_parser = subparsers.add_parser(
        "split", help="Generate chronological splits and fold manifests"
    )
    split_parser.add_argument("--input", required=True, help="Path to input CSV file")
    split_parser.add_argument(
        "--config", default="configs/project.json", help="Path to config file"
    )
    split_parser.add_argument(
        "--output-dir", default="data/processed", help="Output directory for manifests"
    )

    # train
    train_parser = subparsers.add_parser("train", help="Train baseline or challenger models")
    train_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to train: prior, amount, logistic, lightgbm",
    )
    train_parser.add_argument(
        "--config", default="configs/project.json", help="Path to config file"
    )
    train_parser.add_argument(
        "--split-manifest",
        default="data/processed/split_manifest.json",
        help="Path to split manifest",
    )
    train_parser.add_argument(
        "--output-dir", default="artifacts/baselines", help="Output directory for artifacts"
    )
    train_parser.add_argument(
        "--report", default=None, help="Optional custom path for comparison report CSV"
    )

    # tune
    tune_parser = subparsers.add_parser("tune", help="Tune models and select candidate")
    tune_parser.add_argument("--config", default="configs/project.json", help="Path to config file")
    tune_parser.add_argument(
        "--split-manifest",
        default="data/processed/split_manifest.json",
        help="Path to split manifest",
    )
    tune_parser.add_argument(
        "--output-dir", default="artifacts/selection", help="Output directory for selection"
    )
    tune_parser.add_argument(
        "--comparison-report",
        default="reports/development_model_comparison.csv",
        help="Path to output comparison CSV report",
    )

    # calibrate
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate candidate probabilities")
    cal_parser.add_argument(
        "--selection",
        default="artifacts/selection/selected.json",
        help="Path to selection manifest",
    )
    cal_parser.add_argument(
        "--baseline-dir", default="artifacts/baselines", help="Path to baseline artifacts"
    )
    cal_parser.add_argument("--config", default="configs/project.json", help="Path to config file")
    cal_parser.add_argument(
        "--output-dir", default="artifacts/champion", help="Output directory for champion bundle"
    )

    # freeze-policy
    freeze_parser = subparsers.add_parser(
        "freeze-policy", help="Freeze review policy and generate freeze manifest"
    )
    freeze_parser.add_argument(
        "--artifact-dir", default="artifacts/champion", help="Path to champion artifact directory"
    )
    freeze_parser.add_argument(
        "--baseline-dir", default="artifacts/baselines", help="Path to baseline artifacts"
    )
    freeze_parser.add_argument(
        "--split-manifest",
        default="data/processed/split_manifest.json",
        help="Path to split manifest",
    )
    freeze_parser.add_argument(
        "--review-fraction", type=float, default=0.01, help="Review fraction (default 0.01)"
    )
    freeze_parser.add_argument(
        "--output", default="reports/freeze_manifest.json", help="Output path for freeze manifest"
    )

    # evaluate
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run frozen final evaluation on holdout set"
    )
    eval_parser.add_argument(
        "--artifact-dir", default="artifacts/champion", help="Path to champion artifact directory"
    )
    eval_parser.add_argument(
        "--baseline-dir", default="artifacts/baselines", help="Path to baseline artifacts"
    )
    eval_parser.add_argument(
        "--split-manifest",
        default="data/processed/split_manifest.json",
        help="Path to split manifest",
    )
    eval_parser.add_argument(
        "--partition", default="final_test", help="Partition name to evaluate on"
    )
    eval_parser.add_argument(
        "--freeze-manifest", default="reports/freeze_manifest.json", help="Path to freeze manifest"
    )
    eval_parser.add_argument(
        "--replay", action="store_true", help="Explicit replay mode if evaluating viewed holdout"
    )
    eval_parser.add_argument(
        "--output-dir", default="reports/final", help="Output directory for final reports"
    )

    # explain
    explain_parser = subparsers.add_parser("explain", help="Generate global and local explanations")
    explain_parser.add_argument(
        "--artifact-dir", default="artifacts/champion", help="Path to champion artifact directory"
    )
    explain_parser.add_argument(
        "--split-manifest",
        default="data/processed/split_manifest.json",
        help="Path to split manifest",
    )
    explain_parser.add_argument("--partition", default="policy", help="Partition to explain on")
    explain_parser.add_argument(
        "--max-rows", type=int, default=200, help="Max rows for local explanations"
    )
    explain_parser.add_argument(
        "--output-dir", default="reports/explanations", help="Output directory for explanations"
    )

    # score
    score_parser = subparsers.add_parser(
        "score", help="Batch score transactions and generate review queue"
    )
    score_parser.add_argument("--input", required=True, help="Path to unlabelled batch CSV")
    score_parser.add_argument(
        "--artifact-dir", default="artifacts/champion", help="Path to artifact directory"
    )
    score_parser.add_argument("--review-fraction", type=float, default=0.01, help="Review fraction")
    score_parser.add_argument(
        "--output-dir", default="reports/demo", help="Output directory for scored files"
    )

    # monitor
    monitor_parser = subparsers.add_parser(
        "monitor", help="Generate input and score drift monitoring reports"
    )
    monitor_parser.add_argument("--input", required=True, help="Path to new batch CSV")
    monitor_parser.add_argument(
        "--artifact-dir", default="artifacts/champion", help="Path to artifact directory"
    )
    monitor_parser.add_argument(
        "--output-dir", default="reports/monitoring", help="Output directory for monitoring"
    )

    # benchmark
    bm_parser = subparsers.add_parser("benchmark", help="Benchmark single-score API latency")
    bm_parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/score", help="API URL to benchmark"
    )
    bm_parser.add_argument(
        "--input",
        default="examples/synthetic_transaction.json",
        help="Path to request payload JSON",
    )
    bm_parser.add_argument("--warmups", type=int, default=100, help="Number of warmup requests")
    bm_parser.add_argument("--requests", type=int, default=1000, help="Number of measured requests")
    bm_parser.add_argument("--concurrency", type=int, default=1, help="Concurrency level")
    bm_parser.add_argument(
        "--output", default="reports/api_benchmark.json", help="Path to output benchmark report"
    )

    # reproduce
    rep_parser = subparsers.add_parser(
        "reproduce", help="Test reproducibility by retraining frozen configuration"
    )
    rep_parser.add_argument(
        "--artifact-dir", default="artifacts/champion", help="Path to champion artifact directory"
    )
    rep_parser.add_argument(
        "--freeze-manifest", default="reports/freeze_manifest.json", help="Path to freeze manifest"
    )
    rep_parser.add_argument(
        "--split-manifest",
        default="data/processed/split_manifest.json",
        help="Path to split manifest",
    )
    rep_parser.add_argument(
        "--output-dir", default="reports/repeatability", help="Output directory for repeatability"
    )

    # generate-synthetic
    synth_parser = subparsers.add_parser(
        "generate-synthetic",
        help="Generate multi-day scaled synthetic transaction benchmark fixture",
    )
    synth_parser.add_argument(
        "--rows", type=int, default=50000, help="Number of rows to generate (default 50000)"
    )
    synth_parser.add_argument(
        "--days", type=float, default=30.0, help="Temporal duration in days (default 30.0)"
    )
    synth_parser.add_argument(
        "--fraud-rate", type=float, default=0.035, help="Target fraud prevalence (default 0.035)"
    )
    synth_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    synth_parser.add_argument(
        "--output", default="data/raw/train_transaction.csv", help="Output CSV path"
    )

    return parser


def main(args=None):
    parser = build_parser()
    parsed = parser.parse_args(args)
    if not parsed.command:
        parser.print_help()
        sys.exit(1)

    if parsed.command == "validate":
        from fraudguard.data import validate_raw_file

        report = validate_raw_file(
            parsed.input, config_path=parsed.config, report_path=parsed.report
        )
        print(
            f"Validation successful: {report.total_rows} rows, {len(report.selected_columns)} selected columns."
        )
        print(f"Data quality report saved to: {parsed.report}")
    elif parsed.command == "split":
        from fraudguard.splits import build_and_save_split_manifest

        manifest = build_and_save_split_manifest(
            parsed.input, output_dir=parsed.output_dir, config_path=parsed.config
        )
        print(f"Split manifest successfully created at: {parsed.output_dir}/split_manifest.json")
        for p_name, p_info in manifest["partitions"].items():
            print(
                f"  Partition '{p_name}': {p_info['row_count']} rows, {p_info['fraud_count']} fraud ({p_info['fraud_prevalence']:.4%})"
            )
        print(f"  Internal development folds created: {len(manifest['folds'])}")
        print("Development EDA report saved to: reports/development_eda.md")
    elif parsed.command == "train":
        from fraudguard.train import train_and_evaluate_baselines

        comp_df = train_and_evaluate_baselines(
            manifest_path=parsed.split_manifest,
            config_path=parsed.config,
            output_dir=parsed.output_dir,
            selected_models=parsed.models,
            report_path=parsed.report,
        )
        print("Training complete. Baseline comparison table:")
        print(comp_df.to_string(index=False))
    elif parsed.command == "tune":
        from fraudguard.train import tune_and_select_model

        selected = tune_and_select_model(
            manifest_path=parsed.split_manifest,
            config_path=parsed.config,
            output_dir=parsed.output_dir,
            comparison_report_path=parsed.comparison_report,
        )
        print("Tuning and model selection complete.")
        print(f"Selected Champion: {selected['model_family']} ({selected['config_name']})")
        print(f"  Mean AP: {selected['mean_AP']:.5f} (std: {selected['std_AP']:.5f})")
        print(f"  Mean Recall@1%: {selected['mean_Recall@1%']:.5f}")
        print(
            f"  Mean Precision@1%: {selected['mean_Precision@1%']:.5f} (Lift@1%: {selected['mean_Lift@1%']:.2f}x)"
        )
        print(f"  Selection manifest: {parsed.output_dir}/selected.json")
    elif parsed.command == "calibrate":
        from fraudguard.calibration import run_calibration_pipeline

        cal_res = run_calibration_pipeline(
            selection_manifest_path=parsed.selection,
            candidate_bundle_dir="artifacts/selection/candidate_bundle",
            baseline_dir=parsed.baseline_dir,
            split_manifest_path="data/processed/split_manifest.json",
            config_path=parsed.config,
            output_dir=parsed.output_dir,
        )
        print("Calibration evaluation complete.")
        print(f"Decision: {cal_res['decision']}")
        print(
            f"Brier score (Raw vs Calibrated): {cal_res['raw_metrics']['brier_score']:.5f} vs {cal_res['calibrated_metrics']['brier_score']:.5f}"
        )
        print(
            f"AP (Raw vs Calibrated): {cal_res['raw_metrics']['average_precision']:.5f} vs {cal_res['calibrated_metrics']['average_precision']:.5f}"
        )
        print("Reliability curve saved to: reports/calibration_reliability.png")
        print(f"Champion bundle created at: {parsed.output_dir}")
    elif parsed.command == "freeze-policy":
        from fraudguard.policy import freeze_policy_pipeline

        record = freeze_policy_pipeline(
            artifact_dir=parsed.artifact_dir,
            baseline_dir=parsed.baseline_dir,
            split_manifest_path=parsed.split_manifest,
            review_fraction=parsed.review_fraction,
            output_path=parsed.output,
        )
        print("Policy freeze successfully executed.")
        print(
            f"Champion model cutoff tau: {record['champion']['cutoff']:.6f} ({record['champion']['score_type']})"
        )
        if record.get("baseline_lr_reference"):
            print(
                f"Baseline LR reference cutoff tau: {record['baseline_lr_reference']['cutoff']:.6f}"
            )
        print(f"Freeze manifest written to: {parsed.output}")
    elif parsed.command == "evaluate":
        from fraudguard.evaluation import evaluate_final_holdout

        res = evaluate_final_holdout(
            split_manifest_path=parsed.split_manifest,
            freeze_manifest_path=parsed.freeze_manifest,
            artifact_dir=parsed.artifact_dir,
            baseline_dir=parsed.baseline_dir,
            partition_name=parsed.partition,
            output_dir=parsed.output_dir,
        )
        print(f"Frozen final evaluation complete on partition '{parsed.partition}':")
        print(
            f"  Holdout rows: {res['total_rows']}, Total fraud: {res['total_fraud']} ({res['prevalence']:.4%})"
        )
        print(
            f"  Champion AP: {res['champion_model']['average_precision']:.5f} (ROC-AUC: {res['champion_model']['roc_auc']:.5f})"
        )
        top1 = res["champion_model"]["top_k_ranking"]["by_fraction"]["0.01"]
        print(
            f"  Champion Top-1% Precision: {top1['precision']:.5f}, Recall: {top1['recall']:.5f} (Lift: {top1['lift']:.2f}x)"
        )
        print(
            f"  Policy Review Count: {res['champion_model']['policy_metrics']['selected_for_review']}"
        )
        print(f"  Reports saved to: {parsed.output_dir}")
    elif parsed.command == "explain":
        from fraudguard.explain import run_explanation_pipeline

        exp_res = run_explanation_pipeline(
            artifact_dir=parsed.artifact_dir,
            split_manifest_path=parsed.split_manifest,
            partition_name=parsed.partition,
            max_rows=parsed.max_rows,
            output_dir=parsed.output_dir,
        )
        print(f"Explainability pipeline complete on partition '{parsed.partition}':")
        print(f"  Global permutation features evaluated: {len(exp_res['global_importance'])}")
        print(f"  Local transactions explained: {exp_res['total_local_explained']}")
        print(
            f"  Max SHAP additivity error: {exp_res['max_additivity_error']:.2e} (<= 1e-4 check passed)"
        )
    elif parsed.command == "score":
        from pathlib import Path

        import pandas as pd

        from fraudguard.scoring import FraudScorer

        scorer = FraudScorer.from_directory(parsed.artifact_dir)
        df_in = pd.read_csv(parsed.input, low_memory=False)
        scored_df, queue_df = scorer.score_dataframe(df_in, review_fraction=parsed.review_fraction)
        out_dir = Path(parsed.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(out_dir / "scored.csv", index=False)
        queue_df.to_csv(out_dir / "review_queue.csv", index=False)
        print(f"Batch scoring complete for {len(scored_df)} rows:")
        print(f"  Selected for review: {len(queue_df)} rows")
        print(f"  Scored file: {out_dir}/scored.csv")
        print(f"  Review queue file: {out_dir}/review_queue.csv")
    elif parsed.command == "monitor":
        from fraudguard.monitoring import generate_monitoring_report

        rep = generate_monitoring_report(
            input_batch_path=parsed.input,
            artifact_dir=parsed.artifact_dir,
            output_dir=parsed.output_dir,
        )
        print(f"Drift monitoring report generated for {rep['batch_row_count']} rows:")
        print(f"  Total warnings detected: {rep['total_warnings_count']}")
        print(
            f"  Score PSI: {rep['fraud_score_monitoring']['score_psi']} (Warning: {rep['fraud_score_monitoring']['score_psi_warning']})"
        )
        print(
            f"  Reports saved to: {parsed.output_dir}/report.json and {parsed.output_dir}/report.html"
        )
    elif parsed.command == "benchmark":
        from fraudguard.benchmark import run_api_benchmark

        bm_res = run_api_benchmark(
            url=parsed.url,
            payload_path=parsed.input,
            warmup_count=parsed.warmups,
            request_count=parsed.requests,
            output_path=parsed.output,
        )
        print(f"API latency benchmark complete ({bm_res['request_count']} requests):")
        print(
            f"  p50: {bm_res['latency_ms']['p50']} ms | p95: {bm_res['latency_ms']['p95']} ms | p99: {bm_res['latency_ms']['p99']} ms"
        )
        print(
            f"  Mean: {bm_res['latency_ms']['mean']} ms | Target <= 100ms: {'MET' if bm_res['target_p95_le_100ms_met'] else 'MISSED'}"
        )
        print(f"  Report written to: {parsed.output}")
    elif parsed.command == "reproduce":
        from fraudguard.benchmark import run_reproducibility_test

        rep_res = run_reproducibility_test(
            artifact_dir=parsed.artifact_dir,
            freeze_manifest_path=parsed.freeze_manifest,
            split_manifest_path=parsed.split_manifest,
            output_dir=parsed.output_dir,
        )
        print("Reproducibility verification test complete:")
        print(
            f"  Original Policy AP: {rep_res['original_policy_AP']:.5f} vs Reproduced: {rep_res['reproduced_policy_AP']:.5f} (Diff: {rep_res['AP_absolute_diff']:.6f})"
        )
        print(
            f"  Original Policy Brier: {rep_res['original_policy_Brier']:.5f} vs Reproduced: {rep_res['reproduced_policy_Brier']:.5f} (Diff: {rep_res['Brier_absolute_diff']:.6f})"
        )
        print(
            f"  Reproducibility Tolerance (<= 0.001): {'PASSED' if rep_res['reproducibility_tolerance_0_001_met'] else 'FAILED'}"
        )
        print(f"  Report written to: {parsed.output_dir}/reproducibility.json")
    elif parsed.command == "generate-synthetic":
        from pathlib import Path

        from fraudguard.synthetic import generate_synthetic_transactions

        df = generate_synthetic_transactions(
            n_rows=parsed.rows,
            seed=parsed.seed,
            fraud_rate=parsed.fraud_rate,
            days=parsed.days,
            include_target=True,
        )
        out_path = Path(parsed.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Generated {len(df)} synthetic transactions spanning {parsed.days} days.")
        print(
            f"  Target fraud rate: {parsed.fraud_rate:.2%} (Actual fraud cases: {df['isFraud'].sum()})"
        )
        print(f"  Saved to: {parsed.output}")
    else:
        print(f"FraudGuard CLI: command '{parsed.command}' called.")


if __name__ == "__main__":
    main()
