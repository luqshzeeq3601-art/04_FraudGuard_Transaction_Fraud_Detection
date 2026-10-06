"""End-to-end synthetic workflow integration test for CI."""

from fraudguard.data import validate_raw_file
from fraudguard.policy import freeze_policy_pipeline
from fraudguard.scoring import FraudScorer
from fraudguard.splits import build_and_save_split_manifest
from fraudguard.synthetic import generate_synthetic_transactions
from fraudguard.train import train_and_evaluate_baselines, tune_and_select_model


def test_synthetic_end_to_end_pipeline(tmp_path):
    # 1. Generate synthetic raw data
    raw_csv = tmp_path / "raw.csv"
    df_raw = generate_synthetic_transactions(n_rows=200, seed=42, fraud_rate=0.08)
    df_raw.to_csv(raw_csv, index=False)

    # 2. Validate
    report_file = tmp_path / "dq_report.json"
    dq_rep = validate_raw_file(str(raw_csv), report_path=str(report_file))
    assert dq_rep.total_rows == 200
    assert dq_rep.validation_passed is True

    # 3. Split
    proc_dir = tmp_path / "processed"
    _ = build_and_save_split_manifest(str(raw_csv), output_dir=str(proc_dir))
    assert (proc_dir / "split_manifest.json").exists()

    # 4. Train baselines
    base_dir = tmp_path / "baselines"
    comp_df = train_and_evaluate_baselines(
        manifest_path=str(proc_dir / "split_manifest.json"),
        config_path="configs/project.json",
        output_dir=str(base_dir),
        selected_models=["prior", "amount", "logistic"],
    )
    assert len(comp_df) >= 3

    # 5. Tune & select
    sel_dir = tmp_path / "selection"
    selected = tune_and_select_model(
        manifest_path=str(proc_dir / "split_manifest.json"),
        config_path="configs/project.json",
        output_dir=str(sel_dir),
    )
    assert "model_family" in selected

    # 6. Freeze policy
    champ_dir = sel_dir / "candidate_bundle"
    freeze_manifest = tmp_path / "freeze_manifest.json"
    _ = freeze_policy_pipeline(
        artifact_dir=str(champ_dir),
        baseline_dir=str(base_dir),
        split_manifest_path=str(proc_dir / "split_manifest.json"),
        review_fraction=0.01,
        output_path=str(freeze_manifest),
    )
    assert freeze_manifest.exists()

    # 7. Scorer batch & single
    scorer = FraudScorer.from_directory(str(champ_dir))
    df_batch = generate_synthetic_transactions(n_rows=50, seed=99, include_target=False)
    scored_df, queue_df = scorer.score_dataframe(df_batch, review_fraction=0.02)
    assert len(scored_df) == 50
    assert len(queue_df) >= 0
