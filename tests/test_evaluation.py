"""Unit tests for final holdout evaluation and freeze integrity verification."""

import pytest

from fraudguard.evaluation import EvaluationError, evaluate_final_holdout, verify_freeze_integrity


def test_verify_freeze_integrity():
    record = verify_freeze_integrity("reports/freeze_manifest.json", "artifacts/champion")
    assert "champion" in record


def test_verify_freeze_integrity_failure(tmp_path):
    with pytest.raises(EvaluationError, match="Freeze manifest not found"):
        verify_freeze_integrity(str(tmp_path / "nonexistent.json"), "artifacts/champion")


def test_evaluate_final_holdout(tmp_path):
    out_dir = tmp_path / "final_report"
    res = evaluate_final_holdout(
        split_manifest_path="data/processed/split_manifest.json",
        freeze_manifest_path="reports/freeze_manifest.json",
        artifact_dir="artifacts/champion",
        baseline_dir="artifacts/baselines",
        partition_name="final_test",
        output_dir=str(out_dir),
    )

    assert res["evaluation_partition"] == "final_test"
    assert "champion_model" in res
    assert "model_comparison" in res
    assert (out_dir / "final_evaluation.json").exists()
    assert (out_dir / "model_comparison.csv").exists()
    assert (out_dir / "segment_analysis.csv").exists()
    assert (out_dir / "cost_sensitivity.csv").exists()
