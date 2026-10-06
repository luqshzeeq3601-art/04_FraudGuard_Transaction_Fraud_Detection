"""Unit tests for global and local explainability."""

from fraudguard.explain import run_explanation_pipeline


def test_explanation_pipeline(tmp_path):
    out_dir = tmp_path / "explanations"
    res = run_explanation_pipeline(
        artifact_dir="artifacts/champion",
        split_manifest_path="data/processed/split_manifest.json",
        partition_name="policy",
        max_rows=10,
        output_dir=str(out_dir),
    )

    assert "global_importance" in res
    assert len(res["global_importance"]) in [12, 13]  # raw predictors and temporal timestamp
    assert res["max_additivity_error"] <= 1e-4
    assert (out_dir / "global_importance.csv").exists()
    assert (out_dir / "local_contributions.json").exists()
