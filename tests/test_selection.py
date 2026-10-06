"""Unit tests for model tuning and selection rule logic."""

import pandas as pd

from fraudguard.train import tune_and_select_model


def test_tune_and_selection_rule(tmp_path):
    out_dir = tmp_path / "selection"
    selected = tune_and_select_model(
        manifest_path="data/processed/split_manifest.json",
        config_path="configs/project.json",
        output_dir=str(out_dir),
    )

    assert "model_family" in selected
    assert "mean_AP" in selected
    assert "mean_Recall@1%" in selected
    assert (out_dir / "selected.json").exists()
    assert (out_dir / "candidate_bundle" / "manifest.json").exists()
    assert (out_dir / "development_model_comparison.csv").exists()

    # Verify that the comparison table has both LR and LightGBM entries
    comp_df = pd.read_csv(out_dir / "development_model_comparison.csv")
    assert "LogisticRegression" in comp_df["model_family"].values
    assert "LightGBM" in comp_df["model_family"].values
    assert len(comp_df) >= 18  # 6 LR + 12 LightGBM

    # Verify selection rule consistency
    best_ap = comp_df["mean_AP"].max()
    ap_filtered = comp_df[comp_df["mean_AP"] >= best_ap - 0.005]
    best_rec = ap_filtered["mean_Recall@1%"].max()
    rec_filtered = ap_filtered[ap_filtered["mean_Recall@1%"] >= best_rec - 0.02]
    assert selected["config_name"] in rec_filtered["config_name"].values
