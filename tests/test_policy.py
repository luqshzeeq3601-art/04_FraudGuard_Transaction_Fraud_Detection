"""Unit tests for policy freezing and freeze manifest."""

import json
import shutil

import pytest

from fraudguard.policy import freeze_policy_pipeline


def test_freeze_policy_pipeline(tmp_path):
    freeze_output = tmp_path / "freeze_manifest.json"
    champ_copy = tmp_path / "champion"
    shutil.copytree("artifacts/champion", champ_copy)
    base_copy = tmp_path / "baselines"
    shutil.copytree("artifacts/baselines", base_copy)

    record = freeze_policy_pipeline(
        artifact_dir=str(champ_copy),
        baseline_dir=str(base_copy),
        split_manifest_path="data/processed/split_manifest.json",
        review_fraction=0.01,
        output_path=str(freeze_output),
    )

    assert "freeze_timestamp_utc" in record
    assert "champion" in record
    assert "cutoff" in record["champion"]
    assert freeze_output.exists()

    with open(champ_copy / "policy.json", "r") as f:
        champ_pol = json.load(f)
    assert champ_pol["status"] == "frozen"
    assert champ_pol["cutoff"] == pytest.approx(record["champion"]["cutoff"], abs=1e-5)
