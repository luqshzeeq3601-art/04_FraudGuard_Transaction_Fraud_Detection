"""Unit tests for chronological splitting and fold manifests."""

import hashlib
import json

import pandas as pd

from fraudguard.data import validate_dataframe
from fraudguard.splits import (
    build_and_save_split_manifest,
    create_chronological_splits,
    create_expanding_folds,
)


def test_chronological_splits_integrity():
    df = pd.read_csv("tests/fixtures/transactions.csv")
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    partitions = create_chronological_splits(filtered_df)

    assert len(partitions) == 4
    assert set(partitions.keys()) == {"development", "calibration", "policy", "final_test"}

    total_rows = sum(p.row_count for p in partitions.values())
    assert total_rows == len(filtered_df)

    # Check chronological boundary ordering
    assert partitions["development"].end_dt <= partitions["calibration"].start_dt
    assert partitions["calibration"].end_dt <= partitions["policy"].start_dt
    assert partitions["policy"].end_dt <= partitions["final_test"].start_dt

    # Check class diversity
    assert partitions["development"].fraud_count > 0
    assert partitions["calibration"].fraud_count > 0
    assert partitions["policy"].fraud_count > 0


def test_expanding_folds_integrity():
    df = pd.read_csv("tests/fixtures/transactions.csv")
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    partitions = create_chronological_splits(filtered_df)
    folds = create_expanding_folds(filtered_df, partitions["development"])

    assert len(folds) == 3
    # Check expanding training sizes: fold 1 < fold 2 < fold 3
    assert folds[0].train_row_count < folds[1].train_row_count < folds[2].train_row_count
    for fold in folds:
        assert fold.train_end_dt <= fold.val_start_dt
        assert fold.train_fraud_count > 0
        assert fold.val_fraud_count >= 0


def test_split_manifest_reproducibility(tmp_path):
    out_dir1 = tmp_path / "proc1"
    out_dir2 = tmp_path / "proc2"

    m1 = build_and_save_split_manifest("tests/fixtures/transactions.csv", output_dir=str(out_dir1))
    m2 = build_and_save_split_manifest("tests/fixtures/transactions.csv", output_dir=str(out_dir2))

    h1 = hashlib.sha256(json.dumps(m1, sort_keys=True).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps(m2, sort_keys=True).encode()).hexdigest()

    assert h1 == h2
    assert (out_dir1 / "split_manifest.json").exists()
    assert (out_dir2 / "split_manifest.json").exists()
