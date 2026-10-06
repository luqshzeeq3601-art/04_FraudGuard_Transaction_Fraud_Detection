"""Unit tests for data schema and validation."""

import numpy as np
import pandas as pd
import pytest

from fraudguard.data import (
    RAW_PREDICTORS,
    ValidationError,
    normalize_address,
    validate_dataframe,
    validate_raw_file,
    validate_single_transaction,
)


def test_normalize_address():
    assert normalize_address(299.0) == "299"
    assert normalize_address("325") == "325"
    assert normalize_address(np.nan) is None
    assert normalize_address(None) is None
    assert normalize_address("") is None


def test_validate_single_transaction_valid():
    sample = {
        "TransactionID": 10001,
        "TransactionDT": 86400,
        "TransactionAmt": 150.50,
        "ProductCD": "W",
        "dist1": 15.0,
        "dist2": None,
        "card4": "visa",
        "card6": "debit",
        "addr1": 299,
        "addr2": 87,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "M4": "M0",
        "M6": "T",
    }
    cleaned = validate_single_transaction(sample)
    assert cleaned["TransactionID"] == 10001
    assert cleaned["addr1"] == "299"
    assert cleaned["addr2"] == "87"


def test_validate_single_transaction_missing_key():
    sample = {
        "TransactionID": 10001,
        "TransactionDT": 86400,
        "TransactionAmt": 150.50,
        # missing ProductCD and other keys
    }
    with pytest.raises(ValidationError, match="missing required keys"):
        validate_single_transaction(sample)


def test_validate_single_transaction_forbidden_target_key():
    sample = {
        "TransactionID": 10001,
        "TransactionDT": 86400,
        "TransactionAmt": 150.50,
        "ProductCD": "W",
        "dist1": None,
        "dist2": None,
        "card4": "visa",
        "card6": "debit",
        "addr1": 299,
        "addr2": 87,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "M4": "M0",
        "M6": "T",
        "isFraud": 1,  # forbidden
    }
    with pytest.raises(ValidationError, match="unexpected keys"):
        validate_single_transaction(sample)


def test_validate_single_transaction_invalid_amount():
    sample = {
        "TransactionID": 10001,
        "TransactionDT": 86400,
        "TransactionAmt": -50.0,
        "ProductCD": "W",
        "dist1": None,
        "dist2": None,
        "card4": "visa",
        "card6": "debit",
        "addr1": 299,
        "addr2": 87,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "M4": "M0",
        "M6": "T",
    }
    with pytest.raises(ValidationError, match="TransactionAmt must be a non-negative"):
        validate_single_transaction(sample)


def test_validate_dataframe_from_fixture():
    df = pd.read_csv("tests/fixtures/transactions.csv")
    filtered_df, extra = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    assert len(filtered_df) == len(df)
    assert "isFraud" in filtered_df.columns
    for p in RAW_PREDICTORS:
        assert p in filtered_df.columns


def test_validate_dataframe_duplicate_id():
    df = pd.read_csv("tests/fixtures/transactions.csv").copy()
    df.loc[1, "TransactionID"] = df.loc[0, "TransactionID"]
    with pytest.raises(ValidationError, match="duplicate TransactionID"):
        validate_dataframe(df, is_labelled=True, allow_extra_columns=True)


def test_validate_raw_file_report(tmp_path):
    report_file = tmp_path / "report.json"
    report = validate_raw_file("tests/fixtures/transactions.csv", report_path=str(report_file))
    assert report.total_rows == 1000
    assert report.validation_passed is True
    assert report.fraud_count is not None
    assert report_file.exists()
