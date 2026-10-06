"""Data validation, schema enforcement, and profiling for FraudGuard."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RAW_PREDICTORS: List[str] = [
    "TransactionAmt",
    "ProductCD",
    "dist1",
    "dist2",
    "card4",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "M4",
    "M6",
]

NUMERIC_PREDICTORS: List[str] = ["TransactionAmt", "dist1", "dist2"]
CATEGORICAL_PREDICTORS: List[str] = [
    "ProductCD",
    "card4",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "M4",
    "M6",
]

ENVELOPE_FIELDS: List[str] = ["TransactionID", "TransactionDT"]
TARGET_FIELD: str = "isFraud"

REQUIRED_FIELDS: List[str] = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"]
NULLABLE_FIELDS: List[str] = [
    "dist1",
    "dist2",
    "card4",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "M4",
    "M6",
]


class ValidationError(ValueError):
    """Raised when data fails FraudGuard schema validation."""

    pass


@dataclass
class DataQualityReport:
    total_rows: int
    selected_columns: List[str]
    dropped_columns: List[str]
    missing_rates: Dict[str, float]
    column_types: Dict[str, str]
    peak_memory_mb: float
    fraud_count: Optional[int] = None
    fraud_prevalence: Optional[float] = None
    validation_passed: bool = True
    errors: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_address(val: Any) -> Optional[str]:
    """Normalize address code to string representation without implying magnitude."""
    if pd.isna(val) or val is None or val == "" or str(val).lower() == "nan":
        return None
    try:
        # Convert float like 299.0 to integer string '299'
        float_val = float(val)
        if float_val.is_integer() and float_val >= 0:
            return str(int(float_val))
        elif float_val >= 0:
            return str(float_val)
        else:
            raise ValidationError(f"Invalid negative address value: {val}")
    except (ValueError, TypeError):
        s_val = str(val).strip()
        if s_val:
            return s_val
        return None


def validate_single_transaction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single transaction payload for real-time scoring."""
    expected_keys = set(ENVELOPE_FIELDS + RAW_PREDICTORS)
    received_keys = set(data.keys())

    # Check for forbidden or extra keys
    extra_keys = received_keys - expected_keys
    if extra_keys:
        raise ValidationError(f"Payload contains unexpected keys: {sorted(list(extra_keys))}")

    # Check for missing keys
    missing_keys = expected_keys - received_keys
    if missing_keys:
        raise ValidationError(f"Payload missing required keys: {sorted(list(missing_keys))}")

    # Validate TransactionID
    tx_id = data.get("TransactionID")
    if (
        tx_id is None
        or isinstance(tx_id, bool)
        or not isinstance(tx_id, (int, np.integer))
        or tx_id <= 0
    ):
        raise ValidationError(f"TransactionID must be a positive integer, got: {tx_id}")

    # Validate TransactionDT
    tx_dt = data.get("TransactionDT")
    if (
        tx_dt is None
        or isinstance(tx_dt, bool)
        or not isinstance(tx_dt, (int, float, np.number))
        or not np.isfinite(tx_dt)
        or tx_dt < 0
    ):
        raise ValidationError(f"TransactionDT must be a non-negative finite number, got: {tx_dt}")

    # Validate TransactionAmt
    tx_amt = data.get("TransactionAmt")
    if (
        tx_amt is None
        or isinstance(tx_amt, bool)
        or not isinstance(tx_amt, (int, float, np.number))
        or not np.isfinite(tx_amt)
        or tx_amt < 0
    ):
        raise ValidationError(f"TransactionAmt must be a non-negative finite number, got: {tx_amt}")

    # Validate ProductCD
    prod_cd = data.get("ProductCD")
    if (
        prod_cd is None
        or not isinstance(prod_cd, str)
        or len(prod_cd.strip()) == 0
        or len(prod_cd) > 128
    ):
        raise ValidationError(
            f"ProductCD must be a non-empty string <= 128 characters, got: {prod_cd}"
        )

    # Validate nullable numerics
    for col in ["dist1", "dist2"]:
        val = data.get(col)
        if val is not None and not pd.isna(val):
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float, np.number))
                or not np.isfinite(val)
                or val < 0
            ):
                raise ValidationError(
                    f"{col} must be null or finite non-negative number, got: {val}"
                )

    # Validate nullable strings
    for col in ["card4", "card6", "P_emaildomain", "R_emaildomain", "M4", "M6"]:
        val = data.get(col)
        if val is not None and not pd.isna(val):
            if not isinstance(val, str) or len(val) > 128:
                raise ValidationError(f"{col} must be null or string <= 128 characters, got: {val}")

    # Validate address codes
    cleaned = dict(data)
    for col in ["addr1", "addr2"]:
        val = data.get(col)
        cleaned[col] = normalize_address(val)

    return cleaned


def validate_dataframe(
    df: pd.DataFrame,
    is_labelled: bool = False,
    allow_extra_columns: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and filter a DataFrame according to FraudGuard contracts."""
    if df.empty:
        raise ValidationError("Input DataFrame is empty.")

    original_cols = list(df.columns)
    target_set = set(ENVELOPE_FIELDS + RAW_PREDICTORS)
    if is_labelled:
        target_set.add(TARGET_FIELD)

    # Check for missing required columns
    missing_cols = [c for c in target_set if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"DataFrame is missing required columns: {missing_cols}")

    # If extra columns exist
    extra_cols = [c for c in original_cols if c not in target_set]
    if extra_cols and not allow_extra_columns:
        raise ValidationError(f"DataFrame contains unexpected extra columns: {extra_cols}")

    # Filter to selected columns
    selected_cols = ENVELOPE_FIELDS + RAW_PREDICTORS + ([TARGET_FIELD] if is_labelled else [])
    filtered_df = df[selected_cols].copy()

    # Check duplicate TransactionID
    if filtered_df["TransactionID"].duplicated().any():
        dup_count = filtered_df["TransactionID"].duplicated().sum()
        raise ValidationError(f"DataFrame contains {dup_count} duplicate TransactionID values.")

    # Validate TransactionID type & positivity
    if (filtered_df["TransactionID"] <= 0).any() or filtered_df["TransactionID"].isna().any():
        raise ValidationError("TransactionID must be strictly positive and non-null.")

    # Validate TransactionDT
    if (
        (filtered_df["TransactionDT"] < 0).any()
        or filtered_df["TransactionDT"].isna().any()
        or not np.isfinite(filtered_df["TransactionDT"]).all()
    ):
        raise ValidationError("TransactionDT must be finite, non-negative, and non-null.")

    # Validate TransactionAmt
    if (
        (filtered_df["TransactionAmt"] < 0).any()
        or filtered_df["TransactionAmt"].isna().any()
        or not np.isfinite(filtered_df["TransactionAmt"]).all()
    ):
        raise ValidationError("TransactionAmt must be finite, non-negative, and non-null.")

    # Validate ProductCD
    if filtered_df["ProductCD"].isna().any():
        raise ValidationError("ProductCD must not contain null or blank values.")

    # Validate target if labelled
    if is_labelled:
        valid_targets = filtered_df[TARGET_FIELD].isin([0, 1])
        if not valid_targets.all():
            raise ValidationError(
                f"Target '{TARGET_FIELD}' contains invalid values outside {{0, 1}}."
            )

    # Normalize address columns
    for addr_col in ["addr1", "addr2"]:
        filtered_df[addr_col] = filtered_df[addr_col].apply(normalize_address)

    return filtered_df, extra_cols


def validate_raw_file(
    input_path: str,
    config_path: str = "configs/project.json",
    report_path: Optional[str] = None,
) -> DataQualityReport:
    """Validate a raw transaction CSV file and generate a data quality report."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read header first to find columns
    header_df = pd.read_csv(input_path, nrows=5)
    is_labelled = TARGET_FIELD in header_df.columns

    df = pd.read_csv(input_path, low_memory=False)
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

    filtered_df, dropped_cols = validate_dataframe(
        df, is_labelled=is_labelled, allow_extra_columns=True
    )

    missing_rates = {col: float(filtered_df[col].isna().mean()) for col in filtered_df.columns}
    col_types = {col: str(filtered_df[col].dtype) for col in filtered_df.columns}

    fraud_count = int(filtered_df[TARGET_FIELD].sum()) if is_labelled else None
    fraud_prev = float(filtered_df[TARGET_FIELD].mean()) if is_labelled else None

    report = DataQualityReport(
        total_rows=len(filtered_df),
        selected_columns=list(filtered_df.columns),
        dropped_columns=dropped_cols,
        missing_rates=missing_rates,
        column_types=col_types,
        peak_memory_mb=round(memory_mb, 2),
        fraud_count=fraud_count,
        fraud_prevalence=round(fraud_prev, 5) if fraud_prev is not None else None,
        validation_passed=True,
        errors=[],
    )

    if report_path:
        out_path = Path(report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    return report
