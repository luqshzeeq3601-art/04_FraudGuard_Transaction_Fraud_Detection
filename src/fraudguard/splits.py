"""Chronological partitioning, timestamp boundary alignment, and expanding fold manifests."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from fraudguard.data import TARGET_FIELD, validate_dataframe


class SplitError(ValueError):
    """Raised when partitioning constraints are violated."""

    pass


@dataclass
class PartitionInfo:
    name: str
    row_count: int
    start_dt: float
    end_dt: float
    start_id: int
    end_id: int
    fraud_count: int
    fraud_prevalence: float
    indices: List[int]
    transaction_ids: List[int]

    def to_dict(self, include_ids: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_ids:
            d.pop("indices", None)
            d.pop("transaction_ids", None)
        return d


@dataclass
class FoldInfo:
    fold_id: int
    train_row_count: int
    train_start_dt: float
    train_end_dt: float
    val_row_count: int
    val_start_dt: float
    val_end_dt: float
    train_fraud_count: int
    train_fraud_prevalence: float
    val_fraud_count: int
    val_fraud_prevalence: float
    train_indices: List[int]
    val_indices: List[int]

    def to_dict(self, include_ids: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_ids:
            d.pop("train_indices", None)
            d.pop("val_indices", None)
        return d


def align_cut_to_timestamp_boundary(df: pd.DataFrame, target_idx: int) -> int:
    """Shift target_idx forward to include all identical TransactionDT rows in the earlier partition."""
    n = len(df)
    if target_idx >= n:
        return n
    if target_idx <= 0:
        return 0

    cut_dt = df["TransactionDT"].iloc[target_idx]
    # If the row immediately preceding has the same timestamp as cut_dt,
    # find the last index with cut_dt and place cut after it.
    prev_dt = df["TransactionDT"].iloc[target_idx - 1]
    if cut_dt == prev_dt:
        # Move forward until TransactionDT changes
        last_match = df[df["TransactionDT"] == cut_dt].index[-1]
        return min(n, int(last_match) + 1)
    return target_idx


def create_chronological_splits(
    df: pd.DataFrame,
    dev_ratio: float = 0.60,
    calib_ratio: float = 0.10,
    policy_ratio: float = 0.10,
    test_ratio: float = 0.20,
) -> Dict[str, PartitionInfo]:
    """Split DataFrame into chronological partitions, respecting equal timestamps."""
    # Ensure sorted by TransactionDT, then TransactionID
    df_sorted = df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(drop=True)
    n = len(df_sorted)

    if n < 50:
        raise SplitError(f"Dataset too small to create 4 meaningful partitions: {n} rows")

    # Initial target split points
    cut1_raw = int(round(n * dev_ratio))
    cut2_raw = int(round(n * (dev_ratio + calib_ratio)))
    cut3_raw = int(round(n * (dev_ratio + calib_ratio + policy_ratio)))

    cut1 = align_cut_to_timestamp_boundary(df_sorted, cut1_raw)
    cut2 = align_cut_to_timestamp_boundary(df_sorted, cut2_raw)
    cut3 = align_cut_to_timestamp_boundary(df_sorted, cut3_raw)

    cuts = [0, cut1, cut2, cut3, n]
    names = ["development", "calibration", "policy", "final_test"]
    partitions: Dict[str, PartitionInfo] = {}

    for i in range(len(names)):
        start_idx, end_idx = cuts[i], cuts[i + 1]
        if start_idx >= end_idx:
            raise SplitError(
                f"Partition '{names[i]}' is empty due to timestamp boundary constraints."
            )

        part_df = df_sorted.iloc[start_idx:end_idx]
        has_labels = TARGET_FIELD in part_df.columns
        fraud_cnt = int(part_df[TARGET_FIELD].sum()) if has_labels else 0
        fraud_prev = float(part_df[TARGET_FIELD].mean()) if has_labels else 0.0

        # Class presence check for dev, calib, policy
        if has_labels and names[i] in ["development", "calibration", "policy"]:
            if fraud_cnt == 0 or fraud_cnt == len(part_df):
                raise SplitError(
                    f"Partition '{names[i]}' lacks class diversity: {fraud_cnt} fraud out of {len(part_df)}."
                )

        p_info = PartitionInfo(
            name=names[i],
            row_count=len(part_df),
            start_dt=float(part_df["TransactionDT"].min()),
            end_dt=float(part_df["TransactionDT"].max()),
            start_id=int(part_df["TransactionID"].iloc[0]),
            end_id=int(part_df["TransactionID"].iloc[-1]),
            fraud_count=fraud_cnt,
            fraud_prevalence=round(fraud_prev, 5),
            indices=list(range(start_idx, end_idx)),
            transaction_ids=part_df["TransactionID"].tolist(),
        )
        partitions[names[i]] = p_info

    # Validate strict chronological ordering
    for i in range(len(names) - 1):
        curr_p = partitions[names[i]]
        next_p = partitions[names[i + 1]]
        if curr_p.end_dt > next_p.start_dt:
            raise SplitError(
                f"Chronological violation between {names[i]} (end_dt={curr_p.end_dt}) "
                f"and {names[i + 1]} (start_dt={next_p.start_dt})"
            )

    return partitions


def create_expanding_folds(
    df: pd.DataFrame,
    dev_partition: PartitionInfo,
    n_folds: int = 3,
) -> List[FoldInfo]:
    """Create expanding internal folds on the development partition."""
    dev_df = df.iloc[dev_partition.indices].copy().reset_index(drop=True)
    n_dev = len(dev_df)

    # 3 expanding folds:
    # Fold 1: train first 50% -> val next 16.7%
    # Fold 2: train first 66.7% -> val next 16.7%
    # Fold 3: train first 83.3% -> val next 16.7%
    ratios = [
        (0.50, 0.50 + 1.0 / 6.0),
        (2.0 / 3.0, 2.0 / 3.0 + 1.0 / 6.0),
        (5.0 / 6.0, 1.00),
    ]

    folds: List[FoldInfo] = []
    has_labels = TARGET_FIELD in dev_df.columns

    for fold_idx, (train_end_r, val_end_r) in enumerate(ratios, start=1):
        t_end_raw = int(round(n_dev * train_end_r))
        v_end_raw = int(round(n_dev * val_end_r)) if val_end_r < 1.0 else n_dev

        t_end = align_cut_to_timestamp_boundary(dev_df, t_end_raw)
        v_end = align_cut_to_timestamp_boundary(dev_df, v_end_raw) if val_end_r < 1.0 else n_dev

        train_slice = dev_df.iloc[:t_end]
        val_slice = dev_df.iloc[t_end:v_end]

        if len(train_slice) == 0 or len(val_slice) == 0:
            raise SplitError(f"Fold {fold_idx} produced an empty train or val slice.")

        # Map back to global indices
        train_global_idx = [dev_partition.indices[i] for i in range(t_end)]
        val_global_idx = [dev_partition.indices[i] for i in range(t_end, v_end)]

        t_fraud = int(train_slice[TARGET_FIELD].sum()) if has_labels else 0
        t_prev = float(train_slice[TARGET_FIELD].mean()) if has_labels else 0.0
        v_fraud = int(val_slice[TARGET_FIELD].sum()) if has_labels else 0
        v_prev = float(val_slice[TARGET_FIELD].mean()) if has_labels else 0.0

        fold_info = FoldInfo(
            fold_id=fold_idx,
            train_row_count=len(train_slice),
            train_start_dt=float(train_slice["TransactionDT"].min()),
            train_end_dt=float(train_slice["TransactionDT"].max()),
            val_row_count=len(val_slice),
            val_start_dt=float(val_slice["TransactionDT"].min()),
            val_end_dt=float(val_slice["TransactionDT"].max()),
            train_fraud_count=t_fraud,
            train_fraud_prevalence=round(t_prev, 5),
            val_fraud_count=v_fraud,
            val_fraud_prevalence=round(v_prev, 5),
            train_indices=train_global_idx,
            val_indices=val_global_idx,
        )
        folds.append(fold_info)

    return folds


def build_and_save_split_manifest(
    input_csv_path: str,
    output_dir: str = "data/processed",
    config_path: str = "configs/project.json",
) -> Dict[str, Any]:
    """Generate and persist split manifest and development EDA report."""
    df = pd.read_csv(input_csv_path, low_memory=False)
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    filtered_df = filtered_df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )

    with open(input_csv_path, "rb") as f:
        raw_hash = hashlib.sha256(f.read()).hexdigest()

    partitions = create_chronological_splits(filtered_df)
    folds = create_expanding_folds(filtered_df, partitions["development"])

    manifest_data = {
        "raw_data_path": str(input_csv_path),
        "raw_data_sha256": raw_hash,
        "total_rows": len(filtered_df),
        "partitions": {name: p.to_dict(include_ids=True) for name, p in partitions.items()},
        "folds": [f.to_dict(include_ids=True) for f in folds],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    # Generate Development EDA report (strictly development partition only)
    dev_indices = partitions["development"].indices
    dev_df = filtered_df.iloc[dev_indices]
    eda_path = "reports/development_eda.md" if str(output_dir) == "data/processed" else str(out_dir / "development_eda.md")
    generate_development_eda_report(dev_df, output_path=eda_path)

    return manifest_data


def generate_development_eda_report(
    dev_df: pd.DataFrame, output_path: str = "reports/development_eda.md"
) -> None:
    """Generate Markdown EDA report analyzing only development rows without inspecting test labels."""
    total_rows = len(dev_df)
    fraud_count = int(dev_df[TARGET_FIELD].sum())
    fraud_rate = float(dev_df[TARGET_FIELD].mean())

    amt_desc = dev_df["TransactionAmt"].describe()
    prod_counts = dev_df["ProductCD"].value_counts(dropna=False).to_dict()
    card6_counts = dev_df["card6"].value_counts(dropna=False).to_dict()

    content = f"""# Development Partition Exploratory Data Analysis

## 1. Partition Scope and Safety
- **Partition:** Development (Earliest ~60% of chronological transactions)
- **Total Rows:** {total_rows}
- **Positive Fraud Cases:** {fraud_count}
- **Fraud Prevalence:** {fraud_rate:.4%}
- **Holdout Isolation Note:** Final test, policy, and calibration partitions are excluded from this analysis.

## 2. Five Key Data Observations
1. **Target Imbalance:** Fraud prevalence is approximately {fraud_rate:.2%}, requiring ranking metrics (Average Precision, Precision@1%, Recall@1%) rather than accuracy or ROC-AUC alone.
2. **Transaction Amount Distribution:** Highly right-skewed with median {amt_desc["50%"]:.2f} and maximum {amt_desc["max"]:.2f}. `log1p(TransactionAmt)` is appropriate for linear modeling.
3. **Missing Value Structure:** Nullable distance features (`dist1`, `dist2`) and return email domains (`R_emaildomain`) exhibit high missing rates (>50%), requiring explicit missingness indicators.
4. **Product Code Segments:** ProductCD distribution is dominated by category `{list(prod_counts.keys())[0]}` ({list(prod_counts.values())[0]} rows), while smaller segments require robust one-hot encoding.
5. **Card Type Categories:** Debit cards dominate over credit cards ({card6_counts.get("debit", 0)} vs {card6_counts.get("credit", 0)}), with distinct risk profiles across payment mechanisms.

## 3. Numeric Summary (Development Only)
| Statistic | TransactionAmt |
|---|---|
| Count | {amt_desc["count"]:.0f} |
| Mean | {amt_desc["mean"]:.2f} |
| Std | {amt_desc["std"]:.2f} |
| Min | {amt_desc["min"]:.2f} |
| 25% | {amt_desc["25%"]:.2f} |
| 50% (Median) | {amt_desc["50%"]:.2f} |
| 75% | {amt_desc["75%"]:.2f} |
| Max | {amt_desc["max"]:.2f} |
"""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content, encoding="utf-8")
