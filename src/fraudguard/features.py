"""Feature extraction, training-only imputation, missingness indicators, and encoding."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudguard.data import (
    CATEGORICAL_PREDICTORS,
    NUMERIC_PREDICTORS,
    normalize_address,
)


@dataclass
class PreprocessorState:
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    categorical_categories: Dict[str, List[str]] = field(default_factory=dict)
    encoded_feature_names: List[str] = field(default_factory=list)
    raw_feature_mapping: Dict[str, List[str]] = field(default_factory=dict)


class FeaturePreprocessor:
    """Preprocesses raw transaction inputs with training-only fitted statistics."""

    def __init__(self, scale_numeric: bool = True):
        self.scale_numeric = scale_numeric
        self.numeric_medians: Dict[str, float] = {}
        self.mean_amt_per_product: Dict[str, float] = {}
        self.global_mean_amt: float = 100.0
        self.encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.encoded_feature_names: List[str] = []
        self.raw_feature_mapping: Dict[str, List[str]] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit imputation statistics, one-hot encoders, and scalers on training rows only."""
        # 1. Compute numeric medians
        for col in NUMERIC_PREDICTORS:
            valid_vals = df[col].dropna()
            if len(valid_vals) > 0:
                self.numeric_medians[col] = float(valid_vals.median())
            else:
                self.numeric_medians[col] = 0.0

        # 2. Compute ProductCD amount statistics
        if "ProductCD" in df.columns and "TransactionAmt" in df.columns:
            self.global_mean_amt = float(df["TransactionAmt"].mean())
            prod_means = df.groupby("ProductCD")["TransactionAmt"].mean().to_dict()
            self.mean_amt_per_product = {str(k): float(v) for k, v in prod_means.items()}

        # 3. Extract numeric array and log1p transform
        num_features = self._extract_numeric_features(df)
        if self.scale_numeric:
            self.scaler = StandardScaler()
            self.scaler.fit(num_features)

        # 4. Categoricals encoding
        cat_df = self._clean_categoricals(df)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(cat_df)

        # Build feature names and mapping
        self._build_feature_names_and_mapping()
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform inputs using fitted statistics without data leakage."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before transforming data.")

        num_features = self._extract_numeric_features(df)
        if self.scale_numeric and self.scaler is not None:
            num_features = self.scaler.transform(num_features)

        cat_df = self._clean_categoricals(df)
        cat_features = self.encoder.transform(cat_df)

        return np.hstack([num_features, cat_features])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _extract_numeric_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract continuous features, cyclical hour, domain match, and interaction ratios."""
        amt = df["TransactionAmt"].to_numpy().astype(float)
        log_amt = np.log1p(np.clip(amt, 0.0, None))

        dist1_raw = df["dist1"].to_numpy().astype(float)
        dist1_missing = np.isnan(dist1_raw).astype(float)
        dist1_imputed = np.where(
            np.isnan(dist1_raw), self.numeric_medians.get("dist1", 0.0), dist1_raw
        )

        dist2_raw = df["dist2"].to_numpy().astype(float)
        dist2_missing = np.isnan(dist2_raw).astype(float)
        dist2_imputed = np.where(
            np.isnan(dist2_raw), self.numeric_medians.get("dist2", 0.0), dist2_raw
        )

        # Temporal hour cyclical features if TransactionDT is present
        if "TransactionDT" in df.columns:
            dt_vals = df["TransactionDT"].to_numpy().astype(float)
            hours = (dt_vals % 86400) / 3600.0
            hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
            hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
        else:
            hour_sin = np.zeros(len(df), dtype=float)
            hour_cos = np.zeros(len(df), dtype=float)

        # Email domains match indicator
        p_email = df["P_emaildomain"].fillna("").astype(str).to_numpy()
        r_email = df["R_emaildomain"].fillna("").astype(str).to_numpy()
        email_match = ((p_email == r_email) & (p_email != "")).astype(float)

        # ProductCD amount ratio relative to fitted training mean
        mean_map = getattr(self, "mean_amt_per_product", {})
        global_mean = getattr(self, "global_mean_amt", 100.0)
        if "ProductCD" in df.columns and mean_map:
            expected_amt = (
                df["ProductCD"]
                .astype(str)
                .map(mean_map)
                .fillna(global_mean)
                .to_numpy()
                .astype(float)
            )
            amt_to_prod_ratio = amt / np.clip(expected_amt, 1.0, None)
        else:
            amt_to_prod_ratio = amt / max(1.0, global_mean)

        return np.column_stack(
            [
                amt,
                log_amt,
                dist1_imputed,
                dist1_missing,
                dist2_imputed,
                dist2_missing,
                hour_sin,
                hour_cos,
                email_match,
                amt_to_prod_ratio,
            ]
        )

    def _clean_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize categoricals, replacing nulls with a reserved token."""
        cat_dict = {}
        for col in CATEGORICAL_PREDICTORS:
            if col in ["addr1", "addr2"]:
                vals = df[col].apply(normalize_address).fillna("__MISSING__").astype(str)
            else:
                vals = df[col].fillna("__MISSING__").astype(str)
            cat_dict[col] = vals
        return pd.DataFrame(cat_dict)

    def _build_feature_names_and_mapping(self) -> None:
        """Construct descriptive output feature names and track origin raw predictors."""
        numeric_names = [
            "TransactionAmt",
            "log1p_TransactionAmt",
            "dist1_imputed",
            "dist1_is_missing",
            "dist2_imputed",
            "dist2_is_missing",
            "hour_of_day_sin",
            "hour_of_day_cos",
            "email_domains_match",
            "amt_to_product_mean_ratio",
        ]

        self.raw_feature_mapping = {
            "TransactionAmt": [
                "TransactionAmt",
                "log1p_TransactionAmt",
                "amt_to_product_mean_ratio",
            ],
            "dist1": ["dist1_imputed", "dist1_is_missing"],
            "dist2": ["dist2_imputed", "dist2_is_missing"],
            "TransactionDT": ["hour_of_day_sin", "hour_of_day_cos"],
            "P_emaildomain": ["email_domains_match"],
            "R_emaildomain": ["email_domains_match"],
            "ProductCD": ["amt_to_product_mean_ratio"],
        }

        cat_feature_names = list(self.encoder.get_feature_names_out(CATEGORICAL_PREDICTORS))
        for raw_col in CATEGORICAL_PREDICTORS:
            matching = [f for f in cat_feature_names if f.startswith(f"{raw_col}_")]
            if raw_col in self.raw_feature_mapping:
                self.raw_feature_mapping[raw_col].extend(matching)
            else:
                self.raw_feature_mapping[raw_col] = matching

        self.encoded_feature_names = numeric_names + cat_feature_names

    def get_feature_names(self) -> List[str]:
        return list(self.encoded_feature_names)

    def get_raw_mapping(self) -> Dict[str, List[str]]:
        return dict(self.raw_feature_mapping)
