"""Shared inference and scoring pipeline for real-time API and batch workloads."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from fraudguard.artifacts import LoadedBundle, load_bundle
from fraudguard.data import validate_dataframe, validate_single_transaction
from fraudguard.policy import apply_review_policy, calculate_review_capacity


class FraudScorer:
    """Shared scoring pipeline ensuring identical prediction logic between API and batch."""

    def __init__(self, bundle: LoadedBundle):
        self.bundle = bundle
        self.preprocessor = bundle.preprocessor
        self.model = bundle.model
        self.calibrator = bundle.calibrator
        self.policy = bundle.policy
        self.manifest = bundle.manifest

        self.model_version = self.manifest.get("model_version", "fraudguard-v1")
        self.schema_version = bundle.feature_schema.get("schema_version", "transaction-v1")
        self.policy_version = self.policy.get("policy_version", "review-v1")
        self.score_type = (
            "calibrated_probability"
            if bundle.manifest.get("is_calibrated", False)
            else "raw_probability"
        )
        self.cutoff = float(self.policy.get("cutoff", 0.5))
        self.default_fraction = float(self.policy.get("default_review_fraction", 0.01))

    @classmethod
    def from_directory(cls, artifact_dir: str) -> "FraudScorer":
        bundle = load_bundle(artifact_dir)
        return cls(bundle)

    def predict_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features and generate probabilities (calibrated if enabled)."""
        X = self.preprocessor.transform(df)

        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "predict"):
            raw_probs = self.model.predict(X)
        else:
            raise RuntimeError("Model does not support predict_proba or predict.")

        if self.calibrator is not None:
            if hasattr(self.calibrator, "predict_proba"):
                # Calibrator takes raw probs or X
                try:
                    calib_probs = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
                except Exception:
                    calib_probs = self.calibrator.predict_proba(X)[:, 1]
            else:
                calib_probs = raw_probs
            return np.clip(calib_probs, 0.0, 1.0)

        return np.clip(raw_probs, 0.0, 1.0)

    def score_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and score a single transaction dictionary for the REST API."""
        cleaned = validate_single_transaction(payload)
        df_single = pd.DataFrame([cleaned])

        prob = float(self.predict_probabilities(df_single)[0])
        review_rec = bool(prob >= self.cutoff)

        return {
            "TransactionID": int(cleaned["TransactionID"]),
            "fraud_score": round(prob, 4),
            "score_type": self.score_type,
            "review_recommended": review_rec,
            "model_version": self.model_version,
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
        }

    def score_dataframe(
        self,
        df: pd.DataFrame,
        review_fraction: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Batch score unlabelled DataFrame and construct capped review queue."""
        filtered_df, _ = validate_dataframe(df, is_labelled=False, allow_extra_columns=False)
        q = review_fraction if review_fraction is not None else self.default_fraction
        n = len(filtered_df)

        probs = self.predict_probabilities(filtered_df)
        ids = filtered_df["TransactionID"].to_numpy()

        capacity = calculate_review_capacity(n, q)
        selected_mask, ranks = apply_review_policy(
            probs, ids, cutoff=self.cutoff, capacity=capacity
        )

        scored_df = pd.DataFrame(
            {
                "TransactionID": filtered_df["TransactionID"],
                "TransactionDT": filtered_df["TransactionDT"],
                "fraud_score": np.round(probs, 4),
                "score_type": self.score_type,
                "review_recommended": probs >= self.cutoff,
                "rank": ranks,
                "selected_for_review": selected_mask,
                "model_version": self.model_version,
                "schema_version": self.schema_version,
                "policy_version": self.policy_version,
            }
        )

        # Sort scored_df by rank
        scored_df = scored_df.sort_values(by="rank").reset_index(drop=True)

        # Review queue contains only selected candidates
        queue_df = scored_df[scored_df["selected_for_review"]].copy().reset_index(drop=True)

        return scored_df, queue_df
