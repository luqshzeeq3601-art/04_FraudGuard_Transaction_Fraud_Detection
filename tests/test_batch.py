"""Unit tests for batch scoring and review queue construction."""

import pandas as pd
import pytest

from fraudguard.data import ValidationError
from fraudguard.scoring import FraudScorer


def test_batch_scoring_pipeline():
    scorer = FraudScorer.from_directory("artifacts/champion")
    df = pd.read_csv("examples/synthetic_batch.csv")

    scored_df, queue_df = scorer.score_dataframe(df, review_fraction=0.02)
    assert len(scored_df) == len(df)
    assert "fraud_score" in scored_df.columns
    assert "rank" in scored_df.columns
    assert "selected_for_review" in scored_df.columns

    # Ranks should be 1..N
    assert sorted(scored_df["rank"].tolist()) == list(range(1, len(df) + 1))

    # Queue contains only selected items
    assert len(queue_df) == int(scored_df["selected_for_review"].sum())
    assert (queue_df["fraud_score"] >= scorer.cutoff).all()


def test_batch_scoring_duplicate_rejection():
    scorer = FraudScorer.from_directory("artifacts/champion")
    df = pd.read_csv("examples/synthetic_batch.csv").copy()
    df.loc[1, "TransactionID"] = df.loc[0, "TransactionID"]

    with pytest.raises(ValidationError, match="duplicate TransactionID"):
        scorer.score_dataframe(df)
