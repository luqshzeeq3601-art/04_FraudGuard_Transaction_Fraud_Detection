"""Unit tests for artifact saving/loading and shared scorer."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraudguard.artifacts import ArtifactError, load_bundle, save_bundle
from fraudguard.features import FeaturePreprocessor
from fraudguard.scoring import FraudScorer


def test_bundle_save_load_and_hash_integrity(tmp_path):
    df = pd.read_csv("tests/fixtures/transactions.csv")
    preprocessor = FeaturePreprocessor(scale_numeric=True)
    X = preprocessor.fit_transform(df)
    y = df["isFraud"].to_numpy()

    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    policy = {"policy_version": "review-v1", "cutoff": 0.5, "default_review_fraction": 0.01}
    ref_summary = {"reference_row_count": len(df), "numeric": {}, "categorical": {}}

    art_dir = tmp_path / "bundle"
    save_bundle(
        output_dir=str(art_dir),
        preprocessor=preprocessor,
        model=clf,
        feature_names=preprocessor.get_feature_names(),
        raw_mapping=preprocessor.get_raw_mapping(),
        policy=policy,
        reference_summary=ref_summary,
    )

    loaded = load_bundle(str(art_dir), verify_hashes=True)
    assert loaded.manifest["is_calibrated"] is False
    assert (art_dir / "manifest.json").exists()

    # Corrupt a file and verify that load_bundle fails
    policy_file = art_dir / "policy.json"
    with open(policy_file, "a") as f:
        f.write(" ")  # alter hash

    with pytest.raises(ArtifactError, match="Hash mismatch"):
        load_bundle(str(art_dir), verify_hashes=True)


def test_scorer_single_and_batch_parity(tmp_path):
    df = pd.read_csv("tests/fixtures/transactions.csv")
    preprocessor = FeaturePreprocessor(scale_numeric=True)
    X = preprocessor.fit_transform(df)
    y = df["isFraud"].to_numpy()

    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    policy = {"policy_version": "review-v1", "cutoff": 0.5, "default_review_fraction": 0.01}
    ref_summary = {"reference_row_count": len(df), "numeric": {}, "categorical": {}}

    art_dir = tmp_path / "test_scorer_bundle"
    save_bundle(
        output_dir=str(art_dir),
        preprocessor=preprocessor,
        model=clf,
        feature_names=preprocessor.get_feature_names(),
        raw_mapping=preprocessor.get_raw_mapping(),
        policy=policy,
        reference_summary=ref_summary,
    )

    scorer = FraudScorer.from_directory(str(art_dir))

    # Single scoring
    first_row_dict = df.drop(columns=["isFraud"]).iloc[0].to_dict()
    single_res = scorer.score_single(first_row_dict)

    # Batch scoring
    df_unlabelled = df.drop(columns=["isFraud"]).copy()
    scored_df, queue_df = scorer.score_dataframe(df_unlabelled, review_fraction=0.01)

    batch_first_score = scored_df.loc[
        scored_df["TransactionID"] == first_row_dict["TransactionID"], "fraud_score"
    ].iloc[0]

    assert abs(single_res["fraud_score"] - batch_first_score) < 1e-4
    assert 0.0 <= single_res["fraud_score"] <= 1.0
    assert isinstance(single_res["review_recommended"], bool)
