"""Unit tests for training pipelines and baseline models."""

import pandas as pd

from fraudguard.data import validate_dataframe
from fraudguard.splits import create_chronological_splits, create_expanding_folds
from fraudguard.train import (
    AmountClassifier,
    PriorClassifier,
    run_fold_cross_validation,
)


def test_prior_classifier():
    clf = PriorClassifier()
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [0, 0, 1, 1]
    clf.fit(X, y)
    assert clf.prior_ == 0.5
    probs = clf.predict_proba(X)
    assert probs.shape == (4, 2)
    assert (probs[:, 1] == 0.5).all()


def test_amount_classifier():
    clf = AmountClassifier()
    X = [[10.0], [20.0], [100.0]]
    y = [0, 0, 1]
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs[2, 1] > probs[0, 1]


def test_run_fold_cv_logistic():
    df = pd.read_csv("tests/fixtures/transactions.csv")
    filtered_df, _ = validate_dataframe(df, is_labelled=True, allow_extra_columns=True)
    partitions = create_chronological_splits(filtered_df)
    folds = create_expanding_folds(filtered_df, partitions["development"])
    folds_info = [f.to_dict(include_ids=True) for f in folds]

    res = run_fold_cross_validation(filtered_df, folds_info, "logistic")
    assert len(res["folds"]) == 3
    assert res["mean_average_precision"] >= 0.0
    assert res["mean_recall_at_1pct"] >= 0.0
