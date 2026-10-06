"""Unit tests for metrics and policy helpers."""

import numpy as np
import pytest

from fraudguard.metrics import (
    compute_average_precision,
    compute_simulated_cost,
    compute_top_k_metrics,
)
from fraudguard.policy import (
    apply_review_policy,
    calculate_review_capacity,
    determine_policy_cutoff,
)


def test_calculate_review_capacity():
    assert calculate_review_capacity(100, 0.01) == 1
    assert calculate_review_capacity(105, 0.01) == 2  # ceil(1.05) = 2
    assert calculate_review_capacity(1, 0.01) == 1
    assert calculate_review_capacity(50, 0.05) == 3  # ceil(2.5) = 3
    with pytest.raises(ValueError):
        calculate_review_capacity(0, 0.01)
    with pytest.raises(ValueError):
        calculate_review_capacity(100, 0.0)


def test_average_precision_hand_computed():
    # 5 samples: true = [1, 1, 0, 1, 0], predicted scores = [0.9, 0.8, 0.7, 0.6, 0.1]
    # Rankings:
    # 1: score 0.9, y=1 -> Precision=1/1, Recall=1/3
    # 2: score 0.8, y=1 -> Precision=2/2, Recall=2/3
    # 3: score 0.7, y=0 -> Precision=2/3, Recall=2/3
    # 4: score 0.6, y=1 -> Precision=3/4, Recall=3/3
    # AP = 1/3 * (1/1 + 2/2 + 3/4) = 1/3 * (1 + 1 + 0.75) = 2.75 / 3 = 0.916666...
    y_true = np.array([1, 1, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    ap = compute_average_precision(y_true, y_score)
    assert abs(ap - (2.75 / 3.0)) < 1e-5


def test_top_k_metrics_hand_computed():
    # 10 samples, 2 fraud: IDs 1..10
    # True fraud at ID 2 (score 0.8) and ID 5 (score 0.9)
    y_true = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    y_score = np.array([0.1, 0.8, 0.2, 0.3, 0.9, 0.05, 0.4, 0.5, 0.15, 0.25])
    ids = np.arange(1, 11)

    # At q=0.1 (10%), K=ceil(0.1*10)=1. Top 1 is ID 5 (score 0.9, y=1).
    # Precision@10% = 1/1 = 1.0, Recall@10% = 1/2 = 0.5, Prevalence = 0.2, Lift = 5.0
    res = compute_top_k_metrics(y_true, y_score, ids, fractions=[0.1, 0.2])
    f1 = res["by_fraction"]["0.1"]
    assert f1["k"] == 1
    assert f1["selected_fraud"] == 1
    assert f1["precision"] == 1.0
    assert f1["recall"] == 0.5
    assert f1["lift"] == 5.0

    # At q=0.2 (20%), K=2. Top 2 are ID 5 (score 0.9) and ID 2 (score 0.8) - both fraud!
    # Precision@20% = 2/2 = 1.0, Recall@20% = 2/2 = 1.0, Lift = 5.0
    f2 = res["by_fraction"]["0.2"]
    assert f2["k"] == 2
    assert f2["selected_fraud"] == 2
    assert f2["precision"] == 1.0
    assert f2["recall"] == 1.0


def test_simulated_cost_hand_computed():
    # 10 samples, 2 fraud total
    # Selected mask selects 2 samples: 1 true fraud (TP), 1 legitimate (FP)
    # Total fraud = 2 -> Missed fraud (FN) = 1
    # Cost = FN * 100 + Selected * 1 + FP * 2
    # Cost = 1 * 100 + 2 * 1 + 1 * 2 = 104.0
    y_true = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    mask = np.array([False, True, True, False, False, False, False, False, False, False])
    cost_res = compute_simulated_cost(
        y_true, mask, missed_fraud_cost=100.0, review_cost=1.0, legitimate_friction_cost=2.0
    )

    assert cost_res["simulated_cost"] == 104.0
    assert cost_res["cost_review_none"] == 200.0
    assert cost_res["cost_savings"] == 96.0


def test_policy_cutoff_and_ties():
    # 4 samples with tied scores: scores = [0.8, 0.8, 0.5, 0.2], IDs = [102, 101, 103, 104]
    # Sorted order (desc score, asc ID):
    # Rank 1: ID 101 (score 0.8)
    # Rank 2: ID 102 (score 0.8)
    # Rank 3: ID 103 (score 0.5)
    # Rank 4: ID 104 (score 0.2)
    scores = np.array([0.8, 0.8, 0.5, 0.2])
    ids = np.array([102, 101, 103, 104])

    # For q=0.25 (capacity 1): cutoff tau is score at rank 1 = 0.8
    tau = determine_policy_cutoff(scores, ids, fraction=0.25)
    assert tau == 0.8

    # Apply policy with capacity 1: only ID 101 should be selected (tie broken by smaller ID)
    mask, ranks = apply_review_policy(scores, ids, cutoff=tau, capacity=1)
    assert list(ranks) == [2, 1, 3, 4]
    # mask corresponds to original order: [102, 101, 103, 104]
    assert not mask[0]
    assert bool(mask[1]) is True
    assert not mask[2]
    assert not mask[3]
