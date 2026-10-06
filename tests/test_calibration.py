"""Unit tests for independent probability calibration."""

import numpy as np

from fraudguard.calibration import (
    SigmoidCalibrator,
    evaluate_calibration_decision,
)


def test_sigmoid_calibrator_fitting():
    calib = SigmoidCalibrator()
    raw_probs = np.array([0.1, 0.2, 0.4, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1])

    calib.fit(raw_probs, y)
    assert calib.is_fitted is True

    test_p = np.array([0.15, 0.85])
    probs = calib.predict_proba(test_p)
    assert probs.shape == (2, 2)
    assert probs[1, 1] > probs[0, 1]


def test_calibration_decision_logic():
    y = np.array([0, 0, 0, 1, 1])
    ids = np.arange(1, 6)
    # Calibrated probabilities clearly better calibrated
    raw_p = np.array([0.4, 0.4, 0.4, 0.6, 0.6])
    calib_p = np.array([0.05, 0.05, 0.05, 0.95, 0.95])

    res = evaluate_calibration_decision(y, ids, raw_p, calib_p)
    assert res["decision"] == "calibrated_probability"
    assert res["use_calibration"] is True


def test_sigmoid_calibrator_edge_cases():
    calib = SigmoidCalibrator()
    raw_probs = np.array([0.0, 0.5, 1.0])
    y = np.array([0, 0, 1])

    calib.fit(raw_probs, y)
    assert calib.is_fitted is True
    p = calib.predict_proba(np.array([0.1, 0.9]))
    assert p.shape == (2, 2)
    assert 0.0 <= p[0, 1] <= 1.0
