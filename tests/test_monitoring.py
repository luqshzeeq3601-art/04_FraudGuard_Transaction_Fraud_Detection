"""Unit tests for drift monitoring and PSI calculation."""

import numpy as np

from fraudguard.monitoring import calculate_psi, generate_monitoring_report


def test_calculate_psi_identical():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.2, 0.3, 0.5])
    psi = calculate_psi(p, q)
    assert abs(psi) < 1e-4


def test_calculate_psi_shifted():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.1, 0.2, 0.7])
    psi = calculate_psi(p, q)
    assert psi > 0.3  # significant drift


def test_generate_monitoring_report(tmp_path):
    out_dir = tmp_path / "monitoring"
    report = generate_monitoring_report(
        input_batch_path="examples/synthetic_batch.csv",
        artifact_dir="artifacts/champion",
        output_dir=str(out_dir),
    )

    assert "batch_row_count" in report
    assert "missingness" in report
    assert "fraud_score_monitoring" in report
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.html").exists()
