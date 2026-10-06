"""Tests for benchmark and reproducibility verification modules."""

from fraudguard.benchmark import run_api_benchmark, run_reproducibility_test


def test_api_benchmark(tmp_path):
    out_file = tmp_path / "api_bench.json"
    res = run_api_benchmark(
        warmup_count=2,
        request_count=10,
        output_path=str(out_file),
    )
    assert res["successful_requests"] == 10
    assert "latency_ms" in res
    assert out_file.exists()


def test_reproducibility_test(tmp_path):
    out_file = tmp_path / "repro.json"
    res = run_reproducibility_test(
        artifact_dir="artifacts/champion",
        freeze_manifest_path="reports/freeze_manifest.json",
        split_manifest_path="data/processed/split_manifest.json",
        output_dir=str(out_file),
    )
    assert "reproducibility_tolerance_0_001_met" in res
    assert res["reproducibility_tolerance_0_001_met"] is True
    assert out_file.exists()
