"""Isolation and integrity guard tests to ensure test runs do not alter artifacts or reports."""

import json
from pathlib import Path

from fraudguard.artifacts import compute_file_sha256


def test_artifacts_freeze_integrity_guard():
    """Verify that artifacts on disk match reports/freeze_manifest.json exactly."""
    manifest_path = Path("reports/freeze_manifest.json")
    if not manifest_path.exists():
        return

    with open(manifest_path, "r") as f:
        freeze_data = json.load(f)

    # Check champion files
    champ_hashes = freeze_data.get("champion", {}).get("file_hashes", {})
    champ_dir = Path("artifacts/champion")
    for fname, expected_h in champ_hashes.items():
        actual_h = compute_file_sha256(champ_dir / fname)
        assert actual_h == expected_h, f"Integrity failure in {fname}: {actual_h} != {expected_h}"

    # Check baseline LR files if frozen
    lr_hashes = freeze_data.get("baseline_lr_reference", {}).get("file_hashes", {})
    lr_dir = Path("artifacts/baselines/logistic_reference")
    for fname, expected_h in lr_hashes.items():
        actual_h = compute_file_sha256(lr_dir / fname)
        assert actual_h == expected_h, (
            f"Integrity failure in LR {fname}: {actual_h} != {expected_h}"
        )
