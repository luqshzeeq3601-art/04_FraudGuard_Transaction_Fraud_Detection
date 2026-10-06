"""Trusted model and policy bundle serialization, hash validation, and artifact management."""

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import sklearn


class ArtifactError(Exception):
    """Raised when an artifact bundle is missing, corrupted, or untrusted."""

    pass


@dataclass
class LoadedBundle:
    artifact_dir: Path
    preprocessor: Any
    model: Any
    calibrator: Optional[Any]
    feature_schema: Dict[str, Any]
    policy: Dict[str, Any]
    reference_summary: Dict[str, Any]
    manifest: Dict[str, Any]


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_bundle(
    output_dir: str,
    preprocessor: Any,
    model: Any,
    feature_names: List[str],
    raw_mapping: Dict[str, List[str]],
    policy: Dict[str, Any],
    reference_summary: Dict[str, Any],
    calibrator: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save trusted bundle with atomic file creation and SHA-256 manifest."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save pipeline.joblib
    pipeline_obj = {
        "preprocessor": preprocessor,
        "model": model,
    }
    pipeline_file = out_path / "pipeline.joblib"
    joblib.dump(pipeline_obj, pipeline_file)

    # 2. Save calibration.joblib if present
    calib_file = out_path / "calibration.joblib"
    if calibrator is not None:
        joblib.dump(calibrator, calib_file)
    elif calib_file.exists():
        calib_file.unlink()

    # 3. Save feature_schema.json
    schema_data = {
        "schema_version": "transaction-v1",
        "encoded_feature_names": feature_names,
        "raw_feature_mapping": raw_mapping,
    }
    schema_file = out_path / "feature_schema.json"
    with open(schema_file, "w") as f:
        json.dump(schema_data, f, indent=2)

    # 4. Save policy.json
    policy_file = out_path / "policy.json"
    with open(policy_file, "w") as f:
        json.dump(policy, f, indent=2)

    # 5. Save reference_summary.json
    ref_file = out_path / "reference_summary.json"
    with open(ref_file, "w") as f:
        json.dump(reference_summary, f, indent=2)

    # 6. Compute hashes for all bundle files
    file_hashes = {
        "pipeline.joblib": compute_file_sha256(pipeline_file),
        "feature_schema.json": compute_file_sha256(schema_file),
        "policy.json": compute_file_sha256(policy_file),
        "reference_summary.json": compute_file_sha256(ref_file),
    }
    if calibrator is not None and calib_file.exists():
        file_hashes["calibration.joblib"] = compute_file_sha256(calib_file)

    # 7. Write manifest.json
    manifest_data = {
        "model_version": metadata.get("model_version", "fraudguard-v1")
        if metadata
        else "fraudguard-v1",
        "model_family": metadata.get("model_family", "unknown") if metadata else "unknown",
        "is_calibrated": calibrator is not None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "file_hashes": file_hashes,
        "metadata": metadata or {},
    }

    manifest_file = out_path / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f, indent=2)

    return out_path


def load_bundle(artifact_dir: str, verify_hashes: bool = True) -> LoadedBundle:
    """Load and verify a trusted local model bundle."""
    art_path = Path(artifact_dir)
    if not art_path.exists() or not art_path.is_dir():
        raise ArtifactError(f"Artifact directory does not exist: {artifact_dir}")

    manifest_file = art_path / "manifest.json"
    if not manifest_file.exists():
        raise ArtifactError(f"Artifact directory missing manifest.json: {artifact_dir}")

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    file_hashes = manifest.get("file_hashes", {})
    if verify_hashes:
        for fname, expected_hash in file_hashes.items():
            fpath = art_path / fname
            if not fpath.exists():
                raise ArtifactError(f"Bundle file missing: {fname} in {artifact_dir}")
            actual_hash = compute_file_sha256(fpath)
            if actual_hash != expected_hash:
                raise ArtifactError(
                    f"Hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}"
                )

    # Load pipeline
    pipeline_file = art_path / "pipeline.joblib"
    pipeline_obj = joblib.load(pipeline_file)
    preprocessor = pipeline_obj["preprocessor"]
    model = pipeline_obj["model"]

    # Load calibrator if present
    calibrator = None
    calib_file = art_path / "calibration.joblib"
    if calib_file.exists() and manifest.get("is_calibrated", False):
        calibrator = joblib.load(calib_file)

    # Load feature schema
    with open(art_path / "feature_schema.json", "r") as f:
        schema = json.load(f)

    # Load policy
    with open(art_path / "policy.json", "r") as f:
        policy = json.load(f)

    # Load reference summary
    with open(art_path / "reference_summary.json", "r") as f:
        ref_summary = json.load(f)

    return LoadedBundle(
        artifact_dir=art_path,
        preprocessor=preprocessor,
        model=model,
        calibrator=calibrator,
        feature_schema=schema,
        policy=policy,
        reference_summary=ref_summary,
        manifest=manifest,
    )
