"""Unit tests verifying the Dockerfile and container packaging contract."""

from pathlib import Path


def test_dockerfile_and_ignore_contract():
    dockerfile = Path("Dockerfile")
    dockerignore = Path(".dockerignore")

    assert dockerfile.exists()
    assert dockerignore.exists()

    df_content = dockerfile.read_text(encoding="utf-8")
    assert "FROM python:3.10-slim" in df_content
    assert "EXPOSE 8000" in df_content
    assert "FRAUDGUARD_ARTIFACT_DIR=/app/artifacts/champion" in df_content
    assert "uvicorn" in df_content

    di_content = dockerignore.read_text(encoding="utf-8")
    assert "data/" in di_content
    assert "artifacts/" in di_content
    assert ".venv/" in di_content
    assert "reports/" in di_content
