FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy package requirements and metadata
COPY pyproject.toml README.md ./
COPY src/ ./src/
# Trusted champion bundles are supplied at runtime through a read-only mount.

# Install dependencies and package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    mkdir -p /app/artifacts/champion && \
    useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENV FRAUDGUARD_ARTIFACT_DIR=/app/artifacts/champion

CMD ["uvicorn", "fraudguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
