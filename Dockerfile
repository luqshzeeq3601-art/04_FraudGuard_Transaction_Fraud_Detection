FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy package requirements and metadata
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies and package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8000

ENV FRAUDGUARD_ARTIFACT_DIR=/app/artifacts/champion

CMD ["uvicorn", "fraudguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
