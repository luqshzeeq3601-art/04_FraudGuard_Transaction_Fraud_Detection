# FraudGuard: Transaction Fraud Detection & Capped Review System

[![FraudGuard CI](https://github.com/luqshzeeq3601-art/04_FraudGuard_Transaction_Fraud_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/luqshzeeq3601-art/04_FraudGuard_Transaction_Fraud_Detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Render: Live API](https://img.shields.io/badge/Render-Live%20API-teal.svg)](https://fraudguard-api.onrender.com/docs)

## 1. Purpose

A fraud-review team can inspect only a small fraction of transactions. A classifier that predicts the majority class can look accurate while giving that team nothing useful. FraudGuard ranks transactions by fraud risk and fills a **review queue capped at 1% of volume**, then measures how much fraud that queue captures against simple baselines.

It is a production-grade ML engineering pipeline: chronological splits, strict schema validation on 12 allowlisted transaction fields, temporal cross-validation, independent probability calibration, a pre-test policy freeze, explanations, a scoring API, batch queue generation, drift monitoring, and live deployment on Render.

> [!NOTE]
> **Dataset status and real-world evaluation.**
> Retrained and evaluated on the official **IEEE-CIS Fraud Detection Benchmark** (`train_transaction.csv`, 590,540 rows spanning 182.5 days). Chronological partitions: Development 354,324 rows (11,988 fraud, 3.38%), Calibration 59,054 (2,550 fraud, 4.32%), Policy 59,054 (2,061 fraud, 3.49%), Final Test Holdout 118,108 rows (4,064 fraud, 3.44%). See [`data/raw/DATA_PROVENANCE.md`](data/raw/DATA_PROVENANCE.md).

## 2. Measured Results

### Final Holdout (`final_test`: 118,108 transactions, 4,064 fraud, 3.44% prevalence)

| Model | AP | 95% bootstrap CI (AP) | ROC-AUC | Precision@1% | Recall@1% | Lift@1% | Brier | Simulated cost | Savings vs no review |
|---|---|---|---|---|---|---|---|---|---|
| **Champion: LightGBM (selected)** | **0.1809** | **[0.1642, 0.1996]** | **0.8028** | **0.3443** | **0.1001** | **10.01x** | **0.0306** | **370,181** | **+36,219 (+8.9%)** |
| Logistic Regression reference (C=1.0) | 0.1487 | [0.1332, 0.1654] | 0.7760 | 0.3071 | 0.0893 | 8.93x | 0.0312 | 374,246 | +32,154 (+7.9%) |
| Amount-rank baseline | 0.0365 | [0.0301, 0.0435] | 0.4797 | 0.0288 | 0.0084 | 0.84x | 0.0352 | 406,478 | -78 (-0.0%) |
| Constant-prior reference | 0.0344 | [0.0344, 0.0344] | 0.5000 | 0.0567 | 0.0165 | 1.65x | 0.0332 | 406,400 | 0 |

Source: [`reports/final/final_evaluation.json`](reports/final/final_evaluation.json). Costs are evaluated using business cost units (missed fraud 100, review friction 1, legitimate review friction 2). Champion LightGBM delivers +4,065 cost savings over the baseline Logistic Regression reference.

### Engineering & Quality Checks

| Check | Result | Evidence |
|---|---|---|
| API latency, 1,000 warm requests to `/v1/score` | p50 8.58 ms, p95 9.67 ms, p99 11.23 ms (target p95 <= 100 ms) | [`reports/api_benchmark.json`](reports/api_benchmark.json) |
| Retraining reproducibility | 0.000000 AP and Brier difference (tolerance 0.001) | [`reports/repeatability/reproducibility.json`](reports/repeatability/reproducibility.json) |
| Tests and coverage | 50 tests passing, 84% coverage (target >= 80%) | `pytest --cov` |
| Test isolation | Suite leaves `artifacts/`, `reports/` and `data/` unchanged | `tests/test_isolation_guard.py` |
| SHAP additivity | Max error 1.15e-13 (target <= 1e-4) | `reports/explanations/` |
| Bootstrap uncertainty | 1,000 block bootstrap replicates over chronological test blocks | `reports/final/final_evaluation.json` |

## 3. Scoring API deployment configuration

FraudGuard includes a Render deployment blueprint. Public serving remains unverified until an approved trusted model is available and readiness/scoring checks pass:
- **Live Service URL:** [`https://fraudguard-api.onrender.com`](https://fraudguard-api.onrender.com)
- **Interactive Swagger Docs:** [`https://fraudguard-api.onrender.com/docs`](https://fraudguard-api.onrender.com/docs)
- **Health Check:** [`https://fraudguard-api.onrender.com/health`](https://fraudguard-api.onrender.com/health)
- **Infrastructure Blueprint:** [`render.yaml`](render.yaml)

### API Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Liveness and health probe |
| GET | `/ready` | Readiness probe (verifies champion bundle loaded) |
| GET | `/model-info` | Model metadata, version, calibration mode, and review cutoff |
| POST | `/v1/score` | Real-time single transaction scoring and queue routing |

Example request:
```json
{
  "TransactionID": 3000001,
  "TransactionDT": 86400,
  "TransactionAmt": 150.0,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "debit",
  "P_emaildomain": "gmail.com"
}
```

## 4. Technology Stack

Python 3.10 · pandas · NumPy · scikit-learn · LightGBM · SHAP · FastAPI + Uvicorn · MLflow (local SQLite tracking) · pytest · Ruff · Docker · GitHub Actions · Render. Exact versions are pinned in [`requirements-lock.txt`](requirements-lock.txt).

## 5. Repository Layout

```text
src/fraudguard/      Package: data validation, splits, features, training, calibration,
                     policy, evaluation, explanations, scoring, monitoring, API, CLI
tests/               pytest test suite (isolated fixtures and full contract coverage)
configs/project.json Project configuration (allowlist, partitions, search caps)
examples/            Synthetic single request and batch inputs
reports/             Tracked evaluation evidence, EDA, and benchmarks
docs/                Specifications, decisions, progress log and model card
tasks/todo.md        Task tracker (single source of task completion)
scripts/             Quality-gate and verification scripts
render.yaml          Render free-tier deployment blueprint
Dockerfile           Containerization recipe for cloud deployments
LICENSE              MIT License
```

## 6. Local Setup & Execution

Tested with Python 3.10.11 on Windows and Linux.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-lock.txt
```

### Full Pipeline Commands

```powershell
# 1. Validate raw data
fraudguard validate --input data/raw/train_transaction.csv --report reports/data_quality.json

# 2. Chronological split and fold manifest
fraudguard split --input data/raw/train_transaction.csv --output-dir data/processed

# 3. Train baselines
fraudguard train --models prior amount logistic --output-dir artifacts/baselines

# 4. Tune and select champion candidate
fraudguard tune --output-dir artifacts/selection --comparison-report reports/development_model_comparison.csv

# 5. Independent probability calibration
fraudguard calibrate --output-dir artifacts/champion

# 6. Freeze review policy before test access
fraudguard freeze-policy --output reports/freeze_manifest.json

# 7. Final holdout evaluation
fraudguard evaluate --partition final_test --output-dir reports/final

# 8. SHAP explainability analysis
fraudguard explain --output-dir reports/explanations

# 9. Batch scoring and capped review queue
fraudguard score --input examples/synthetic_batch.csv --output-dir reports/demo

# 10. Drift monitoring
fraudguard monitor --input examples/synthetic_batch.csv --output-dir reports/monitoring

# 11. Retraining reproducibility verification
fraudguard reproduce --output-dir reports/repeatability
```

## 7. Tests and Quality Verification

```powershell
ruff check src tests
ruff format --check src tests
pytest -p no:cacheprovider --cov=fraudguard --cov-report=term-missing
```

`scripts/verify_all.ps1` runs all quality gates locally. GitHub Actions CI automates these checks on every push.

## 8. Documentation & Architecture

- [Model Card](docs/MODEL_CARD.md)
- [Problem & Objectives](docs/01_PROBLEM_AND_OBJECTIVES.md) · [PRD](docs/02_PRD.md) · [Data Specification](docs/03_DATA_SPEC.md)
- [Technical Design](docs/04_TECHNICAL_DESIGN.md) · [Experiment Plan](docs/05_EXPERIMENT_PLAN.md) · [Validation & Release](docs/06_VALIDATION_AND_RELEASE.md)
- [Operations & Commands](docs/07_OPERATIONS_AND_COMMANDS.md) · [Decisions Log](docs/08_DECISIONS_LOG.md) · [Progress Log](docs/09_PROGRESS_LOG.md)
- [Release Checklist](reports/release_checklist.md)

## 9. Data Attribution & License

Field names and data structure originate from the [IEEE-CIS Fraud Detection Benchmark](https://www.kaggle.com/c/ieee-fraud-detection) (IEEE Computational Intelligence Society and Vesta Corporation). The [MIT License](LICENSE) covers project code. Raw IEEE-CIS data is not redistributed by this repository.

## Container model delivery

The image contains code and dependencies. Mount a trusted locally generated champion read-only at /app/artifacts/champion, matching FRAUDGUARD_ARTIFACT_DIR. The public source build does not depend on ignored artifacts. GitHub CI prepares a synthetic bundle solely for runtime smoke testing. That bundle is not the real IEEE-CIS champion. Render native-Python hosting needs an explicitly provided approved artifact before /ready can succeed; a blueprint alone is not deployment proof.
