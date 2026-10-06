# FraudGuard: Transaction Fraud Detection & Capped Review System

## 1. Project Purpose & Summary

FraudGuard estimates transaction fraud risk and produces a ranked, capacity-limited review queue under a strictly capped 1% operational inspection budget.
It is an end-to-end local machine learning engineering demonstrator built on chronological splits, strict schema validation on 12 allowlisted transaction features, temporal cross-validation, independent probability calibration, policy freezing, explainability, fast inference API (<25ms), batch queue generation, and drift monitoring.

> [!NOTE]
> **Dataset Status & Evidence Boundary**:
> Evaluated on a 50,000-row multi-day scaled synthetic benchmark fixture (`data/raw/train_transaction.csv`) spanning 24.85 calendar days (~25 days), modeled after IEEE-CIS transaction features. Partitioned chronologically into Development (30,002 rows, 1,056 fraud), Calibration (4,998 rows, 186 fraud), Policy (5,000 rows, 152 fraud), and Final Test (10,000 rows, 356 fraud). This project demonstrates complete local ML engineering architecture; it does not connect to live payment gateways or claim Malaysian production deployment.

---

## 2. Measured Results Summary

### Final Holdout Evaluation (`final_test` Chronological Partition: 10,000 transactions, 356 fraud cases, 3.56% prevalence)

| Model Family / Configuration | AP | 95% Bootstrap CI (AP) | ROC-AUC | Precision@1% | Recall@1% | Lift@1% | Brier Score | Simulated Cost (Units) | Cost Savings vs None |
|---|---|---|---|---|---|---|---|---|---|
| **Champion: LightGBM (Selected)** | **0.4753** | **[0.4102, 0.5344]** | **0.9526** | **0.6700** | **0.1882** | **18.82x** | **0.0243** | **29,066.0** | **+6,534.0 units (18.4%)** |
| Baseline: Logistic Regression ($C=1.0$) | 0.4177 | [0.3541, 0.4795] | 0.9312 | 0.6500 | 0.1826 | 18.26x | 0.0262 | 29,270.0 | +6,330.0 units (17.8%) |
| Baseline: Amount-Rank Benchmark | 0.0865 | [0.0521, 0.1284] | 0.6756 | 0.1300 | 0.0365 | 3.65x | 0.0341 | 34,574.0 | +1,026.0 units (2.9%) |
| Baseline: Constant-Prior Reference | 0.0356 | [0.0356, 0.0356] | 0.5000 | 0.0400 | 0.0112 | 1.12x | 0.0343 | 35,600.0 | 0.0 units (0.0%) |

*All metrics are verified from `reports/final/final_evaluation.json`.*

> [!IMPORTANT]
> **Synthetic Rule Recovery Disclosure**:
> The performance separation where LightGBM improves over Logistic Regression (AP 0.4753 vs 0.4177) reflects the synthetic fixture generator's planted interaction rules (non-linear night-hour risks, high-risk email domains, and ProductCD transaction amount patterns) being effectively recovered by tree ensembles and engineered transaction interaction features. These numbers demonstrate model-family differentiation and the local ML engineering architecture; they do not represent real-world fraud detection efficacy on the full IEEE-CIS competition dataset.

### Engineering & Serving Benchmarks
- **API Latency (1,000 requests to `/v1/score`)**:
  - **p50:** 8.58 ms | **p95:** 9.67 ms | **p99:** 11.23 ms | **Mean:** 8.67 ms (Target $\le 100$ ms: **MET**)
- **Training Reproducibility**: Exact 0.000000 AP & Brier discrepancy between fresh retrain and frozen artifact (Tolerance $\le 0.001$: **MET**).
- **Test Coverage**: 50 unit and integration tests passing, **84% code coverage** (Target $\ge 80\%$: **MET**).
- **SHAP Additivity Check**: Max error $1.32 \times 10^{-14} \le 10^{-4}$ (**MET**).
- **Stationary Block Bootstrap Uncertainty**: 1,000 replicates across 6 distinct final test day blocks (**STATUS: SUCCESS**).

---

## 3. Quickstart & CLI Commands

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2. Full Pipeline Execution
```powershell
# 1. Generate scaled 50,000-row synthetic benchmark dataset (~25 days)
fraudguard generate-synthetic --rows 50000 --days 25 --fraud-rate 0.035 --output data/raw/train_transaction.csv

# 2. Validate raw data
fraudguard validate --input data/raw/train_transaction.csv --report reports/data_quality.json

# 3. Chronological split & manifest generation
fraudguard split --input data/raw/train_transaction.csv --output-dir data/processed

# 4. Train baselines
fraudguard train --models prior amount logistic --output-dir artifacts/baselines

# 5. Tune & select candidate
fraudguard tune --output-dir artifacts/selection --comparison-report reports/development_model_comparison.csv

# 6. Independent probability calibration
fraudguard calibrate --output-dir artifacts/champion

# 7. Freeze review policy & manifest
fraudguard freeze-policy --output reports/freeze_manifest.json

# 8. Final holdout evaluation
fraudguard evaluate --partition final_test --output-dir reports/final

# 9. Global & local SHAP explanations
fraudguard explain --output-dir reports/explanations

# 10. Batch score & generate review queue
fraudguard score --input examples/synthetic_batch.csv --output-dir reports/demo

# 11. Drift monitoring
fraudguard monitor --input examples/synthetic_batch.csv --output-dir reports/monitoring

# 12. API latency benchmark
fraudguard benchmark --warmups 100 --requests 1000 --output reports/api_benchmark.json

# 13. Retraining reproducibility verification
fraudguard reproduce --output-dir reports/repeatability
```

### 3. Serving & Container Contract
- **Start FastAPI Server**:
  ```powershell
  uvicorn fraudguard.api:app --host 0.0.0.0 --port 8000
  ```
- **Verify Container Contract & Verification Script**:
  ```powershell
  powershell -File scripts/verify_container.ps1
  ```

---

## 4. Key Artifacts Directory

- **Frozen Champion Bundle:** `artifacts/champion/` (`pipeline.joblib`, `policy.json`, `manifest.json`)
- **Freeze Manifest:** `reports/freeze_manifest.json`
- **Final Evaluation Reports:** `reports/final/final_evaluation.json`, `reports/final/model_comparison.csv`
- **Development Model Comparison:** `reports/development_model_comparison.csv`
- **Explanations:** `reports/explanations/global_importance.csv`, `reports/explanations/local_contributions.json`
- **Monitoring:** `reports/monitoring/report.json`, `reports/monitoring/report.html`
- **Benchmarks:** `reports/api_benchmark.json`, `reports/repeatability/reproducibility.json`
- **Model Card:** [MODEL_CARD.md](docs/MODEL_CARD.md)

---

## 5. Scope & Limitations

1. **Synthetic Fixture Scale & Rule Recovery**: Evaluated on a 50,000-row fixture spanning 24.85 days (~25 days) modeled after IEEE-CIS features; performance reflects synthetic rule recovery of planted generator patterns. Ingestion of the full ~590k IEEE-CIS dataset requires Kaggle credentials.
2. **Compact Feature Scope**: Derived exclusively from the 12 allowlisted transaction fields and transaction timestamp without identity tables or opaque variables.
3. **Operational Review Cap**: Strictly bounds manual queue size to 1% of transaction volume.
4. **Container Status (T17 & T18 OPEN)**: Packaging contract and verification script are tested locally; live Docker container execution and full remote CI gate remain open pending active Docker daemon and remote runner authorization.
