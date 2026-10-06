# FraudGuard Release & Quality Checklist

## 1. Requirement & Quality Verification

| Gate / Requirement | Status | Evidence Location | Notes |
|---|---|---|---|
| **FR01: Strict Schema & Ingestion** | PASSED | `reports/data_quality.json`, `tests/test_data.py` | 12 allowed features + timestamp, type checking, error logging across 50,000 rows |
| **FR02: Chronological Splitting** | PASSED | `data/processed/split_manifest.json`, `tests/test_splits.py` | 60/10/10/20 partition (30002/4998/5000/10000 rows, 24.85 days ~25 days) with timestamp group preservation |
| **FR03: Baseline Models** | PASSED | `reports/baseline_comparison.csv`, `tests/test_training.py` | Prior (AP 0.036), Amount (AP 0.086), Logistic Regression (AP 0.418 on final test, 0.423 on dev CV) |
| **FR04: MLflow Tracking** | PASSED | `mlruns/` | Runs logged with parameters, CV folds, and metrics |
| **FR05: Model Selection** | PASSED | `artifacts/selection/selected.json`, `reports/development_model_comparison.csv`, `tests/test_selection.py` | `LGBM_unweighted_n200_lr0.03_l15` selected via CV AP (0.487) and simplicity rule |
| **FR06: Probability Calibration** | PASSED | `reports/calibration_comparison.json`, `reports/calibration_reliability.png` | Raw probabilities selected (Brier 0.02115 on independent 4,998-row calibration partition) |
| **FR07: Policy Freezing** | PASSED | `reports/freeze_manifest.json`, `tests/test_policy.py` | Hashes and cutoff ($\tau = 0.529162$) frozen prior to test holdout access |
| **FR08: Holdout Evaluation** | PASSED | `reports/final/final_evaluation.json`, `tests/test_evaluation.py` | Evaluated on 10,000 final transactions (Champion AP 0.4753, Cost 29066 vs LR AP 0.4177, Cost 29270; 95% Bootstrap CI [0.4102, 0.5344]) |
| **FR09: Explainability** | PASSED | `reports/explanations/global_importance.csv`, `reports/explanations/local_contributions.json` | AP permutation importance & TreeExplainer SHAP ($err \le 1.32 \times 10^{-14}$) |
| **FR10: Scoring API** | PASSED | `src/fraudguard/api.py`, `tests/test_api.py` | `/health`, `/ready`, `/model-info`, `/v1/score` with strict 64 KiB payload limit |
| **FR11: Batch Scoring & Review** | PASSED | `reports/demo/scored.csv`, `reports/demo/review_queue.csv`, `tests/test_batch.py` | Top-ranked capped review queue with deterministic tie-breaking |
| **FR12: Drift Monitoring** | PASSED | `reports/monitoring/report.json`, `reports/monitoring/report.html`, `tests/test_monitoring.py` | Missingness, new categories, score PSI & feature PSI |
| **NFR01: API Latency** | PASSED | `reports/api_benchmark.json` | p95 = 9.67 ms ($\le 100$ ms target MET) |
| **NFR02: Reproducibility** | PASSED | `reports/repeatability/reproducibility.json` | Exact 0.000000 AP/Brier tolerance |
| **NFR03: Test Coverage** | PASSED | `tests/` | 50 passed, 84% coverage ($\ge 80\%$ target MET) |
| **Docker Container (T17)** | OPEN | `Dockerfile`, `scripts/verify_container.ps1`, `tests/test_container_contract.py` | Packaging contract verified; live container run open due to host Docker daemon offline |
| **Synthetic CI & Gate (T18)** | OPEN | `.github/workflows/ci.yml`, `tests/test_synthetic_flow.py` | Local flow passes; gate open pending T17 live build and remote GitHub Actions execution |

---

## 2. Release Sign-off Summary

- **Local MVP Tasks Completed**: T01–T16, T19–T20 completed and verified with evidence.
- **T17 & T18 Status**: Kept OPEN (T17 requires live Docker service on host; T18 depends on T17 and remote GitHub Actions runner).
- **Quality Gates**: G1–G4 verified; G5 and G6 marked OPEN pending live container build and remote CI execution.
