# FraudGuard task tracker

## 1. Rules

- This is the **only task-completion checklist**. All implementation tasks start unchecked.
- Read [START_HERE](../docs/00_START_HERE.md) and the owning specifications before execution.
- Complete a task only after acceptance checks pass; append exact commands/results to the progress log.
- Commands below are future verification interfaces. Define `$fgPython` as specified in [operations](../docs/07_OPERATIONS_AND_COMMANDS.md) after T01.
- File lists name planned primary files. Common decision/progress log updates are additional routine documentation.
- Estimates are human focused-work estimates, total approximately **70 hours**, within the plan's 60-80 hour range. Split tasks into smaller increments if they exceed a focused session.

## 2. Data-integrity milestone

### T01: Environment and minimal package

- [x] T01 complete.
- **Depends on:** none. **Estimate:** 3 hours.
- **Acceptance:** inspect Python/Git/Docker/memory; create isolated installable package with CLI help; pin compatible dependencies and record exact versions/hardware.
- **Verify:** `& $fgPython -m pip check`; `& $fgPython -m fraudguard.cli --help`; `& $fgPython -m ruff check src`.
- **Primary files:** `pyproject.toml`, `requirements-lock.txt`, `configs/project.json`, `src/fraudguard/__init__.py`, `src/fraudguard/cli.py`.
- **Owner spec:** technical design sections 2-4; operations section 2.
- **Git:** inspect current state; initialize this folder only if a standalone repository is appropriate. Create minimal ignore rules before adding generated files. Never stage neighbouring projects.

### T02: Dataset access and provenance

- [x] T02 complete.
- **Depends on:** T01. **Estimate:** 2 hours, excluding external waiting.
- **Acceptance:** verify actual Kaggle access/terms; acquire the original labelled transaction file through an authorized method; record filenames, sizes and hashes without exposing credentials.
- **Verify:** inspect applicable terms and provenance; `Get-FileHash -Algorithm SHA256 data/raw/train_transaction.csv`; confirm raw data is excluded from Git.
- **Primary files:** `data/raw/DATA_PROVENANCE.md`, `.gitignore`, `docs/10_SOURCES.md`.
- **Owner spec:** data specification section 1.
- **Blocker rule:** user performs required consent or credential setup; synthetic T05 can proceed while this is pending.

### T03: Validate and profile the allowed transaction fields

- [x] T03 complete.
- **Depends on:** T02. **Estimate:** 4 hours.
- **Acceptance:** labelled and scoring schemas enforce the data contract; data-quality report records selected columns/errors/memory; synthetic tests cover missing, unknown, duplicate and invalid values.
- **Verify:** `& $fgPython -m pytest tests/test_data.py -p no:cacheprovider --basetemp .tmp/pytest`; run the documented `validate` command and inspect the report.
- **Primary files:** `src/fraudguard/data.py`, `src/fraudguard/cli.py`, `tests/test_data.py`, `tests/fixtures/transactions.csv`, `reports/data_quality.json`.
- **Owner spec:** data specification sections 2-4 and 7.

### T04: Freeze chronological partitions and fold manifests

- [x] T04 complete.
- **Depends on:** T03. **Estimate:** 4 hours.
- **Acceptance:** reproducible 60/10/10/20 partitions and expanding development folds preserve timestamp groups; split checks fail invalid chronology/class support; development-only EDA records five observations without test-label analysis.
- **Verify:** `& $fgPython -m pytest tests/test_splits.py -p no:cacheprovider --basetemp .tmp/pytest`; run `split` twice and compare manifest IDs/hashes.
- **Primary files:** `src/fraudguard/splits.py`, `src/fraudguard/cli.py`, `tests/test_splits.py`, `data/processed/split_manifest.json`, `reports/development_eda.md`.
- **Owner spec:** data specification sections 5-7.

### Checkpoint G1

- [x] G1 verified: provenance, schema, memory report, partition/fold integrity and holdout restrictions have evidence.

## 3. First scoring slice

### T05: Ranking and cost metric helpers

- [x] T05 complete.
- **Depends on:** T01. **Estimate:** 3 hours.
- **Acceptance:** AP, top-K metrics, capacity/ties and synthetic cost formula match hand-computed cases; unavailable denominators remain visible; policy metrics are distinct from unconditional ranking.
- **Verify:** `& $fgPython -m pytest tests/test_metrics.py -p no:cacheprovider --basetemp .tmp/pytest`.
- **Primary files:** `src/fraudguard/metrics.py`, `src/fraudguard/policy.py`, `tests/test_metrics.py`.
- **Owner spec:** experiment plan sections 4, 7-8.

### T06: Prior, amount and Logistic Regression baselines

- [x] T06 complete.
- **Depends on:** T04, T05. **Estimate:** 4 hours.
- **Acceptance:** training-only preprocessing and baseline temporal folds work; metrics/runtime/memory are logged to local MLflow; preserve the unweighted C=1 LR reference fitted on all development rows, separately from the provisional winner.
- **Verify:** `& $fgPython -m pytest tests/test_training.py -p no:cacheprovider --basetemp .tmp/pytest`; run documented baseline `train` command and inspect fold/run artifacts.
- **Primary files:** `src/fraudguard/features.py`, `src/fraudguard/train.py`, `src/fraudguard/cli.py`, `tests/test_training.py`, `reports/baseline_comparison.csv`.
- **Owner spec:** experiment plan sections 1-5.

### T07: Trusted provisional bundle and shared scorer

- [x] T07 complete.
- **Depends on:** T06. **Estimate:** 3 hours.
- **Acceptance:** provisional baseline bundle saves/loads with schema/version/hash checks; one shared function returns finite scores; missing/corrupt/untrusted-path bundles fail visibly.
- **Verify:** `& $fgPython -m pytest tests/test_scoring.py -p no:cacheprovider --basetemp .tmp/pytest`; compare saved/reloaded synthetic fixture scores within 1e-8.
- **Primary files:** `src/fraudguard/artifacts.py`, `src/fraudguard/scoring.py`, `tests/test_scoring.py`, `artifacts/baselines/provisional/manifest.json`.
- **Owner spec:** technical design sections 5 and 7. Provisional cutoff is 0.5 and must be labelled provisional.

### T08: Single-score API vertical slice

- [x] T08 complete.
- **Depends on:** T07. **Estimate:** 4 hours.
- **Acceptance:** liveness/readiness/model-info/score routes obey strict schemas and byte limits; API calls shared scorer; synthetic request scores match offline baseline and error tests pass.
- **Verify:** `& $fgPython -m pytest tests/test_api.py -p no:cacheprovider --basetemp .tmp/pytest`; run Uvicorn and execute documented synthetic health/readiness/score requests.
- **Primary files:** `src/fraudguard/api.py`, `tests/test_api.py`, `examples/synthetic_transaction.json`.
- **Owner spec:** technical design section 5; validation section 5.

### Checkpoint G2

- [x] G2 verified: hand-checked metrics and baseline -> saved bundle -> API score work end to end. Provisional status is visible.

## 4. Candidate selection and freeze

### T09: LightGBM challenger

- [x] T09 complete.
- **Depends on:** T06. **Estimate:** 4 hours.
- **Acceptance:** weighted/unweighted challenger uses the same raw allowlist and fold manifests; class ratio is computed inside training folds; bounded CPU runs and probability diagnostics are recorded.
- **Verify:** `& $fgPython -m pytest tests/test_training.py -p no:cacheprovider --basetemp .tmp/pytest`; run challenger `train`, compare fold IDs and inspect resource usage.
- **Primary files:** `src/fraudguard/train.py`, `src/fraudguard/features.py`, `tests/test_training.py`, `configs/project.json`, `reports/challenger_comparison.csv`.
- **Owner spec:** experiment plan sections 2-3.

### T10: Bounded tuning and model selection

- [x] T10 complete.
- **Depends on:** T09. **Estimate:** 5 hours.
- **Acceptance:** search respects configuration caps and development-only data; declared AP/Recall@1%/simplicity rule selects the candidate; selection file references exact runs/configuration and no test metrics.
- **Verify:** `& $fgPython -m pytest tests/test_selection.py -p no:cacheprovider --basetemp .tmp/pytest`; run `tune` and hand-check the selection rule against the comparison table.
- **Primary files:** `src/fraudguard/train.py`, `src/fraudguard/cli.py`, `tests/test_selection.py`, `artifacts/selection/selected.json`, `reports/development_model_comparison.csv`.
- **Owner spec:** experiment plan sections 3 and 5.

### T11: Independent calibration decision

- [x] T11 complete.
- **Depends on:** T10. **Estimate:** 3 hours.
- **Acceptance:** classifiers fitted on development only; mappings fitted on calibration only; raw versus sigmoid choice uses the specified policy criteria for both selected candidate and fixed LR reference, with independent reliability evidence.
- **Verify:** `& $fgPython -m pytest tests/test_calibration.py -p no:cacheprovider --basetemp .tmp/pytest`; run `calibrate` and verify classifier hash is unchanged during calibrator fitting.
- **Primary files:** `src/fraudguard/calibration.py`, `src/fraudguard/cli.py`, `tests/test_calibration.py`, `reports/calibration_comparison.json`, `reports/calibration_reliability.png`.
- **Owner spec:** experiment plan sections 1 and 6.

### T12: Freeze cutoff, capacity policy and champion bundle

- [x] T12 complete.
- **Depends on:** T11, T08. **Estimate:** 3 hours.
- **Acceptance:** independent policy-derived cutoffs and deterministic caps pass tie/underfill tests for champion and LR reference; champion bundle includes final score type/versions; freeze manifest hashes all final inputs before test access and API uses the final policy.
- **Verify:** `& $fgPython -m pytest tests/test_policy.py tests/test_api.py -p no:cacheprovider --basetemp .tmp/pytest`; run `freeze-policy` and verify file hashes and pre-test timestamp.
- **Primary files:** `src/fraudguard/policy.py`, `src/fraudguard/artifacts.py`, `src/fraudguard/cli.py`, `tests/test_policy.py`, `reports/freeze_manifest.json`.
- **Owner spec:** experiment plan section 7; technical design sections 5-7.

### Checkpoint G3

- [x] G3 verified: selection, calibration and policy contracts pass, and the frozen final bundle is ready before final-test labels are inspected.

## 5. Final scientific evaluation

### T13: Frozen final evaluation and uncertainty

- [x] T13 complete.
- **Depends on:** T12. **Estimate:** 5 hours.
- **Acceptance:** final holdout is scored under the unchanged freeze; baseline/ranking/policy/calibration/cost and time-block intervals are generated on complete partitions; minimum/stretch targets and low-support segments are reported honestly.
- **Verify:** `& $fgPython -m pytest tests/test_evaluation.py -p no:cacheprovider --basetemp .tmp/pytest`; execute documented `evaluate`, inspect denominators/hash checks and hand-check representative counts/costs.
- **Primary files:** `src/fraudguard/metrics.py`, `src/fraudguard/cli.py`, `tests/test_evaluation.py`, `reports/final/final_evaluation.json`, `reports/final/model_comparison.csv`.
- **Owner spec:** experiment plan sections 4 and 8-11. Do not retune after viewing results.

### T14: Global and offline local explanations

- [x] T14 complete.
- **Depends on:** T13. **Estimate:** 4 hours.
- **Acceptance:** policy-partition AP permutation importance and <= 200 local explanations exist; transformed features map to raw names; raw-space additive check passes and noncausal wording respects calibration.
- **Verify:** `& $fgPython -m pytest tests/test_explain.py -p no:cacheprovider --basetemp .tmp/pytest`; run `explain` and inspect signs, grouped features and additive errors <= 1e-4.
- **Primary files:** `src/fraudguard/explain.py`, `src/fraudguard/cli.py`, `tests/test_explain.py`, `reports/explanations/global_importance.csv`, `reports/explanations/local_contributions.json`.
- **Owner spec:** experiment plan section 10.

### Checkpoint G4

- [x] G4 verified: final evidence and explanation checks complete; no revised policy/model selection uses final-test outcomes.

## 6. Local operational delivery

### T15: Validated batch scoring and capped review queue

- [x] T15 complete.
- **Depends on:** T12. **Estimate:** 4 hours.
- **Acceptance:** strict unlabelled CSV scores every valid input row with deterministic rank/version fields; qualifying queue obeys cap and underfill semantics; malformed/oversized batches fail as a whole.
- **Verify:** `& $fgPython -m pytest tests/test_batch.py -p no:cacheprovider --basetemp .tmp/pytest`; run documented synthetic `score` and compare selected count/scores with offline expectations.
- **Primary files:** `src/fraudguard/scoring.py`, `src/fraudguard/cli.py`, `tests/test_batch.py`, `examples/synthetic_batch.csv`.
- **Owner spec:** technical design section 6; data specification section 2.

### T16: Input and score monitoring

- [x] T16 complete.
- **Depends on:** T15. **Estimate:** 3 hours.
- **Acceptance:** frozen development reference compares missingness, unseen categories and numeric/score PSI using the specified thresholds/support rules; synthetic perturbations trigger documented diagnostics; no fraud-quality or automatic-retrain claim is made without labels.
- **Verify:** `& $fgPython -m pytest tests/test_monitoring.py -p no:cacheprovider --basetemp .tmp/pytest`; run `monitor` on unchanged and deliberately shifted synthetic fixtures and inspect affected diagnostics.
- **Primary files:** `src/fraudguard/monitoring.py`, `src/fraudguard/cli.py`, `tests/test_monitoring.py`, `reports/monitoring/report.json`, `reports/monitoring/report.html`.
- **Owner spec:** technical design section 8; validation section 6.

### T17: Container with mounted trusted artifacts

- [ ] T17 complete (Contract tested; Docker daemon offline on host).
- **Depends on:** T15. **Estimate:** 3 hours.
- **Acceptance:** image excludes raw/local artifacts from build context; read-only model mount yields ready API and synthetic score; missing mount returns not-ready, with exact local runtime evidence recorded.
- **Verify:** run the documented `docker build` and `docker run`; call `/ready` and `/v1/score`; inspect build context exclusions and mounted-artifact behaviour.
- **Primary files:** `Dockerfile`, `.dockerignore`, `tests/test_container_contract.py` if a meaningful contract test is needed.
- **Owner spec:** operations section 9; validation section 7. Docker absence leaves this task open.

### T18: Synthetic CI and coverage gate

- [ ] T18 complete (Local workflow passed; full gate depends on live Docker build in T17 and remote GitHub Actions execution).
- **Depends on:** T16, T17. **Estimate:** 2 hours.
- **Acceptance:** pipeline installs locked dependencies and runs lint/format/core tests/coverage plus container build; fixtures and bundle are synthetic; no Kaggle access, restricted data or credentials are required.
- **Verify:** run the documented quality commands locally; inspect workflow paths/exclusions and perform a local workflow-equivalent run. Record remote CI as unverified until an authorized repository run exists.
- **Primary files:** `.github/workflows/ci.yml`, `pyproject.toml`, `tests/conftest.py`, `tests/test_synthetic_flow.py`.
- **Owner spec:** PRD FR12 and validation section 7. Publishing/pushing is separate authorization.

### Checkpoint G5

- [ ] G5 verified (OPEN: Local batch and monitoring verified, but container build/run in T17 and remote CI in T18 remain open).

## 7. Reproducibility and handoff

### T19: API benchmark and training repeatability

- [x] T19 complete.
- **Depends on:** T18. **Estimate:** 4 hours.
- **Acceptance:** warm p50/p95/p99 and errors are measured under the defined 100+1,000-request protocol; repeated selected training configuration meets tolerance or reports a reproducibility defect; hardware/memory/runtime are documented.
- **Verify:** run documented `benchmark` and `reproduce`; compare development/policy metrics and score parity without tuning. Inspect request counts and timings.
- **Primary files:** `src/fraudguard/cli.py`, `tests/test_benchmark.py`, `reports/api_benchmark.json`, `reports/reproducibility.json`.
- **Owner spec:** validation section 8 and PRD NFR1-NFR2.

### T20: Evidence-backed portfolio handoff

- [x] T20 complete.
- **Depends on:** T14, T19. **Estimate:** 3 hours.
- **Acceptance:** README/model card show actual results, misses, limits and commands; each Must requirement maps to verified evidence; local deliverable contains no data/credential exposure and all remaining gaps are stated.
- **Verify:** inspect every linked artifact and command; run final quality checks if recent changes require them; compare PRD requirements with the completed task checklist; inspect Git status/ignore rules without publishing.
- **Primary files:** `README.md`, `docs/MODEL_CARD.md`, `docs/07_OPERATIONS_AND_COMMANDS.md`, `docs/06_VALIDATION_AND_RELEASE.md`, `reports/release_checklist.md`.
- **Owner spec:** PRD sections 6 and 8; validation section 9.

### Checkpoint G6

- [ ] G6 verified (OPEN: Local MVP handoff documented with empirical evidence and explicit limitations; live Docker and remote CI remain open).
