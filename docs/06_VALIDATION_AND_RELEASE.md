# 06. Validation and release evidence

## 1. Definition of done

Complete a task only when its acceptance checks pass, outputs exist and the progress log records exact verification evidence. Mark tasks only in [tasks/todo.md](../tasks/todo.md). A report describing intended behaviour is not runtime proof.

Separate three outcomes:

- **Integrity passed:** data provenance, chronology, feature rules and evaluation freeze are valid.
- **Engineering passed:** software, tests, CLI/API parity and delivery checks pass.
- **Model usefulness:** minimum and stretch objectives are explicitly met or missed.

An honest performance shortfall can still produce a completed portfolio demonstrator. It cannot justify operational readiness or fabricated improvements.

## 2. Milestone gates

| Gate | Tasks | Required evidence |
|---|---|---|
| G1: data integrity | T01-T04 | Environment, provenance, schema report and reproducible timestamp-disjoint split |
| G2: first scoring slice | T05-T08 | Hand-checked metrics, baseline bundle and synthetic CLI/API parity |
| G3: selection freeze | T09-T12 | CV comparison, calibration decision, frozen policy and all hashes |
| G4: final science | T13-T14 | Final evaluation, uncertainty, misses and explanation checks |
| G5: usable local delivery | T15-T18 | Batch contract, monitoring, local container and synthetic CI |
| G6: handoff | T19-T20 | Benchmark, repeatability, traceability and completed model card |

## 3. Data and leakage tests

- Missing required fields, invalid labels, duplicate IDs, negative required values, nonfinite values and invalid encoded categories fail visibly.
- Nullable values remain distinguishable from malformed values.
- Equal timestamps never cross partitions or development folds.
- All training timestamps precede validation/calibration/policy/test timestamps as appropriate.
- Changing validation values cannot change fitted training medians or vocabularies.
- Model feature names cannot include `isFraud`, `TransactionID`, `TransactionDT` or forbidden groups.
- Final-test labels never appear in training, calibration or policy function inputs.
- A split manifest rerun yields the same IDs and hashes.

Use original synthetic fixtures with ties, missing values, unknown categories and two-class examples. Restricted benchmark rows do not belong in tests.

## 4. Decision and metric tests

- Verify AP against the library on a small hand-checkable ranking; name it AP consistently.
- Verify precision, recall, lift and cost using manually computed expected counts.
- Test equal scores, all-zero scores, one-row batches, q=1 and fractional capacity rounding.
- Verify score-descending/ID-ascending order and exact cap behaviour.
- Check candidate underfill, empty queues, no fraud labels and no reviewed rows.
- Confirm the fixed cutoff is unchanged by test/batch data.
- Keep unconditional top-K metrics distinct from cutoff-and-cap metrics.

## 5. API and artifact tests

1. Synthetic input yields the same score in Python scorer, CLI and HTTP response within 1e-8.
2. Nullable and unseen categories score without refitting transforms.
3. Strict type checks reject booleans/numeric strings where prohibited; target and extra fields fail.
4. Missing/corrupted/incompatible bundle yields 503 readiness and score failure; `/health` remains a liveness check.
5. Payload limits work even without a trustworthy Content-Length header.
6. Errors and logs contain useful categories without exposing payloads or credentials.
7. Batch limits, duplicates and malformed rows fail the whole batch by default.

## 6. Model evidence checks

- All baseline/challenger runs use the same allowed raw predictors and manifests.
- All learned preprocessing is fitted within the permitted training partition.
- Calibrator training is independent of classifier training.
- Freeze manifest precedes final-test metric access.
- Selected model and baseline comparisons use identical final-test rows.
- Every metric includes partition, row count, positive count and policy/version identifiers.
- Reliability, cost assumptions, uncertainty limits and segment support are visible.
- Explanation sums match raw model output within 1e-4; calibrated probability claims respect the transformation.

## 7. Engineering checks

Commands are defined in [operations](07_OPERATIONS_AND_COMMANDS.md); they are planned until implemented.

```powershell
& $fgPython -m ruff check src tests
& $fgPython -m ruff format --check src tests
& $fgPython -m pytest -p no:cacheprovider --basetemp .tmp/pytest --cov=fraudguard --cov-report=term-missing
```

- Test coverage gate: >= 80% for core data, scoring, metrics and policy logic. Review uncovered branches; do not add assertion-free tests to raise a number.
- CI uses synthetic data and a synthetic fitted bundle. It does not prove full-data model quality.
- Docker image excludes data, reports, local MLflow files and model artifacts from build context; model directory is mounted read-only at runtime.
- Inspect running readiness and score response after building the image. A successful build alone is insufficient.
- If Docker is unavailable, record the exact limitation and leave T17/G5 open rather than claiming container validation.

## 8. Benchmark and reproducibility

- Warm single-score API: 100 warmups, then 1,000 sequential requests, concurrency 1, localhost.
- Measure end-to-end client elapsed time, including JSON/HTTP and preprocessing. Exclude offline explanations.
- Report p50/p95/p99, hardware, versions, thread settings, model size and error count.
- Target p95 <= 100 ms is provisional until measured. Cold start is reported separately.
- Run the same selected training configuration twice in the pinned environment; AP/Brier tolerance is 0.001 absolute.
- Repeatability uses development/policy metrics and fixed final-model scores, without another model-selection cycle on test labels.

## 9. Local handoff and future publication

Required local handoff:

1. Must requirements traced to completed tasks and evidence paths.
2. README describes actual implemented commands and measured results.
3. Model card filled with measured values, intended use and limitations.
4. Decision/progress logs identify next work and any unverified gate.
5. Check raw data, local scores, credentials and model files are ignored appropriately.

Public release is separate from local completion. Before any authorized publication, recheck dataset/model reuse permissions, review synthetic-only examples and publish only permitted source, tests and aggregate reports. Public cloud spend or external submission is not authorized by this planning package.
