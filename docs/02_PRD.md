# 02. Product requirements document

| Field | Value |
|---|---|
| Product | FraudGuard |
| Version | Planning specification 1.0 |
| Owner | ZeeqRyz |
| Prepared | 6 October 2026, Malaysia time |
| Status | Ready for implementation handoff; implementation not started |
| Product outcome | Rank transaction fraud risk within a review workload |

## 1. Product scope

FraudGuard provides a transaction score, a review recommendation and a batch review queue. The MVP is a local decision-support demonstration, evaluated on a historical benchmark. It must show whether extra model complexity improves a meaningful review decision.

## 2. User stories

| ID | Story | Priority |
|---|---|---|
| US1 | As an operations lead, I can rank a batch and limit the queue to my review capacity. | Must |
| US2 | As an analyst, I can inspect model contributions without treating them as proof of fraud. | Must, offline |
| US3 | As an engineer, I can score one valid transaction through an API. | Must |
| US4 | As an engineer, I can reproduce the selected model and evaluation. | Must |
| US5 | As a maintainer, I can detect input and score distribution changes. | Must |
| US6 | As a nontechnical reviewer, I can explore results in a dashboard. | Later |

## 3. Functional requirements

| ID | Requirement | Acceptance criteria | Tasks |
|---|---|---|---|
| FR1 | Acquire and validate permitted benchmark data | Provenance recorded; invalid rows and duplicate IDs fail visibly; only allowlisted columns retained | T02-T03 |
| FR2 | Create chronological partitions | All boundaries keep equal timestamps together; no future rows train an earlier fold; manifest reproducible | T04 |
| FR3 | Establish metrics and baselines | Prior and Logistic Regression reports use the same folds and capacity rules | T05-T06 |
| FR4 | Compare a bounded challenger | LightGBM and small search obey development-only selection; all runs logged | T09-T10 |
| FR5 | Select probability mapping and review cutoff | Calibration/policy separation respected; artifact includes cutoff, capacity default and provenance | T11-T12 |
| FR6 | Final evaluation | Frozen pipeline evaluated on latest labelled holdout; AP, ranking, policy, calibration and cost reports produced | T13 |
| FR7 | Single score API | `POST /v1/score` validates input, returns finite score, recommendation and version; matches CLI | T07-T08, T12 |
| FR8 | Batch review queue | CLI writes scores for every valid row, sorted deterministically; queue never exceeds capacity | T15 |
| FR9 | Explain model behaviour | Global permutation report and offline signed local contributions for at most 200 rows | T14 |
| FR10 | Health and metadata | Liveness independent of model; readiness and model-info reflect actual artifact state | T08, T12 |
| FR11 | Monitoring | Reference/batch comparison shows missingness, unknown categories and score shifts; no automatic retraining | T16 |
| FR12 | Reproducible delivery | Synthetic tests in CI, local Docker smoke check, documented commands and completed model card | T17-T20 |

## 4. Product flow

1. User supplies a validated transaction payload or CSV.
2. Shared preprocessing maps the input to the frozen feature schema.
3. Model and optional calibrated mapping return a fraud-risk score.
4. A single request returns `review_recommended` from the frozen cutoff. It cannot know an analyst's remaining capacity.
5. Batch scoring filters candidates above the cutoff, sorts by score descending then ID ascending, and selects at most K rows. Batch selection is `selected_for_review`; it may be false even when the single-request recommendation is true.
6. Reports distinguish unrestricted top-K ranking quality from the actual cutoff-and-cap policy.

## 5. Nonfunctional requirements

| ID | Requirement | Evidence / target |
|---|---|---|
| NFR1 | API responsiveness | <= 100 ms warm p95 over 1,000 sequential local requests after 100 warmups, concurrency 1; record hardware |
| NFR2 | Repeatability | Same environment/data/config/seed gives AP and Brier within 0.001 absolute; parity tolerance 1e-8 |
| NFR3 | Input limits | Single JSON <= 64 KiB; batch CLI <= 100,000 rows and 100 MiB; reject violations |
| NFR4 | Failure behaviour | HTTP 422 invalid fields, 413 payload limit, 503 not-ready model; no partial scoring by default |
| NFR5 | Code quality | Ruff lint/format pass; meaningful unit/integration tests; >= 80% coverage on core data, scoring, metrics and policy modules |
| NFR6 | Portability | Runs in a local venv and container; CPU only |
| NFR7 | Privacy | No raw transaction payloads, credential values or per-transaction scores in routine logs; no dataset in Git or CI |
| NFR8 | Resource discipline | Read selected CSV columns, prefer sparse LR encoding, cap model threads; report measured memory and training runtime |

## 6. Required artifacts

- Dataset/schema/split manifests and data-quality report.
- Baseline and challenger comparison with run identifiers.
- Trusted model bundle, feature schema, policy and reference summary.
- Frozen final evaluation, review-budget curves, simulated cost sensitivity and calibration plots.
- Offline explanations and segment/time-bin error analysis.
- CLI/API parity, local benchmark, container and CI evidence.
- Model card, README results and exact reproduction commands.

Artifact formats belong in [technical design](04_TECHNICAL_DESIGN.md). Completion evidence belongs in [validation](06_VALIDATION_AND_RELEASE.md).

## 7. Out of scope

- Payment authorization, automatic decline/blocking, live transaction ingestion and bank integration.
- Streaming/Kafka, feature store, Kubernetes, deep learning or automatic retraining.
- Identity join, opaque feature groups, entity-history features and SMOTE in the MVP.
- Public cloud deployment, public dataset redistribution and dashboard development.
- Real-world fraud-reduction, recovery, ROI or Malaysian applicability claims.

## 8. Release criteria

1. All Must requirements have evidence and linked completed tasks.
2. Data/model integrity gates and engineering gates pass.
3. Final usefulness targets have an honest met/missed result; shortfall is visible in README and model card.
4. Acquisition/reuse restrictions and absence of public deployment authorization are accounted for.
5. No requirement is marked done from a planned command or fabricated metric.

## 9. Assumptions and change control

- Default review capacity is an illustrative 1%, configurable per batch within `(0, 1]`.
- Costs are synthetic; stakeholder validation is later work.
- Acquisition may require user-performed Kaggle consent.
- This document is the product contract. Changes to scope, capacity semantics or label meaning require a decision entry and updates to affected documents.
