# 01. Problem and objectives

## 1. Problem statement

A transaction-review team has limited capacity. Reviewing too little fraud leaves risk undetected; reviewing too many legitimate transactions increases workload and customer friction. A classifier that mainly predicts the common class can show high accuracy while providing little value to that team.

**Project question:** can a small classical ML system rank labelled fraudulent transactions more effectively than simple baselines, when the reviewer can inspect only a small fraction of transactions?

This is a portfolio product hypothesis. No stakeholder interview, actual bank workflow or Malaysian fraud prevalence has been validated.

## 2. Intended users and decisions

| User | Decision | Output |
|---|---|---|
| Fraud operations lead | Which transactions receive scarce review slots? | Ranked, capped queue and workload metrics |
| Fraud analyst | Which supplied attributes drove the model score? | Offline model explanations with feature values |
| ML engineer | Is the model reliable enough for a local demo? | Reproducible evaluations, service tests and monitoring |
| Portfolio reviewer | Can the candidate justify the trade-offs? | Model card, results, assumptions and reproducible commands |

## 3. Proposed solution

1. Validate the transaction file and preserve its provenance.
2. Learn a fraud score with Logistic Regression and a LightGBM challenger.
3. Evaluate later transactions using ranking quality, calibration and review workload.
4. Freeze a cutoff on a pre-test policy partition; rank qualifying transactions and cap the queue.
5. Serve the same model through a local API and CLI, with evidence of parity and latency.

## 4. Why this adds to the portfolio

- Project 1 covers sensor maintenance and anomaly/RUL decisions.
- Project 2 covers churn and retention decisions.
- Project 4 adds chronological fraud evaluation, low-budget precision-recall, cost sensitivity and transaction scoring.
- The value comes from reliable evaluation and operational constraints, rather than adding many algorithms or a large dashboard.

These descriptions identify project topics; this planning work did not re-audit the earlier projects' quality or completion.

## 5. Objectives and evidence

| ID | Objective | Required evidence |
|---|---|---|
| BO1 | Keep review volume within the supplied capacity | Queue length <= ceil(review fraction x valid batch rows), with deterministic ties |
| BO2 | Show fraud captured and false alerts at 0.5%, 1%, 2% and 5% workload | Ranking table and separate deployed-policy table |
| BO3 | Explain the effect of assumed costs | Sensitivity report with cost units and no real savings claim |
| MO1 | Establish honest baselines | Constant-prior and Logistic Regression metrics on identical partitions |
| MO2 | Seek better ranking than the baseline | Predeclared selection rule; final AP above test prevalence and Lift@1% > 1 are demo usefulness gates |
| MO3 | Test probability reliability | Brier score, log loss and reliability plot before and after calibration |
| MO4 | Prevent evaluation leakage | Timestamp-disjoint partitions, training-only transforms and a recorded final-test freeze |
| EO1 | Reproduce training and scoring | Pinned environment, config, file hashes, run manifest and command logs |
| EO2 | Make service scores match offline scores | Identical synthetic fixture scores within absolute tolerance 1e-8 |
| EO3 | Demonstrate responsive local scoring | Warm single-request API p95 <= 100 ms on recorded hardware, excluding optional explanation work |
| EO4 | Make failures and changes visible | Validation tests, synthetic CI, container smoke check and monitoring report |

## 6. Improvement targets

- **Stretch:** final AP at least 10% relatively above Logistic Regression, and final Recall@1% at least 2 percentage points higher.
- These are hypotheses, not guaranteed results or champion-selection rules.
- If LightGBM does not justify its complexity, retain Logistic Regression.
- If final usefulness gates fail, report a completed engineering demonstrator with a performance shortfall. Do not present it as suitable for operational adoption.

## 7. Constraints

- Local CPU, solo owner, no paid compute assumed.
- Anonymized benchmark fields do not establish feature availability at a real payment gateway.
- The benchmark label is observed `isFraud`, not a promised future chargeback horizon.
- Class prevalence, sample count, memory demand and actual feature values are measured after acquisition.
- Synthetic cost units do not justify RM-denominated savings or automated blocking.

Dataset facts and methodological sources: [source register](10_SOURCES.md). Evaluation details: [experiment plan](05_EXPERIMENT_PLAN.md).
