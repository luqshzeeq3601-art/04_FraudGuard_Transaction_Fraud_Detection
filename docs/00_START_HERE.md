# 00. Execution guide

## 1. What to deliver

Deliver a small, defensible fraud-review demonstrator: transaction features enter one shared scoring pipeline; fraud-risk scores and a capped review queue come out. Prove model quality on later transactions and prove the service matches the offline model.

The implementer acts as senior ML engineer and project manager: validate the data and evaluation first, keep the delivery small, record evidence and manage dependencies.

## 2. Reading order

1. `README.md` and `AGENTS.md` at the project root.
2. [Problem and objectives](01_PROBLEM_AND_OBJECTIVES.md), then [PRD](02_PRD.md).
3. [Plan](../tasks/plan.md) and [task tracker](../tasks/todo.md).
4. [Data specification](03_DATA_SPEC.md) and [experiment plan](05_EXPERIMENT_PLAN.md) before any modelling.
5. [Technical design](04_TECHNICAL_DESIGN.md) and [commands](07_OPERATIONS_AND_COMMANDS.md) before implementation.
6. [Validation and release](06_VALIDATION_AND_RELEASE.md) before closing a milestone.
7. [Decisions](08_DECISIONS_LOG.md), [progress](09_PROGRESS_LOG.md) and [risks](11_RISKS_AND_BACKLOG.md) whenever resuming.

Read [sources](10_SOURCES.md) for evidence and [the model card](MODEL_CARD.md) before making public claims.

## 3. Defaults to execute

| Item | Contract |
|---|---|
| Dataset | IEEE-CIS labelled transaction training file only |
| Feature scope | 12 allowlisted raw predictors; no identity join or opaque C/D/V features |
| Development split | Earliest approximately 60% |
| Calibration / policy / test | Next approximately 10% / 10% / latest 20% |
| Models | Constant prior, Logistic Regression, LightGBM |
| Main measures | Average precision (AP), Recall@1%, Precision@1%, Lift@1% |
| Default workload | At most 1% of the supplied batch, rounded up; batch ceiling still applies |
| Policy | Frozen score cutoff plus a deterministic capacity cap |
| Interface | CLI and FastAPI; benchmark feature payload, no payment connector |
| Task authority | `tasks/todo.md` only |

## 4. First execution session

1. Start **T01**: inspect Python, Git, available memory and Docker. Create the local environment and minimal package.
2. Start **T02**: confirm Kaggle access and actual applicable terms. If consent is required, record the exact blocker and continue synthetic engineering work where dependencies allow.
3. Complete T03-T04 before fitting models. Full-data training remains blocked until acquisition, schema and split gates pass.
4. Use the commands document as a future interface specification. Those commands do not work yet.

## 5. Resume procedure

1. Read the latest log entry and inspect actual artifacts, not just their recorded status.
2. Find the first unchecked task with completed prerequisites.
3. Load only its relevant specifications and code paths.
4. Complete the bounded task, run its checks and record results.
5. Continue within the authorized session. Record a precise blocker when dependent work cannot proceed.

## 6. Document authority

| Question | Owning document |
|---|---|
| What problem and outcome matter? | `01_PROBLEM_AND_OBJECTIVES.md` |
| Which features and behaviours belong in MVP? | `02_PRD.md` |
| What inputs, predictors and partitions are valid? | `03_DATA_SPEC.md` |
| What are the API, stack and artifact contracts? | `04_TECHNICAL_DESIGN.md` |
| How are models, calibration and policy chosen? | `05_EXPERIMENT_PLAN.md` |
| What evidence closes a gate? | `06_VALIDATION_AND_RELEASE.md` |
| Which commands must exist? | `07_OPERATIONS_AND_COMMANDS.md` |
| What happens next? | `tasks/todo.md` and `09_PROGRESS_LOG.md` |

Update the owning file first when changing a contract, then update linked files and record a decision. The roadmap has been extracted here; execution does not depend on re-reading Excel.
