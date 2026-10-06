# 11. Risks and later work

## 1. Risk register

| ID | Risk | Likelihood / impact | Mitigation and owner |
|---|---|---|---|
| R01 | Kaggle access or reuse restrictions | Medium / high | T02 confirms terms; user handles consent; PM records blocker |
| R02 | Temporal leakage or test-driven tuning | Medium / high | Timestamp manifests, development-only transforms and freeze gate; ML engineer |
| R03 | Compact predictors miss useful fraud signal | High / medium | Compare baselines honestly; wider features require availability audit; data scientist |
| R04 | Unknown anonymized-field availability | High / high for real deployment | State benchmark input boundary; no production payment claim; ML engineer |
| R05 | Imbalance creates misleading accuracy | High / high | AP, capacity recall, prevalence and false alerts; data scientist |
| R06 | Class weighting distorts probabilities | Medium / medium | Independent calibration and reliability report; data scientist |
| R07 | Memory or training budget exceeded | Medium / medium | Selected columns, sparse LR, thread/search caps, development-only smoke run; ML engineer |
| R08 | Model quality changes over time | High / high | Later holdout, elapsed-time error analysis and monitoring; ML engineer |
| R09 | Frozen cutoff overfills candidate list or underfills queue | Medium / medium | Deterministic cap and separate candidate/selected counts; ML engineer |
| R10 | Cost simulation mistaken for real savings | Medium / high | Synthetic units, sensitivity and explicit review assumptions; PM |
| R11 | Model artifact or payload handling causes exposure | Medium / high | Trusted local bundles, strict schemas, size caps, payload-free logs; ML engineer |
| R12 | Docker/CI cannot be verified | Medium / medium | Keep gate open and report actual limitation; ML engineer |
| R13 | Scope expands into UI, streaming or production banking | Medium / high | PRD Must scope, task dependencies and backlog; PM |
| R14 | Time-dependent bootstrap has insufficient support | Medium / medium | Report block/positive counts and interval limitations; data scientist |

## 2. Stop or re-plan conditions

1. Data cannot be obtained within verified terms: stop real-data work, preserve synthetic scaffolding and request dataset direction.
2. Chronological partitions cannot support the specified classes: profile and request an evaluation decision before training.
3. Holdout was used for tuning: disclose invalidated evaluation and propose a new protocol; do not relabel the same test as unseen.
4. Hardware cannot finish the bounded search: reduce search breadth before execution and record the change; do not quietly replace full evaluation with a smoke subset.
5. Final usefulness fails: release only as a transparent portfolio experiment, with the shortfall visible.

## 3. Later backlog, after MVP

| Item | Reason | Prerequisite |
|---|---|---|
| Streamlit exploration dashboard | Nontechnical demo | Stable score/queue/report contracts |
| Wider transaction features | Potential ranking gain | Availability audit, feature semantics and new recorded experiment |
| Identity join | Additional signals | Join integrity, missing-identity policy, serving availability and terms |
| Delayed-label/embargo study | More realistic payment feedback | Label availability assumptions or actual timestamps |
| Entity-history features | Past-behaviour signals | Stateful event ordering and leakage-safe history contract |
| Public demo hosting | Shareable portfolio | User authorization, sanitized artifacts and reuse permission |
| Authenticated production service | Real integration | Stakeholder requirements, threat model and operational review |
| Human review feedback study | Validate cost assumptions | Actual workflow, outcomes and permitted data |

## 4. PM review routine

- At each milestone, compare evidence with PRD and objectives.
- Each week, review hours spent, risks, remaining tasks and scope.
- Keep model-quality misses separate from software defects.
- Use [the task tracker](../tasks/todo.md) as the only completion checklist. Log changes to estimates and decisions without silently editing measured outcomes.
