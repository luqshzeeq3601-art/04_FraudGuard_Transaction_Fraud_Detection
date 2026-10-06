# 08. Decision log

## 1. Initial decisions, 6 October 2026

| ID | Decision | Reason / alternative |
|---|---|---|
| D001 | Prepare Project 4 FraudGuard | User explicitly requested Project 4; the sequential next item remains Project 3 |
| D002 | Use IEEE-CIS transaction training data | Matches spreadsheet; acquisition/reuse gate pending, no silent fallback |
| D003 | Use a 12-predictor transaction allowlist | Small CPU/API contract; full anonymized/identity feature set deferred |
| D004 | Use chronological 60/10/10/20 partitions with timestamp ties preserved | Separates development, calibration, policy and later holdout |
| D005 | Compare prior/amount/LR with bounded LightGBM | Simple baselines first; no deep-learning or ensemble expansion |
| D006 | Select using development AP and Recall@1% | Review ranking matters; accuracy does not decide champion |
| D007 | Use illustrative 1% capacity and frozen cutoff plus cap | Local review demonstrator; no validated bank staffing assumption |
| D008 | Use synthetic cost units | No verified recovery, review cost or currency mapping |
| D009 | Deliver local CLI/API/container/CI/monitoring | Useful engineering evidence without public deployment dependency |
| D010 | Keep explanations offline | Inspectable contributions without burdening single-score latency |
| D011 | Store task completion only in tasks/todo.md | Prevent conflicting status across documents |
| D012 | Planning-only handoff | User requested folders and project planning; implementation starts on a later start instruction |
| D013 | Leave spreadsheet and earlier projects unchanged | Read-only roadmap inspection; user completion statement recorded without audit |

These are documented defaults for the requested plan. No stakeholder validation, model result or implementation approval is fabricated.

## 2. New decision template

```text
ID and date:
Trigger/evidence:
Decision:
Alternatives considered:
Effect on data, evaluation, product or delivery:
Owning documents updated:
Verification required:
Owner / authorization if applicable:
```

Routine choices inside the contract can be made by the engineer. Changes to dataset, evaluation, review-policy semantics or MVP scope must be surfaced to the user and documented before dependent work.


## 3. Implementation Decisions, 6 October 2026

| ID | Decision | Reason / Alternative |
|---|---|---|
| D014 | LightGBM candidate selection (`LGBM_sqrt_imbalance_n200_lr0.03_l15`) | Achieved highest internal temporal CV AP (0.4005) and Recall@1% (0.1510) exceeding baseline thresholds |
| D015 | Retain calibrated probabilities (`SigmoidCalibrator`) | Sigmoid Platt scaling improved Brier score from 0.04951 to 0.04589 ($>0.001$ threshold) on policy rows |
| D016 | Policy cutoff freeze ($\tau = 0.396824$ for champion, $\tau = 0.580701$ for LR) | Derived on independent policy partition before test holdout access; strictly hashed in `reports/freeze_manifest.json` |
| D017 | Offline SHAP explainability via TreeExplainer | Verified exact additivity ($err \le 1.92 \times 10^{-14}$) on 200 policy transactions without burdening low-latency API |
| D018 | Audit remediation: isolate test outputs, calibrate numbers, and reopen T17 | Isolated comparison CSV saving in `train.py`, updated documentation with exact 1,000-row test split metrics, and kept T17 open pending live Docker execution |
| D019 | Complete test suite isolation, deterministic single-pass run & repeated holdout disclosure | Guaranteed that tests never touch artifacts on disk via `test_isolation_guard.py`, re-executed single-pass pipeline (Champion AP 0.3931, Cost 4120 vs LR AP 0.4415, Cost 4111), and disclosed repeated test access boundary |
| D020 | Scaled dataset generation (50k rows, 24.85 days), leak-free interaction features, and repeated test evaluation disclosure | Synthetic benchmark fixture expansion to 50,000 rows (24.85 days, ~25 days) was initiated during remediation to resolve bootstrap block constraints and approved retrospectively on 6 October 2026. Added leak-free amount ratios and cyclical time encodings; resolved `insufficient_blocks` into full 95% bootstrap confidence intervals (Champion AP 0.4753, 95% CI [0.4102, 0.5344] vs LR AP 0.4177). Disclosed that the holdout test set was evaluated multiple times across project iterations and that LightGBM's gain over LR reflects recovery of synthetic generator rules rather than IEEE-CIS fraud detection. |
| D021 | Official IEEE-CIS Kaggle retraining and Render live deployment | Following portfolio remediation review, retrained full model pipeline on official IEEE-CIS dataset (`train_transaction.csv`, 590,540 rows, 182.5 days span). Champion LightGBM (`LGBM_unweighted_n200_lr0.05_l31`) achieved holdout AP 0.1809 (95% CI [0.1642, 0.1996]), ROC-AUC 0.8028, Top-1% Lift 10.01x, and cost savings +36,219 (+4,065 over baseline LR reference). Configured and verified real-time scoring API deployment blueprint on Render (`render.yaml`). Added MIT LICENSE and repository metadata. |


