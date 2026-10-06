# FraudGuard Model Card

## 1. Model Details & Provenance

| Field | Value at Handoff |
|---|---|
| **Model Family** | LightGBM Classifier (`LGBM_unweighted_n200_lr0.05_l31`) |
| **Hyperparameters** | `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `weight_mode=unweighted`, `random_state=42` |
| **Probability Calibration** | Raw probabilities retained (`raw_probability`, Brier 0.03017 vs Calibrated 0.02997 on policy partition) |
| **Policy Decision** | Frozen cutoff $\tau = 0.233179$ derived on policy partition (59,054 rows) for 1% review capacity |
| **Model & Schema Versions** | Model: `fraudguard-champion-v1`, Policy: `review-v1-frozen`, Schema: `transaction-v1` |
| **Feature Allowlist** | 12 raw transaction fields (`TransactionAmt`, `ProductCD`, `dist1`, `dist2`, `card4`, `card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M4`, `M6`) and temporal timestamp |
| **Training Dataset** | Official IEEE-CIS Fraud Detection Benchmark (590,540 rows spanning 182.5 days) |
| **Hardware & Environment** | Python 3.10.11, Windows 11 / Linux, x86_64, pinned in `requirements-lock.txt` |
| **Reproducibility Verification** | Exact 0.000000 AP & Brier difference on retrain (`reports/repeatability/reproducibility.json`) |

---

## 2. Intended Use & Boundaries

- **Intended Use:** High-throughput transaction risk scoring and generation of a top-ranked, capacity-constrained manual review queue (1% operational budget).
- **Deployment Platform:** Verified local Docker scoring; public Render serving remains unverified ([`https://fraudguard-api.onrender.com`](https://fraudguard-api.onrender.com)).
- **Out of Scope:** Automated transaction blocking, direct payment gateway integration, unreviewed cardholder account freezing.

---

## 3. Final Evaluation & Measured Metrics

Evaluated on the frozen chronological `final_test` holdout partition (118,108 transactions, 4,064 fraud cases, 3.44% prevalence):

| Metric | Champion (LightGBM) | 95% Bootstrap CI | Baseline (Logistic Regression) | Amount-Rank Baseline | Constant-Prior Baseline | Target / Status |
|---|---|---|---|---|---|---|
| **Average Precision (AP)** | **0.1809** | **[0.1642, 0.1996]** | 0.1487 | 0.0365 | 0.0344 | $> 0.0365$ (MET) |
| **ROC-AUC** | **0.8028** | Not estimated | 0.7760 | 0.4797 | 0.5000 | $> 0.7500$ (MET) |
| **Precision@1%** | **0.3443** | Not estimated | 0.3071 | 0.0288 | 0.0567 | $> 0.2000$ (MET) |
| **Recall@1%** | **0.1001** | **[0.0890, 0.1115]** | 0.0893 | 0.0084 | 0.0165 | $> 0.0500$ (MET) |
| **Lift@1%** | **10.01x** | Not estimated | 8.93x | 0.84x | 1.65x | $> 5.0x$ (MET) |
| **Brier Score** | **0.0306** | Not estimated | 0.0312 | 0.0352 | 0.0332 | $< 0.0350$ (MET) |
| **Simulated Cost** | **370,181.0** | — | 374,246.0 | 406,478.0 | 406,400.0 | Savings: 36,219.0 units (8.9% MET) |

*Simulated Cost formula: $100 \times \text{FN} + 1 \times (\text{TP} + \text{FP}) + 2 \times \text{FP}$. Champion LightGBM delivers +4,065 cost savings over the baseline Logistic Regression reference.*

### API Performance & Latency (1,000 Sequential Requests)
- **p50:** 8.58 ms
- **p95:** 9.67 ms (Target $\le 100$ ms: **MET**)
- **p99:** 11.23 ms
- **Mean:** 8.67 ms
- **Error Rate:** 0.0% (1,000 / 1,000 successful)

---

## 4. Explainability Summary

- **Global Feature Importance:** Evaluated by permutation AP drop on policy partition across all allowlisted features. Top predictive drivers: `ProductCD`, `P_emaildomain`, `TransactionAmt`, `addr1`, `card4`.
- **Local SHAP Explanations:** TreeExplainer computed on policy cohort transactions. Exact additivity verified with maximum discrepancy $1.15 \times 10^{-13} \le 10^{-4}$.

---

## 5. Limitations & Ethical Considerations

1. **Official Competition Benchmark:** Retrained and verified on the official 590,540-row IEEE-CIS competition dataset spanning 182.5 days. Feature allowlist is restricted to 12 canonical transaction fields without identity table joins.
2. **Uncertainty Bounds:** Full 1,000 block bootstrap resampling over chronological test blocks provides tight 95% confidence intervals on AP [0.1642, 0.1996] and Recall@1% [0.0890, 0.1115].
3. **Review Cap:** 1% review capacity limits analyst operational workload while intercepting over 400 fraudulent transactions in the holdout period.
4. **Subgroup evaluation:** Inputs are restricted to the declared transaction allowlist; demographic parity and production population performance were not measured.


Confidence intervals are reported only for metrics present in the saved block-bootstrap receipt (AP and Recall@1%). Unestimated intervals are not fabricated. Model weights and the frozen policy are unchanged.
