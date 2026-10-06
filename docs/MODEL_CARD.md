# FraudGuard Model Card

## 1. Model Details & Provenance

| Field | Value at Handoff |
|---|---|
| **Model Family** | LightGBM Classifier (`LGBM_unweighted_n200_lr0.03_l15`) |
| **Hyperparameters** | `n_estimators=200`, `learning_rate=0.03`, `num_leaves=15`, `min_child_samples=50`, `weight_mode=unweighted`, `random_state=42` |
| **Probability Calibration** | Raw probabilities selected (Brier 0.02115 on independent 4,998-row calibration partition) |
| **Policy Decision** | Cutoff $\tau = 0.529162$ derived on policy partition (5,000 rows) for 1% review capacity |
| **Model & Schema Versions** | Model: `fraudguard-champion-v1`, Policy: `review-v1-frozen`, Schema: `1.0.0` |
| **Feature Allowlist** | 12 raw transaction fields (`TransactionAmt`, `ProductCD`, `dist1`, `dist2`, `card4`, `card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M4`, `M6`) and temporal timestamp |
| **Hardware & Environment** | Python 3.10.11, Windows 11, x86_64, pinned in `requirements-lock.txt` |
| **Reproducibility Verification** | Exact 0.000000 AP & Brier difference on retrain (`reports/repeatability/reproducibility.json`) |

---

## 2. Intended Use & Boundaries

- **Intended Use:** High-throughput transaction risk scoring and generation of a top-ranked, capacity-constrained manual review queue (1% operational budget).
- **Out of Scope:** Automated transaction blocking, live payment gateway integration, cardholder profiling without human review, credit decisioning.

---

## 3. Final Evaluation & Measured Metrics

Evaluated on the frozen chronological `final_test` partition (10,000 transactions, 356 fraud cases, 3.56% prevalence):

| Metric | Champion (LightGBM) | 95% Bootstrap CI | Baseline (Logistic Regression) | Amount-Rank Baseline | Constant-Prior Baseline | Target / Status |
|---|---|---|---|---|---|---|
| **Average Precision (AP)** | **0.4753** | **[0.4102, 0.5344]** | 0.4177 | 0.0865 | 0.0356 | $> 0.0865$ (MET) |
| **ROC-AUC** | **0.9526** | **[0.9380, 0.9650]** | 0.9312 | 0.6756 | 0.5000 | $> 0.8000$ (MET) |
| **Precision@1%** | **0.6700** | **[0.5500, 0.7800]** | 0.6500 | 0.1300 | 0.0400 | $> 0.3000$ (MET) |
| **Recall@1%** | **0.1882** | **[0.1480, 0.2228]** | 0.1826 | 0.0365 | 0.0112 | $> 0.0652$ (MET) |
| **Lift@1%** | **18.82x** | **[15.45x, 21.91x]** | 18.26x | 3.65x | 1.12x | $> 5.0x$ (MET) |
| **Brier Score** | **0.0243** | **[0.0210, 0.0280]** | 0.0262 | 0.0341 | 0.0343 | $< 0.0350$ (MET) |
| **Simulated Cost** | **29,066.0** | — | 29,270.0 | 34,574.0 | 35,600.0 | Savings: 6,534.0 units (18.4% MET) |

*Simulated Cost formula: $100 \times \text{FN} + 1 \times (\text{TP} + \text{FP}) + 2 \times \text{FP}$.*

### API Performance & Latency (1,000 Sequential Requests)
- **p50:** 8.58 ms
- **p95:** 9.67 ms (Target $\le 100$ ms: **MET**)
- **p99:** 11.23 ms
- **Mean:** 8.67 ms
- **Error Rate:** 0.0% (1,000 / 1,000 successful)

---

## 4. Explainability Summary

- **Global Feature Importance:** Top predictive features by permutation AP drop on policy partition:
  1. `ProductCD` (+0.278 AP drop)
  2. `P_emaildomain` (+0.126 AP drop)
  3. `TransactionAmt` (+0.071 AP drop)
  4. `addr1` (+0.044 AP drop)
  5. `dist1` (+0.018 AP drop)
- **Local SHAP Explanations:** TreeExplainer computed on 200 policy transactions. SHAP additivity verified with maximum discrepancy $1.32 \times 10^{-14} \le 10^{-4}$.

---

## 5. Limitations & Ethical Considerations

1. **Benchmark Fixture Context & Synthetic Rule Recovery:** Evaluated on a 50,000-row synthetic benchmark fixture (`train_transaction.csv`) spanning 24.85 days (~25 days). The performance separation where LightGBM improves over Logistic Regression reflects recovery of planted interaction rules (night-hour risks, high-risk email domains, ProductCD amounts) rather than real fraud detection efficacy on IEEE-CIS. Full-scale real evaluation requires the Kaggle IEEE-CIS dataset.
2. **Uncertainty Bounds:** The ~5-day temporal span of the final test partition provided adequate distinct day blocks for block bootstrap resampling (1,000 valid stationary block bootstrap replicates).
3. **Label Delay:** Production fraud labels typically experience 30–90 day chargeback lag; real-time performance must be monitored using unlabelled distribution drift indicators.
4. **Capacity Constraints:** Fixed 1% manual review cap ensures analyst queues are not overwhelmed, capturing 67 fraud cases out of 100 reviewed transactions.
5. **Demographic & Fair Lending:** No protected personal attributes (race, gender, age) are utilized. Features are strictly transactional and routing metadata.
