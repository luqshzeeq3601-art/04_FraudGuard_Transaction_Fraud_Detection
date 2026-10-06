# Development Partition Exploratory Data Analysis

## 1. Partition Scope and Safety
- **Partition:** Development (Earliest ~60% of chronological transactions)
- **Total Rows:** 120
- **Positive Fraud Cases:** 11
- **Fraud Prevalence:** 9.1667%
- **Holdout Isolation Note:** Final test, policy, and calibration partitions are excluded from this analysis.

## 2. Five Key Data Observations
1. **Target Imbalance:** Fraud prevalence is approximately 9.17%, requiring ranking metrics (Average Precision, Precision@1%, Recall@1%) rather than accuracy or ROC-AUC alone.
2. **Transaction Amount Distribution:** Highly right-skewed with median 81.89 and maximum 863.00. `log1p(TransactionAmt)` is appropriate for linear modeling.
3. **Missing Value Structure:** Nullable distance features (`dist1`, `dist2`) and return email domains (`R_emaildomain`) exhibit high missing rates (>50%), requiring explicit missingness indicators.
4. **Product Code Segments:** ProductCD distribution is dominated by category `W` (81 rows), while smaller segments require robust one-hot encoding.
5. **Card Type Categories:** Debit cards dominate over credit cards (89 vs 29), with distinct risk profiles across payment mechanisms.

## 3. Numeric Summary (Development Only)
| Statistic | TransactionAmt |
|---|---|
| Count | 120 |
| Mean | 119.70 |
| Std | 128.65 |
| Min | 3.96 |
| 25% | 39.20 |
| 50% (Median) | 81.89 |
| 75% | 154.83 |
| Max | 863.00 |
