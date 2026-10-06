# Development Partition Exploratory Data Analysis

## 1. Partition Scope and Safety
- **Partition:** Development (Earliest ~60% of chronological transactions)
- **Total Rows:** 354324
- **Positive Fraud Cases:** 11988
- **Fraud Prevalence:** 3.3833%
- **Holdout Isolation Note:** Final test, policy, and calibration partitions are excluded from this analysis.

## 2. Five Key Data Observations
1. **Target Imbalance:** Fraud prevalence is approximately 3.38%, requiring ranking metrics (Average Precision, Precision@1%, Recall@1%) rather than accuracy or ROC-AUC alone.
2. **Transaction Amount Distribution:** Highly right-skewed with median 70.00 and maximum 31937.39. `log1p(TransactionAmt)` is appropriate for linear modeling.
3. **Missing Value Structure:** Nullable distance features (`dist1`, `dist2`) and return email domains (`R_emaildomain`) exhibit high missing rates (>50%), requiring explicit missingness indicators.
4. **Product Code Segments:** ProductCD distribution is dominated by category `W` (252135 rows), while smaller segments require robust one-hot encoding.
5. **Card Type Categories:** Debit cards dominate over credit cards (257663 vs 95802), with distinct risk profiles across payment mechanisms.

## 3. Numeric Summary (Development Only)
| Statistic | TransactionAmt |
|---|---|
| Count | 354324 |
| Mean | 134.32 |
| Std | 237.95 |
| Min | 0.29 |
| 25% | 43.38 |
| 50% (Median) | 70.00 |
| 75% | 125.50 |
| Max | 31937.39 |
