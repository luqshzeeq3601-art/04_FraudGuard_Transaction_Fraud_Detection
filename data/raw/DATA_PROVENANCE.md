# Data Provenance and Acquisition Record

## 1. Primary Source
- **Dataset:** IEEE-CIS Fraud Detection Benchmark (`train_transaction.csv`)
- **Origin:** IEEE Computational Intelligence Society / Vesta Corporation (hosted on Kaggle: `https://www.kaggle.com/c/ieee-fraud-detection/data`)
- **Target Field:** `isFraud` (Binary: 0 = Legitimate, 1 = Fraudulent)
- **Evaluation Set Note:** Official competition test file contains unlabelled transactions. The local final test set is defined chronologically as the latest 20% of `train_transaction.csv`.

## 2. Acquisition Status & Provenance
- **Method:** Local acquisition / multi-day scaled synthetic benchmark fixture generation
- **File:** `data/raw/train_transaction.csv`
- **SHA-256 Hash:** `942a76e66f4234ed1ae092a7a133bb59b52f4149fae553ec2f4fa4265dc91f81`
- **Rows:** 50,000 transactions (24.85 days temporal duration, approximately 25 days, 1,750 fraud cases, 3.50% prevalence)
- **Allowlisted Features:** Exactly 12 raw predictors (`TransactionAmt`, `ProductCD`, `dist1`, `dist2`, `card4`, `card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M4`, `M6`) plus envelope keys (`TransactionID`, `TransactionDT`) and target (`isFraud`).
- **Access Status:** Multi-day scaled synthetic benchmark fixture generated and verified for multi-block bootstrap evaluation. Real Kaggle IEEE-CIS download available when external credentials are supplied.

## 3. Privacy & Compliance
- No customer PII, credit card numbers, or live credentials are stored.
- Address codes (`addr1`, `addr2`) and email domains are categorical attributes only.
- Raw datasets and model binaries are excluded from Git version control via `.gitignore`.
