# Data Provenance and Acquisition Record

## 1. Primary Source
- **Dataset:** Official IEEE-CIS Fraud Detection Benchmark (`train_transaction.csv`)
- **Origin:** IEEE Computational Intelligence Society / Vesta Corporation (hosted on Kaggle: `https://www.kaggle.com/c/ieee-fraud-detection/data`)
- **Target Field:** `isFraud` (Binary: 0 = Legitimate, 1 = Fraudulent)
- **Evaluation Set Note:** Official competition test file contains unlabelled transactions. The local final test set is defined strictly chronologically as the latest 20% of `train_transaction.csv` (118,108 transactions).

## 2. Acquisition Status & Provenance
- **Method:** Automated Kaggle API acquisition (`kaggle datasets download -d lnasiri007/ieeecis-fraud-detection`)
- **File:** `data/raw/train_transaction.csv`
- **File Size:** 683,351,067 bytes
- **SHA-256 Hash:** `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642`
- **Rows:** 590,540 transactions (182.5 days temporal duration, 20,663 fraud cases, 3.499% prevalence)
- **Allowlisted Features:** Exactly 12 raw predictors (`TransactionAmt`, `ProductCD`, `dist1`, `dist2`, `card4`, `card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M4`, `M6`) plus envelope keys (`TransactionID`, `TransactionDT`) and target (`isFraud`).
- **Chronological Partitions:**
  - `development` (60%): 354,324 rows (11,988 fraud, 3.3830%)
  - `calibration` (10%): 59,054 rows (2,550 fraud, 4.3180%)
  - `policy` (10%): 59,054 rows (2,061 fraud, 3.4900%)
  - `final_test` (20%): 118,108 rows (4,064 fraud, 3.4410%)

## 3. Privacy & Compliance
- No customer PII, credit card numbers, or live credentials are stored.
- Address codes (`addr1`, `addr2`) and email domains are categorical attributes only.
- Raw datasets and model binaries are excluded from Git version control via `.gitignore`.
