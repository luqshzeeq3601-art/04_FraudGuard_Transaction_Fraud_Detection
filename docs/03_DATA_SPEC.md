# 03. Data specification

## 1. Source and access

- Primary source: [IEEE-CIS competition data](https://www.kaggle.com/c/ieee-fraud-detection/data).
- MVP input: `train_transaction.csv`; target: `isFraud`.
- Official competition test files have no public target labels. They are not the local final evaluation set.
- Identity information is a separate table and is incomplete across transactions. It is excluded from MVP.
- The source lists reuse as subject to competition rules. Terms, access and redistribution permission are **unverified** until T02. See [sources](10_SOURCES.md).

At T02, record download URL/time, exact filenames and sizes, SHA-256 hashes, access method and a dated summary of applicable terms in `data/raw/DATA_PROVENANCE.md`. Keep archives and raw data local and ignored by Git.

Do not substitute another dataset silently. If access fails, synthetic fixtures can validate software but cannot support fraud-quality claims.

## 2. Input contracts

| Field | Role | Validation |
|---|---|---|
| `TransactionID` | Identifier, output and tie-break only | Unique positive integer; reject duplicates in a batch |
| `TransactionDT` | Partitioning and monitoring only | Finite nonnegative number; elapsed seconds from an unspecified reference |
| `isFraud` | Training/evaluation target only | Integer 0 or 1; absent from API and scoring CSV |
| `TransactionAmt` | Numeric predictor | Required finite nonnegative number; never relabel as RM |
| `ProductCD` | Categorical predictor | Required nonempty string, <= 128 characters |
| `dist1`, `dist2` | Nullable numeric predictors | Null or finite nonnegative number; verify source distribution at T03 |
| `card4`, `card6` | Nullable categorical predictors | Null or nonempty string, <= 128 characters |
| `addr1`, `addr2` | Nullable categorical predictors | Null or nonnegative integer code, normalized to string internally |
| `P_emaildomain`, `R_emaildomain` | Nullable categorical predictors | Null or nonempty string, <= 128 characters; domains are not full email addresses |
| `M4`, `M6` | Nullable categorical predictors | Null or nonempty string, <= 128 characters |

**Exactly 12 raw predictors:** `TransactionAmt`, `ProductCD`, `dist1`, `dist2`, `card4`, `card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M4`, `M6`.

- Raw training ingestion selects required fields and ignores other original benchmark columns intentionally. Record dropped column names.
- Serving/scoring require all envelope and predictor keys/columns. Nullable fields may contain null/blank, but missing columns/keys are errors.
- Reject unknown API/scoring fields, including `isFraud`. Training ingestion has a separate labelled schema.
- Reject booleans used as numbers, nonfinite values, numeric-string API coercion and invalid ID types.
- CSV ingestion permits documented numeric parsing and blank-to-null conversion; errors include row/field details without dumping payloads.
- If actual official values contradict a range, profile and resolve the rule before dropping rows or fitting a model.

## 3. Predictor restrictions

- Exclude `TransactionID`, raw `TransactionDT` and `isFraud` from model features.
- Exclude all identity, C, D and V groups in MVP. Their point-of-decision availability and semantics are not established here.
- Do not derive calendar dates, weekdays or actual local hours from the relative time origin.
- Allowed row-local derivation: `log1p(TransactionAmt)` and numeric missingness indicators.
- No label aggregates, target encoding, full-dataset category frequencies, future lookups or entity histories.
- The compact feature set prioritizes a tractable API contract. It may reduce performance. Wider feature experiments require a new decision and availability audit.

## 4. Preprocessing

1. Share normalization and feature creation across training, batch and API.
2. Fit numeric medians on training rows only; a fully missing numeric training column falls back to 0 with an explicit diagnostic.
3. Preserve missingness indicators for nullable numerics.
4. Map categorical nulls to a reserved missing category and normalize address codes without implying numeric magnitude.
5. Fit sparse one-hot encoding on training data only, with infrequent-category grouping where supported by the pinned version. Handle unseen values without refitting; record unknown rates.
6. Scale numeric inputs for LR; use the same raw allowlist for LightGBM with model-appropriate fitted preprocessing. Both must have their own complete trusted bundle.

## 5. Chronological partitions

1. Sort by `TransactionDT`, then `TransactionID`.
2. Compute proposed row cut points at 60%, 70% and 80% of the labelled file without using labels.
3. Move each cut to the end of its equal-`TransactionDT` group. Keep timestamps entirely within one partition.
4. Save exact IDs, row counts, elapsed-time ranges and boundary rules in `data/processed/split_manifest.json`.

| Partition | Approximate fraction | Permitted use |
|---|---|---|
| Development | Earliest 60% | EDA, training, internal temporal folds and model selection |
| Calibration | Next 10% | Fit sigmoid mapping for a frozen development-trained model |
| Policy | Next 10% | Choose raw/calibrated mapping, freeze cutoff and report pre-test policy |
| Final test | Latest 20% | One frozen final evaluation, then diagnostic analysis without retuning |

- Fractions can vary because tied timestamps stay together. Fail if a partition is empty or boundaries cease to be chronological.
- Require both classes in development, calibration and policy before the full model protocol. A class-deficient final test makes affected metrics unavailable; report it rather than redefining the holdout.
- Schema integrity checks may scan the file. Development analysis must not inspect final-test label distributions, correlations or performance before T13.
- Final-test labels are stored separately from serving inputs and kept out of training command paths. Record access in the evaluation log.

## 6. Internal folds

Within development, create three expanding folds: first approximately 50% -> next 1/6, first 2/3 -> next 1/6, first 5/6 -> final 1/6. Move boundaries to keep equal timestamps together. All transforms are refit inside each fold.

Use explicit elapsed-time-aware manifests, not shuffled CV. Transaction spacing is irregular; report each fold's elapsed duration rather than claiming equal-duration folds from equal row counts.

The split measures later-transaction performance. It does not establish performance on unseen customers/cards, label-delay realism or a new country. A delayed-label/embargo evaluation is later work because label-availability times are absent.

## 7. Quality and memory gate

T03 reports row count, selected-column types, missingness and invalid values. Development EDA reports prevalence, amount distributions, category counts and at least five useful observations.

Read only selected columns, use explicit types after validation and record peak memory. Start with synthetic fixtures and a development-only smoke subset if needed. Full evaluation uses all permitted rows; a subset run must be labelled a smoke run.

## 8. Planned local paths

| Path | Content |
|---|---|
| `data/raw/` | Original file/archive and provenance |
| `data/processed/` | Validated selected columns and partition manifests |
| `tests/fixtures/` | Hand-authored synthetic inputs and labels |
| `reports/data_quality.json` | Schema and development profile evidence |

Raw data, row-level derived scores, model bundles and local tracking stores are ignored by Git. No data-derived rows enter synthetic fixtures or public examples.
