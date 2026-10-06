# 10. Source and roadmap register

## 1. Local roadmap evidence

Inspected on **6 October 2026**:

`../Classical_ML_Portfolio_Plan_Malaysia.xlsx`

| Location | Evidence used |
|---|---|
| `Project Tracker!A1` | Suggested roadmap order: 2, 1, 3, then 4-6 |
| `Project Tracker!A6:J6` | Project 3: demand forecasting and inventory optimisation |
| `Project Tracker!A7:J7` | Project 4: fraud detection, precision-recall/cost metrics, real-time API, IEEE-CIS |
| `Datasets!A7:D7` | IEEE-CIS competition URL |
| `Engineer Checklist!A4:A11` | Repo, README, tracking, API, Docker, deployment, monitoring and business impact expectations |
| `Project Tracker!G4:G5` | Both earlier projects still marked Not started, despite user's completion update |

The checklist's public free-tier deployment is deferred from MVP. Local delivery is the planned scope; external deployment requires authorization. Employer examples come from the workbook, not a current vacancies audit.

## 2. Verified primary references

| ID | Reference | Fact or engineering use | Checked |
|---|---|---|---|
| S01 | [IEEE-CIS data description](https://www.kaggle.com/c/ieee-fraud-detection/data) | Binary `isFraud`; transaction/identity files; relative `TransactionDT`; official test labels absent | 6 Oct 2026, search-index description readable |
| S02 | [IEEE-CIS competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules) | Access and reuse terms to inspect at T02 | 6 Oct 2026, full page body not readable through research tool |
| S03 | [scikit-learn metric evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) | AP differs from trapezoidal PR area; precision/recall diagnostics | 6 Oct 2026 |
| S04 | [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | Temporal order and equal-spacing limitation; supports using explicit transaction boundaries | 6 Oct 2026 |
| S05 | [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html) | Calibrator should use data independent of classifier fitting | 6 Oct 2026 |
| S06 | [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) | Class weighting can distort probability estimates; do not combine imbalance knobs blindly | 6 Oct 2026 |

The stable documentation URLs can change versions. T01 must verify the APIs for the versions actually installed. Search-index visibility is not proof of download access or permission to redistribute the competition data.

## 3. Implementation-time official references

- [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [FastAPI request models](https://fastapi.tiangolo.com/tutorial/body/)
- [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/)
- [MLflow documentation](https://mlflow.org/docs/latest/)
- [SHAP documentation](https://shap.readthedocs.io/en/latest/)
- [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)
- [pytest documentation](https://docs.pytest.org/en/stable/)

These links are discovery pointers. Specific APIs and compatible versions still require verification at the relevant task.

## 4. Assumptions, not sourced facts

- 1% review capacity, cost weights, latency target and effort estimates are project defaults.
- The compact feature set and 60/10/10/20 split are proposed engineering decisions.
- No actual bank needs, Malaysian fraud prevalence, realized savings or model quality is established.
- Full competition rules, data acquisition and real field ranges remain unverified.
