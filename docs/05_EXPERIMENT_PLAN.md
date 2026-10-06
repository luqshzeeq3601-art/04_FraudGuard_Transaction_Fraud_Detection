# 05. Experiment and decision plan

## 1. Nonnegotiable protocol

1. Freeze the data hash, predictor allowlist and timestamp-disjoint partitions before modelling.
2. Select model family/hyperparameters on the three development folds only.
3. Fit the selected classifier once on all development rows.
4. Fit sigmoid calibration on calibration rows without refitting the classifier.
5. Choose raw/calibrated mapping and freeze a cutoff on policy rows.
6. Save the complete configuration/bundle hashes before T13 reads final-test labels.
7. Evaluate the frozen candidate once. Further final-test analysis is diagnostic; improvements require a new stated protocol or new holdout.

Neither calibration nor policy rows become classifier training rows for the MVP final evaluation. No refit on all labelled data after test is required.

## 2. Experiments

| ID | Model / change | Purpose |
|---|---|---|
| E00 | Constant training prevalence | No-skill probability reference |
| E01 | Amount-descending rule | Simple review-ranking reference; no probability claims |
| E02 | LR, no class weighting | Interpretable supervised baseline |
| E03 | LR, balanced weights | Test imbalance trade-off and calibration |
| E04 | LightGBM, unweighted | CPU challenger on the same raw predictors |
| E05 | LightGBM, positive weight = sqrt(negative/positive count) | Test modest cost emphasis; compute ratio inside each training fold |
| E06 | Bounded hyperparameter search | Improve the most promising LR and boosting variants |
| E07 | Raw versus sigmoid score mapping | Check probability reliability after weighting |
| E08 | Cutoff and capacity simulation | Evaluate a frozen workload policy and assumed costs |

Seed: 42. Cap threads at 4 or fewer based on actual hardware. Record training/runtime/memory for every experiment.

## 3. Search budget

- LR: `C` in {0.1, 1.0, 10.0}, class weights in {none, balanced}; at most 6 configurations.
- LightGBM: at most 12 seeded configurations drawn from `num_leaves` {15, 31, 63}, `learning_rate` {0.03, 0.05, 0.1}, `n_estimators` {200, 400}, `min_child_samples` {50, 100, 200}, weight {1, sqrt(train imbalance)}.
- Use deterministic CPU settings where supported by the pinned version. No GPU or unbounded tuning.
- No early stopping against calibration, policy or final-test rows. Fixed estimator counts keep the initial search protocol simple.
- If hardware is constrained, reduce the search budget before running and record why. Smoke-subset metrics are not full-data results.

## 4. Metric definitions

| Metric | Meaning / rule |
|---|---|
| AP | `average_precision_score`; primary ranking summary. Do not interchange with trapezoidal PR-AUC. |
| ROC-AUC | Supporting discrimination measure; unavailable when one class is absent |
| Precision@q | Fraud count in unconditional top K / K |
| Recall@q | Fraud count in unconditional top K / all fraud in that partition |
| Lift@q | Precision@q / partition fraud prevalence |
| Policy precision/recall | Same counts for the actual cutoff-qualified capped queue, which can contain fewer than K rows |
| Brier / log loss | Probability diagnostics; evaluate raw and selected mapping |
| Review rate | Selected rows / valid input rows |
| Observed fraud amount captured | Sum of `TransactionAmt` for labelled fraud selected, with dataset units stated |

For q in {0.005, 0.01, 0.02, 0.05}, K = ceil(q x N), capped at N. Sort descending score then ascending ID. Ties do not enlarge the queue. Report both class counts and denominators.

Zero selected rows: policy precision is unavailable, selected count is 0 and recall is 0 if positive labels exist. Zero fraud labels: recall and lift are unavailable, not 0. Rank-metric helpers must test these cases explicitly.

## 5. Champion-selection rule

1. Find the development-CV configuration with highest mean AP.
2. Consider configurations within 0.005 absolute mean AP of that best result.
3. Among them prefer highest mean Recall@1%. If recall differs by at most 0.02 absolute, prefer LR over LightGBM, then unweighted over weighted, then lower complexity.
4. Report fold means, standard deviations and per-fold durations. Do not claim fold independence.
5. A more complex model must justify its selection using this rule, not a desired resume headline.

Final-test improvement targets do not influence champion selection. Final reports include the frozen selected candidate and the untuned LR reference, each fitted under the permitted protocol.

The fixed LR reference is **E02, unweighted, C=1.0**. T06 fits its final reference classifier on all development rows and preserves its own run ID/bundle. Do not replace this reference with a retrospectively weaker configuration. If it is also the selected champion, state that the two are the same model and report no improvement claim against itself.

## 6. Calibration decision

- Use a supported sigmoid calibrator for a frozen fitted classifier; verify the installed API before implementation.
- Fit it on the calibration partition only. Do not use shuffled cross-validation over the entire dataset.
- On policy rows, retain sigmoid mapping only if Brier improves by at least 0.001 absolute, AP decreases by no more than 0.005 and Recall@1% decreases by no more than 0.02 absolute. Otherwise retain raw probabilities.
- Save raw-versus-calibrated Brier/log-loss/reliability evidence. Weighting can improve ranking while distorting probability estimates; calibration is evaluated, not assumed.

Apply this mapping protocol independently to the selected candidate and the fixed LR reference. Reuse the same bundle when they are identical. Preserve both reference and selected versions for T13.

## 7. Freeze review policy

Default q = 0.01 is a **product assumption**, not a tuned optimum. For policy partition size N, sort the selected scores and set tau to the score of the K-th ranked row, K = ceil(q x N).

- Single-request recommendation: p >= tau.
- Batch selection: keep p >= tau, rank deterministically and take at most ceil(requested q x batch N) rows.
- Ties at tau may create more candidates than capacity; the ID tie-break resolves selection.
- A shifted batch may underfill the queue because tau remains fixed. Report this effect.
- At other review fractions, unconditional rank metrics describe capacity-only ranking, while policy metrics apply the same frozen tau and the changed cap.
- Do not recompute tau from final-test scores or labels. Do not call the policy cost-optimal.

Freeze the LR reference's own cutoff from its own selected policy scores using the identical q and tie rule. Final policy comparisons must use each model's frozen cutoff, rather than transferring the champion's numeric cutoff to a differently scaled baseline.

## 8. Cost-sensitive evaluation

Simulation assumptions: missed labelled fraud = **100 cost units**, each reviewed row = **1 unit**, each reviewed legitimate row = **2 additional friction units**.

`simulated_cost = FN * missed_fraud_cost + selected_count * review_cost + FP * legitimate_friction_cost`

- Sweep missed-fraud cost {20, 100, 500}, review cost {0.5, 1, 2} and friction {0, 2, 5}; report sensitivity at each budget.
- Compare selected policy with review-none, amount-ranking and a capacity-matched random-selection expectation.
- Random expectation: expected selected fraud = K x prevalence. It does not require a lucky random draw.
- The simulation assumes all selected labelled fraud is identified in review and treated as avoided cost. This assumption is unvalidated.
- Report cost differences as simulated units. Observed fraud amount captured is descriptive exposure, not recovered money or prevented loss.

## 9. Uncertainty, segments and failure analysis

- Report paired 95% intervals for AP and Recall@1% differences using 1,000 elapsed-day block bootstrap replicates (`floor(TransactionDT / 86400)`). They are relative-time bins, not real calendar dates.
- Recompute rank metrics inside each replicate; preserve the frozen cutoff for policy metrics. Report replicates with unavailable one-class metrics and the number of valid replicates.
- If too few blocks or positives support intervals, disclose the limitation; do not replace dependence-aware intervals with unexplained row bootstrap certainty.
- Report results by elapsed-time bin, ProductCD and amount bands. Freeze amount bands using development quantiles.
- Require at least 200 rows and 20 positives for segment recall comparison; otherwise show counts and insufficient support.
- No demographic fairness conclusion is possible from absent protected-attribute labels.

## 10. Explainability

- Global: permutation importance on policy rows using AP, 5 repeats, seeded.
- Local: offline SHAP or an exact supported linear decomposition for at most 200 selected rows.
- Group transformed columns back to raw predictor names and verify additive sums in the model's raw output space within 1e-4.
- Explain positive/negative contributions to raw model output. A sigmoid calibrator changes the probability scale; do not describe raw SHAP values as additive calibrated-probability changes.
- Name opaque codes literally. Say a category influenced the model; do not invent card location, criminal intent or causal claims.

## 11. Final evidence and usefulness

Produce `reports/final/final_evaluation.json`, `reports/final/model_comparison.csv`, ranking/policy tables, plots and a freeze manifest. AP above final-test prevalence and Lift@1% > 1 are minimum usefulness checks. Stretch improvements and all misses remain visible in [the model card](MODEL_CARD.md).

Methods are project decisions informed by [primary sources](10_SOURCES.md), not a competition-approved evaluation protocol.
