# Delhivery Last-Mile Delay Intelligence Pipeline

**Program:** Artificial Intelligence and Machine Learning Certification Program · E&ICT Academy, IIT Roorkee  
**Project:** Reproducible synthetic-data pipeline for delivery-time regression, delay alerts, hub segmentation, and batch actions.

All shipment records in this repository are synthetic. The Delhivery name is used only as the scenario context; no real shipment records are included.

## How to run

Run these commands in order from a fresh clone, at the repository root. Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` prints `24 1224 48` and creates the three source CSVs in `data/`. The remaining scripts build the modelling table, reports, hub segments, model files, and new-shipment action list. The shared `clean_shipments()` and `build_features()` functions live in `src/shipment_prep.py`; prediction imports those same functions rather than duplicating preprocessing.

`package_model.py` regenerates `models/delay_model.pkl`, `models/delay_threshold.csv`, `models/feature_order.csv`, and `models/impute_values.csv`. The committed pickle can be loaded reliably only with the scikit-learn version pinned in `requirements.txt`; if the installed versions differ, rerun `python package_model.py` rather than relying on the committed pickle. Scripts write CSVs without row indices; output metrics are rounded to four decimal places where appropriate. Random seeds and feature order are pinned for reproducibility.

## Data findings

- **Service promise risk:** Express has the highest delay rate at **48.97%** (119 delayed out of 243); Standard is 20.56% and Economy is 8.57%. See `outputs/part_a_delay_by_group.csv`.
- **Hub tiers:** observed delay rate rises from **4.00% for Tier 1** to **21.50% for Tier 2** and **44.00% for Tier 3**. Tier is useful as a broad operational indicator, but it is not a substitute for measured hub behaviour.
- **Distance matters:** delay rate rises from **2.19%** in the 0–200 km band to **13.68%** at 201–400 km, **43.64%** at 401–600 km, and **85.23%** above 600 km. The distance bands use the specified right-inclusive bins.
- **Strongest delivery-hours correlations:** the three largest absolute correlations excluding `delivery_hours` itself are `promised_hours` (**0.8372**), `distance_km` (**0.4013**), and `n_stops` (**0.3247**). These are associations, not causal estimates.
- **Declared value transformation:** the median-filled `declared_value_inr` has mean **INR 1,533.0542** and median **INR 1,121.5000**, while its natural log has mean **7.0277** and median **7.0224**. Logging compresses the right-skewed monetary scale, bringing the mean and median much closer together.
- **Data quality:** the raw export has 1,224 rows; exact duplicate removal leaves 1,200 unique shipments. Five implausible weight sentinels are treated as missing before median imputation. The modelling features and target contain zero missing cells after preparation.

## Model results (regression)

- **Baseline:** the validation mean-prediction baseline has R² **-0.0212**. It performs slightly worse than predicting the validation mean, showing that a constant prediction does not explain parcel-to-parcel variation.
- **Validation winner:** `linear_regression` has RMSE **4.8912 hours** and R² **0.9313** on validation. Gradient boosting is close (RMSE **4.9165 hours**, R² **0.9306**), while random forest reaches R² **0.9128** and the selected decision tree reaches R² **0.8867**.
- **Tree depth and bias/variance:** depth 1 is high-bias: both training and validation errors are high, and its validation RMSE is materially worse than the best depth. The unlimited tree has near-zero training error but a large training-to-validation gap, indicating high variance. Depth 6 is selected by the lowest validation RMSE; see `outputs/part_b_tree_depth.csv` for all eight depths and the actual gap values.
- **Learning rate:** 0.1 is the best of the tested gradient-boosting learning rates. At 0.01, 100 boosting stages learn too slowly; at 1.0, updates are too aggressive and overshoot the useful solution. The validation R² values are in `outputs/part_b_gb_learning_rate.csv`.
- **Cross-validation and shipping decision:** the two lowest-validation-RMSE candidates are evaluated using the same shuffled five-fold splitter on the 1,020 training-plus-validation rows. The validation winner is linear regression and is the final test-scored model; the test R² is **0.9493**. Cross-validation ranks gradient boosting above linear regression by mean R² (**0.9353** versus **0.9261**), so the rankings disagree. I retain linear regression for this reproducible exercise because it won the pre-designated validation RMSE comparison and then scored strongly on the untouched test set; before production, I would compare time-based backtests and retraining stability before choosing between these close candidates. See `outputs/part_b_cv.csv`.

## Model results (classification)

- **Blind baseline:** predicting every parcel as on-time gives accuracy **0.7667**, but recall and F1 are **0.0000** because it identifies none of the 42 delayed validation parcels. Accuracy alone is misleading with an imbalanced target.
- **Feature signal and noise:** the five bookkeeping columns have a combined random-forest importance share of approximately **0.2245** (sum the `rf_importance` values for `is_noise=True` in `outputs/part_c_feature_scores.csv`). All five rank outside the top seven by F-score. The top seven are `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag`, and `sale_event_flag`; the smallest F-score among these is above 30, while each noise feature is below 10. The feature selector uses training data only.
- **Feature-count experiment:** see `outputs/part_c_selectk.csv` for validation F1 at k = 5, 7, 10, 12, and 17. The project follows the required k = 7 for tuning and packaging. This reduces inputs while retaining the seven strongest univariate signals, but F-score is a screening statistic rather than proof of causal importance.
- **Tuning:** grid search, randomized search, and the untuned forest on the same seven features are compared in `outputs/part_c_tuning.csv`. The validation F1 values are close; in this run the untuned seven-feature forest is best. With only 180 validation parcels, a small F1 difference can be sampling noise, so the validation result should not be treated as a universal ranking.
- **Threshold trade-off:** the chosen threshold is **0.5**, the highest tested threshold whose validation recall is at least 0.80. At that threshold, validation precision is **0.7955** and recall is **0.8333**: lower thresholds catch more delayed parcels but generate more unnecessary calls, while higher thresholds miss more delays.
- **Final test:** test precision is **0.6889**, recall **0.7561**, and F1 **0.7209**. Recall is lower than the validation recall; that is an honest out-of-sample difference, and the test set contains only 41 delayed parcels, so a few additional misses materially change the estimate. The test set was not used to tune the model or threshold.

## Operations recommendation

- **Escalate playbook:** the `Escalate` segment contains 9 hubs, with a mean hub-level delay rate of **46.00%**, mean delivery time of **64.6504 hours**, mean distance **470.4757 km**, and mean route stops **84.3444**. Prioritize hub-manager review, route capacity checks, and exception handling for these hubs.
- **Tier mismatches:** `outputs/tier_vs_segment_mismatch.csv` identifies four cases where measured behaviour differs from the nominal tier: **HUB07 Kolkata** (Tier 1, Watch), **HUB13 Nagpur** (Tier 2, Escalate), **HUB15 Kochi** (Tier 2, Escalate), and **HUB22 Dehradun** (Tier 3, Watch). Keep tier as a planning label, but use observed segment behaviour to choose operational playbooks and investigate mismatches instead of assuming the tier is ground truth.
- **Alert operating point:** the threshold is **0.5**. On the held-out test set, precision **0.6889** means about **6.9 of every 10 flagged parcels** were actually delayed; recall **0.7561** means about **7.6 of every 10 delayed parcels** were caught.
- **New batch:** `outputs/new_shipment_actions.csv` contains 48 parcels. **8 were flagged**, and **6 of those 8** were assigned `Escalate to hub manager`; the other flagged parcels receive a proactive customer call. Unflagged parcels receive `No action`. The action list is deterministic and is regenerated by `python predict_delay.py`.
- **Why retain K = 3:** the across-to-within separation ratio rises from **2.894** at K = 2 to **5.3928** at K = 3, supporting three meaningfully separated operating groups. Ratios rise further for K = 4 (**6.3047**) and K = 5 (**6.8866**), but those solutions include clusters smaller than six hubs (sizes `8-7-5-4` and `8-7-4-3-2`), which are too small to justify their own repeatable playbook. Three segments balance separation with operational usefulness.

## Production plan

### Readiness mapped to this repository

- **Accuracy:** regression MAE/RMSE/R², classifier confusion matrices, and the final held-out test report are persisted as CSVs.
- **Reliability:** the saved threshold and explicit action rules make the alert behaviour inspectable; unknown or missing source values are handled by shared median imputation and an `Unknown` payment placeholder.
- **Reproducibility:** pinned package versions, fixed random seeds, committed generator, exact feature order, and a scripted run sequence reproduce the reports.
- **Performance:** the forest and scoring path are batch-oriented and score 48 rows in one call; a production deployment should measure p95 scoring latency and throughput at expected daily volume.
- **Maintainability:** preprocessing is centralized in `src/shipment_prep.py`, while regression, classification, segmentation, packaging and scoring have separate scripts and outputs.
- **Consistency:** both training and inference use the same `clean_shipments()` and `build_features()` functions and the stored imputation medians and feature order.
- **Monitorability:** output CSVs provide auditable probabilities, flags, actions, evaluation metrics, hub segments and a pickle reload check. A production system should add timestamped logs, model/version IDs and alert outcomes.

### Monitoring triggers

Initial triggers below are proposed operational guardrails and should be recalibrated with the regional operations team after a baseline period:

1. **Weekly delivered-parcel delay share:** investigate if it exceeds **30%** for two consecutive weeks, excluding incomplete/in-flight parcels.
2. **Weekly flagged-parcel share:** review capacity and threshold if more than **25%** of dispatched parcels are flagged in a week or if the share changes by more than **10 percentage points** from the trailing eight-week baseline.
3. **Recall on matured parcels:** alert the model owner if rolling four-week recall falls below **0.75** once at least 100 parcel outcomes are known; report precision alongside recall to quantify unnecessary calls.
4. **Feature drift:** investigate if the population stability index for `distance_km` or `n_stops` exceeds **0.20**, or if missing-value rates exceed **5%** for any key input.

### Retraining and sale events

Retrain and validate quarterly as a starting schedule, and sooner after material route/network changes or sustained performance-trigger breaches. Keep a time-based holdout in addition to the reproducible random split before production release. For the 15–28 September sale window (`sale_event_flag=1`), compare outcomes with prior sale periods or similar promotional events, not ordinary weeks; sale traffic has a different service mix and can interact with Express promises. Data drift could appear as a shift in lane distance, stop counts, service mix, rain exposure, sale-event share, hub delay rates, or the distribution/calibration of predicted probabilities.

## Limitations

1. **Synthetic data:** the target is generated from a known formula with artificial noise. Real shipment operations include unobserved events, changing capacity, customer availability, address quality and operational interventions; results will not automatically transfer to production.
2. **Small test positive class:** only 41 delayed parcels are in the classification test set, so precision and recall have appreciable sampling uncertainty.
3. **Single random split and threshold:** threshold selection is based on one 180-row validation split. A time-based validation and cost-sensitive evaluation are needed before deployment.
4. **Feature availability:** the model assumes dispatch-time inputs such as distance, stops, rain, sale status and hub capacity are available and accurate at scoring time. Operational data latency and data quality may reduce performance.
5. **Cluster interpretation:** K-Means segments summarize hub-level averages and are not causal explanations. Segment assignments and playbooks should be reviewed periodically as network conditions change.
6. **No live integration:** the project creates a batch CSV action list; it does not call customer systems, notify hub managers, or include human feedback loops.

## Repository outputs

The `outputs/` directory contains the checks, delay evidence, correlations, regression diagnostics and metrics, classifier scores and thresholds, hub profiles and segments, model reload validation, and final new-shipment actions. The `models/` directory contains the reusable classifier, chosen threshold, feature ordering, and imputation values.
