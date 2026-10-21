# Model evaluation

The executed reference run is `artifacts/v2`; small reviewable evidence is published in
[reports/reference](../reports/reference/results.json). It uses the real Kaggle raw
files, not synthetic fixtures. Seed 42 locks **699 holdout components / 65,324 offers**;
**2,792 components / 234,931 offers** remain for development. A 20% group split yields
21.76% of rows because whole unequal-sized groups remain together.

There is no valid observation timestamp for chronological validation. Travel dates
do not supply booking chronology. The estimand is performance on unseen related
itineraries within the historical sampling frame. See [group definitions](feature_engineering.md).

Three-fold GroupKFold tunes the small prespecified grids. Every imputer, scaler and
encoder fits inside each development fold. Hyperparameters and model family are
selected by mean CV MAE in original INR. Selection is written to selection.json before
any holdout predictions. All seven prespecified finalists are evaluated after this
freeze; none is selected or retuned using these test values.

| Model | CV MAE (INR) | Holdout MAE (INR) | RMSE (INR) | R-squared | Median AE (INR) | MAPE |
|---|---:|---:|---:|---:|---:|---:|
| Global median | 16,044.04 | 16,503.92 | 26,764.01 | -0.3733 | 3,924.00 | 69.52% |
| Route/class median | 4,655.96 | 5,164.26 | 8,520.59 | 0.8608 | 2,416.50 | 34.64% |
| Ridge | 4,852.93 | 4,837.10 | 7,028.48 | 0.9053 | 3,312.07 | 46.01% |
| Decision Tree | 3,857.40 | 3,880.81 | 5,896.59 | 0.9333 | 2,428.23 | 31.76% |
| Log-target tree | 3,734.91 | 3,649.22 | 5,726.48 | 0.9371 | 2,157.72 | 27.35% |
| Histogram boosting | 3,574.29 | 3,582.12 | 5,332.92 | 0.9455 | 2,403.15 | 31.49% |
| **Log-target histogram boosting** | **3,518.65** | **3,425.24** | **5,295.11** | **0.9462** | **2,094.82** | **26.38%** |

Exact metrics, parameters and fold errors are in [model_comparison.csv](../reports/reference/model_comparison.csv)
and [results.json](../reports/reference/results.json). MAE is the primary budget-error
metric; RMSE penalizes large misses. R-squared depends on target variance and can look strong
when cabin classes dominate the scale. Median AE can conceal expensive tails.
MAPE weights cheap offers heavily and is undefined for zero fares (these are rejected);
WAPE weights by fare mass and can obscure low-fare failures. Selected-model WAPE is
**16.03%**. All point predictions are inverse-transformed before currency metrics.
Log-model exponentiation does not generally estimate an arithmetic conditional mean.

A paired bootstrap resamples **699 whole holdout components**, 500 repetitions, seed 42.
The selected-model MAE 95% percentile interval is **INR 3,207.52-3,619.79**. The paired
MAE difference versus route/class median interval is **INR -2,065.01 to -1,415.15**.
This is conditional on this trained model/split and defined cluster independence;
it excludes training/selection uncertainty and is not an individual prediction interval.

## Segment and failure analysis

[All segments](../reports/reference/segment_errors.csv) include counts and multiple
metrics. No lead-time buckets are reported. Price buckets use training quartiles
(INR 4,755, 7,420, 42,457); these are hindsight diagnostics, not deployable inputs.

| Segment | Holdout n | MAE (INR) | R-squared | MAPE | Interpretation |
|---|---:|---:|---:|---:|---|
| Economy | 44,410 | 2,276.45 | 0.1915 | 33.45% | Large relative errors remain despite strong aggregate R-squared. |
| Business | 20,914 | 5,864.65 | 0.6125 | 11.38% | Lower percentage error but material budget exposure. |
| AirAsia | 3,067 | 1,772.37 | 0.0331 | 32.85% | Low within-carrier R-squared; high aggregate performance is not universal. |
| Vistara | 31,411 | 4,315.88 | 0.9374 | 23.20% | Scale/mix differs; expensive misses remain. |
| StarAir | 1 | 2,505.87 | undefined | 26.15% | One offer cannot establish reliability. |
| Trujet | 0 | unavailable | unavailable | unavailable | No holdout evidence for this carrier. |

The [observed-versus-estimated and residual plots](../reports/reference/figures/holdout_errors.png)
show separate cabin bands, dispersion that changes with price, and severe underestimation
of some expensive offers. The largest miss is a Vistara Bangalore-to-Delhi business
offer at INR **111,883**, estimated at **54,044.48** (absolute error **57,838.52**).
See [25 largest errors](../reports/reference/largest_errors.csv). Missing timing,
inventory and fare inclusions are plausible explanations, not verified causes.

Mean signed error is **INR -635.50**; underprediction occurs for **43.24%** of offers.
Thus fewer but larger negative errors outweigh many small positive errors. A single
symmetric contingency or global confidence interval would conceal this asymmetry.

This published holdout is now consumed. Future experimentation needs a new
prospectively locked dataset/holdout; do not tune against these disclosed errors.
