# Model selection and operational trade-offs

Selected: **HistGradientBoostingRegressor on log fare**, wrapped in
TransformedTargetRegressor and the shared fitted preprocessing pipeline.

The prespecified rule chooses minimum three-fold grouped-CV MAE in INR. Its CV MAE
is **3,518.65**, compared with **3,574.29** for raw-target histogram boosting,
**3,734.91** for the log-target tree, and **4,655.96** for the realistic route/class
median. Fold MAEs are **3,610.38 / 3,530.62 / 3,414.94**; their SD is **80.24 INR**.
The small raw/log boosting difference is not evidence of universal superiority.

Search configuration: Ridge alpha [1, 10]; trees depth [12, 20], min_samples_leaf 30;
boosting max_leaf_nodes [15, 31], 120 iterations, learning_rate 0.08,
l2_regularization 1, early_stopping disabled, fixed seeds. Both raw/log boosting
choose 31 leaves; both tree variants choose depth 12; Ridge chooses alpha 10.
Early stopping is disabled to avoid an additional ungrouped internal validation split.

The selected search/refit took **107.38 seconds** on this local environment, versus
**18.53 seconds** for the log-target tree and **0.99 seconds** for route/class median.
These are measured development runtimes, not serving SLAs or portable benchmarks.
Single-thread numerical work bounds resource use. Histogram boosting adds complexity
but has lower development error without adding dependencies beyond scikit-learn.
The tree and route/class median remain credible simpler alternatives for a constrained
budget or stronger interpretability requirement.

Selection was frozen before holdout evaluation. The chosen model then achieved
**3,425.24 INR holdout MAE**, a **33.67% reduction in average absolute historical error**
relative to route/class median. This is not a savings percentage. Strong aggregate
R-squared does not resolve weak within-economy performance or expensive underpredictions.

Log-space modeling makes positive point estimates natural, but can emphasize
proportional structure and introduces retransformation bias relative to a mean-fare
target. No smearing correction was fit on holdout data. The actual selection target
is observed INR MAE, not normality of residuals or best training R-squared.

Tree SHAP is available on the exact encoded representation in log(INR). That
explanation is less direct than a route median lookup or linear contribution.
See [explainability](model_explainability.md). Selection for this historical portfolio
run is not approval for current-market use.
