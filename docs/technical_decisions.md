# Technical decisions

| Decision | Alternative | Reason and trade-off |
|---|---|---|
| Historical configuration benchmark | Live or future fare forecast | Raw per-offer booking/observation timestamps are absent. A more limited, defensible target is preferable to invented lead time. |
| Original economy/business CSVs | Mix in Clean_Dataset | Preserves raw times and provenance; mixing variants double-counts offers and imports undocumented filtering/derived timing. |
| Drop booking timing and travel date | Fixed historical date or today's date | Neither represents a measured booking event. Exclusion sacrifices predictive signal that cannot be faithfully reproduced. |
| Connected flight/input components | Random rows or only exact full-row deduplication | Prevents repeated itineraries and identical model inputs from leaking. Estimates generalization to new itinerary components, a stricter and different estimand from new offers for known flights. |
| Fixed 20% group holdout, three development folds | Chronological holdout | There is no genuine observation timestamp. Group holdout is reproducible but does not test future-market transportability. |
| Fold-local end-to-end pipelines | Three separately serialized preprocessing/model/cap files | Eliminates transform skew and fold leakage with fewer moving parts. |
| Median and route/class median, Ridge, tree, histogram boosting | Large model zoo or deep learning | Covers statistical, linear and nonlinear references with CPU-scale reproducibility. Hyperparameter grids are small and fixed. |
| Raw and log-target tree/boosting variants | Assume log-price is automatically superior | Evaluate original-INR MAE after inverse transformation. Exponentiated log regressions need not estimate the arithmetic conditional mean. |
| Minimum grouped-CV MAE | Lowest test RMSE / training R-squared | Direct budget error criterion defined before final evaluation; record runtime, variance and interpretability trade-offs. No claim of a universal best model. |
| Group bootstrap for average error | IID row bootstrap or a prediction interval | Respects defined cluster dependence. Conditional intervals exclude training and selection uncertainty and do not cover individual fares. |
| Permutation importance + aligned local explanations | Raw-column zip of encoded SHAP values | Different tools answer different questions; attribution units must match the fitted output scale. Correlation still limits interpretation. |
| Manifest + hash + dependency checks | Implicitly trust any pickle | Detects corruption and incompatibility before loading. A hash is not authentication; serialization still requires a trusted artifact. |
| One process-local cached model | Reload every request / model-serving service | Suitable for portfolio scale with predictable startup and modest architecture. A process restart is required to switch versions. |
| Offline drift descriptors | Invented production drift, feature store, Kubernetes | There is no live telemetry or scale need. Drift thresholds and retraining gates require an operational pilot. |
| Keep legacy artifacts quarantined | Silently serve old pickles | Retains historical work but v2 never loads them. Original request format is available through an explicitly deprecated adapter. |
