# Monitoring and lifecycle

This repository has no production observations, measured drift, operational service
level or automated retraining. Implemented utilities build a training-only reference,
warn about unsupported inputs and create an offline descriptive batch report.

| Signal | Implementation or collection plan | Decision |
|---|---|---|
| Schema violations | FastAPI 422 responses; middleware status/path logs | Inspect malformed clients; do not retrain to accommodate invalid schemas. |
| Latency and inference errors | Request completion logs in milliseconds; sanitized failures | Establish a pilot baseline and agreed alert limits; roll back after persistent error/latency regression. |
| Input support | Known training categories, route and airline/cabin membership; numeric min/max | Warn; obtain a quote and review unsupported segments. Marginal support is not proof of joint support. |
| Input drift | Offline categorical total variation, unseen fraction, numeric median shift and outside-range fraction | Compare sufficiently sized batches and investigate source/season/mix changes before action. No universal drift cutoff is asserted. |
| Prediction distribution | Retain aggregate quantiles/histograms by model version in a future pilot | Compare equal segments/time windows; shifts can reflect legitimate request mix. |
| Performance with labels | Join later verified observed fares using stable offer/configuration IDs and timestamps | Track MAE, RMSE, signed error, underprediction and segment support against the route/class baseline. |
| Versioning | Dataset hashes, source snapshot, model and schema versions, locked splits/dependencies | Every release gets a new run directory; never overwrite evidence. |

Data drift is a change in the distribution of inputs. Concept drift is a change in
the relationship between inputs and target. Performance deterioration is observed
worsening of error when valid outcome labels become available. Input drift alone
establishes neither concept drift nor poor model performance.

Run `python -m src.cli drift --data-dir data/raw --model-dir artifacts/v2 --output reports/local/drift`
for an illustrative offline comparison. Comparing this source against its training
reference partly reflects the held-out mix; it is not operational drift evidence.
Unseen categories and numeric range checks have deterministic tests.

Retrain only after source validity and business scope are reviewed and a new temporal
or grouped evaluation is specified. Keep a fresh, prospectively locked holdout; repeated
tuning against this published test set invalidates its role. Release criteria should
include baseline-relative performance, important-segment reliability, schema
compatibility and operational checks agreed with the stakeholder. Revert MODEL_DIR
to the previous compatible trusted run and restart if these regress.

Do not log raw user payloads, credentials or itineraries by default. Pilot telemetry
needs retention/access policies and stable request-to-outcome identifiers before
performance monitoring can be implemented.
