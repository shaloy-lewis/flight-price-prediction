# Success criteria

Criteria were defined before inspecting the revised holdout predictions.

| Dimension | Evidence and acceptance rule |
|---|---|
| Correctness | Same canonical inputs produce the same prediction independent of today's date; parsers reject invalid formats; input frames remain unchanged. |
| Leakage control | No related-flight component or identical canonical feature vector crosses holdout or CV boundaries; all learned transformations fit inside training folds. |
| Model selection | Minimum mean grouped-CV MAE in INR across the prespecified candidate set; fixed configuration order breaks exact ties. Record runtime, fold variability and explanation limitations when assessing the complexity trade-off. |
| Historical value | Compare locked holdout MAE and RMSE with global and route/class medians. A paired group-bootstrap MAE difference interval below zero is evidence of improvement for this historical sampling frame, not a business ROI estimate. |
| Reliability | Report airline/cabin/route/stops/price-segment sizes and errors, large failures, signed error and underprediction rate. Do not hide weak or absent segments. |
| Reproducibility | Source-file hashes, split assignments, config, source snapshot, environment, versioned manifest, tests and documented commands. |
| Serving | Typed requests; useful 422 errors; 503 for missing artifacts; sanitized 500 failures; INR responses; explicit historical scope. |
| Deployment | Local functionality can be verified, but current-market usefulness and service latency/availability targets require stakeholder agreement and prospective data. |

No arbitrary MAE percentage, latency SLO, savings target or readiness score is invented.
A positive overall comparison is insufficient if high-cost segments remain unreliable.
Bootstrap intervals describe estimation uncertainty for average error conditional on a
fixed trained model; they are not uncertainty bands for individual offers.
