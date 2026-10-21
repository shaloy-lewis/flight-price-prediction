# Technical audit

Audit baseline: commit `34004c0`, inspected 21 October 2026. The working tree was clean;
implementation uses branch `improve/rigorous-fare-benchmark`. References below are to the
original commit, so they remain stable after refactoring. All source files, entry points,
the exploratory notebook (including stored outputs), packaging, Docker, requirements,
tests and the three existing artifact files were inspected. Pickles were not executed
during inspection; their provenance is unverified. No real CSVs were initially present.

| Priority | Evidence at baseline | Finding and consequence |
|---|---|---|
| P0 | `src/utils/utils.py:70-79`; `src/pipeline/prediction_pipeline.py:33` | Training subtracts a fixed 10 February 2022 reference; serving subtracts today. Neither is an observed booking timestamp. Features drift with the clock and imply unsupported forecasting. |
| P0 | `api.py:60-61`; `src/pipeline/prediction_pipeline.py:56-60` | SHAP uses a different preprocessing path from prediction, then zips encoded contributions with raw columns. Labels and dimensions are wrong; contributions are log-price, not INR. |
| P0 | `src/components/data_ingestion.py:35-46` | Exact full-row deduplication followed by random row splitting does not protect related itineraries or identical model inputs with different fares. |
| P0 | `src/utils/utils.py:52-88`; `api.py:19-29` | Parsers accept partial duration strings, silently produce missing stops, mutate frames, and log invalid fares. Request fields are unrestricted strings with inconsistent airline/code defaults. |
| P0 | `app.py:28,47-48,55` | A selected date shadows the imported date class; two-plus-stop spelling disagrees with parsing. Duration via modulo-day arithmetic loses multi-day itineraries. |
| P1 | `src/components/model_trainer.py:24-47`; `src/components/model_evaluation.py:13-42` | One unseeded tree, no development CV, no recorded comparisons/MAE/segment analysis/uncertainty; results only in logs. |
| P1 | `experiment/experiments.ipynb` (cells defining `models`, `result_df`, final markdown) | Many models are compared on the same test set; notebook recommends Random Forest while deployed trainer uses a tree. Stored outputs are historical and are not reproducible evidence for revised code. |
| P1 | `README.md:6-8`; `app.py:18-24` | Broad booking savings and timing claims exceed evidence from a historical snapshot. Provenance, license, schema, limitations and a decision framework are absent. |
| P1 | `src/pipeline/training_pipeline.py:6-16`; `tests/` | Importing training executes disk I/O and training. Test directories contain only empty initializers. |
| P2 | `api.py:51,71-72`; `app.py:7,76` | Models reload per request/rerun; all server failures become client errors exposing internal exception paths. |
| P2 | `artifacts/`; `Dockerfile:22-24` | Separate legacy pickles have no manifest, hash, input contract, version compatibility or linked evaluation. Docker embeds these artifacts. |
| P2 | empty `pyproject.toml`, `setup.cfg`, `tox.ini`, `requirements_dev.txt`; `.github/workflows/.gitkeep` | No active lint/test/build configuration or CI. Requirements freeze many transitive and unused dependencies. |

The original duration caps are fit on the training split (`data_transformation.py:82-88`),
and the original encoder is fit on training only (`:102-103`); these are positive controls,
not evidence of test leakage. The reference-date feature is an information-validity
problem independently of ordinary train/test leakage.

## Execution plan and phase gates

1. **P0:** define historical fare benchmarking; strict shared canonical schema; eliminate
   invented lead time; deterministic duplicate-aware groups; fold-local preprocessing;
   one end-to-end model artifact; transformation, split and explanation tests.
2. **P1:** obtain the original dataset if publicly accessible; record source and hashes;
   implement reproducible quality/EDA, median and route/class baselines, Ridge, tree and
   log-target comparison, grouped CV, a locked final holdout, INR evaluation, segment
   errors, cluster bootstrap intervals and permutation importance. Publish actual results
   only after execution. If retrieval fails, use explicitly synthetic fixtures for tests.
3. **P2:** cached serving, typed API/errors, historical-scope UI, manifest validation,
   offline monitoring, package/build/CI/Docker configuration, README and review.
4. At every gate run relevant checks and record commands, results and environmental
   limitations in `docs/validation_results.md`. Finish with a hiring-panel review and
   grounded interview answers. No deployment or measured savings is implied.
