# Implementation summary

## Original issues by priority

P0: unsupported and inconsistent reference-date lead time; random row split of related
offers; mismatched encoded SHAP labels/path/units; weak parsing, mutation and request
validation; broken UI date/duration handling. P1: no reproducible training-only model
selection, realistic baseline, segment failures or decision framing. P2: per-request
model reloads, internal errors exposed, unversioned artifacts and no active CI.
[Original file/line evidence](project_audit.md).

## Files changed and why

| Paths | Action and purpose |
|---|---|
| src/utils/constants.py, src/utils/utils.py | Replace historical/current-date feature fabrication with versioned features and pure anchored date/time/duration/stops parsers. |
| src/schema.py, src/config.py | New canonical/legacy schemas and explicit validated experiment configuration. |
| src/components/data_ingestion.py | Original-file hashes, schema/quality/exclusion reports, connected-component leakage groups and deterministic holdout. |
| src/components/data_transformation.py | One fold-local imputation/scaling/encoding constructor. |
| src/components/model_trainer.py, model_evaluation.py | Seven prespecified candidates including realistic medians, INR metrics, segment errors and cluster-bootstrap uncertainty. |
| src/pipeline/training_pipeline.py | Import-safe grouped CV, frozen selection, final evaluation, exact evidence and versioned artifact creation. |
| src/artifacts.py, src/explainability.py, src/monitoring.py, src/reporting.py, src/analysis.py, src/cli.py | New artifact checks, aligned output-scale explanations, support/drift utilities, headless charts, reproducible EDA and CLI workflows. |
| src/pipeline/prediction_pipeline.py | Validated inference through the fitted end-to-end estimator; legacy Python adapter retained. |
| api.py, app.py | Cached model initialization, typed contracts, useful errors, historical scope, consistent airline/details and UI validation. |
| src/logger/logging.py, src/exception/exception.py | Retain compatibility names while removing import-time filesystem writes and path-leaking exception formatting. |
| scripts/download_data.py, scripts/publish_report.py, configs/experiment.json | Pinned original source retrieval, small evidence publication and prespecified configuration. |
| pyproject.toml, setup.py, requirements.txt, requirements_dev.txt, requirements.lock.txt, MANIFEST.in | Real packaging, optional dependencies, tested transitive constraints, lint/test/static/build configuration and reproducible source distribution inputs. |
| .github/workflows/ci.yml | Install/check/test/build/container-build workflow, independent of Kaggle access. Hosted execution remains unverified. |
| Dockerfile, docker-compose.yml, .dockerignore, init_setup.sh | Pinned Python, non-root service, read-only model mount, readiness and no embedded legacy models. |
| .gitignore | Exclude tools/data/generated run outputs; explicitly retain curated small CSV reports. |
| tests/conftest.py, tests/unit/test_features.py, test_methodology.py, tests/integration/test_serving.py, test_training.py, test_streamlit.py | Small deterministic synthetic fixtures and 71 actual invariant/contract/integration checks. |
| experiment/experiments.ipynb | Retain historical exploratory source, clear stale outputs and withdraw its unsupported deployment conclusion. |
| README.md, docs/*.md | Replace unsupported positioning with actual methodology/results, business scenarios, architecture, migration, review, interview preparation and reproduction. |
| reports/reference/* | New small real-data quality, provenance, EDA, exact comparison, segment/failure, explanation and API evidence. No full CSV dataset or model binary is added. |

No original source dataset was supplied or fabricated. Original unversioned artifacts
remain untouched for historical context; v2 neither loads nor embeds them. Empty old
setup.cfg/tox.ini are inert; active configuration is pyproject.toml. Template scaffold
is retained. No file was intentionally deleted.

## Methodology and actual outcomes

Raw data without observed booking timestamps supports historical estimation only.
Group-aware holdout/CV, fold-local preprocessing, seeded compact comparisons and frozen
selection replace random-row/single-tree evaluation. Hist_log is selected by minimum
development MAE, with runtime/complexity caveats documented.

Held-out MAE/RMSE are **INR 3,425.24 / 5,295.11**, versus route/class baseline
**5,164.26 / 8,520.59**. Mean absolute historical error falls **33.67%**; no savings
is measured. Economy and high-cost failures limit use.
[Exact methodology/results](model_evaluation.md) | [Selection](model_selection.md)

Validation: **71 tests**, lint/format, selected static checks, dependency check,
sdist/wheel build, installed-wheel import, source retrieval hashes, real-data EDA/training/
evaluation, API HTTP and Streamlit checks pass. Docker runtime/build remains blocked
by the stopped daemon; hosted CI and deployment are unverified.
[Commands and evidence](validation_results.md)

[Remaining roadmap](roadmap.md) | [Independent-role self-review](final_review.md) |
[Ten grounded interview questions](interview_preparation.md)
