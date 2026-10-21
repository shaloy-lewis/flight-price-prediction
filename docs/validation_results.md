# Reproduction and validation record

Executed locally on **21 October 2026**, Windows, Python **3.12.10**, branch
`improve/rigorous-fare-benchmark`, baseline commit
`34004c097cbe0a9def299c499f16cae30a4aba22`. Initial working tree was clean.
Validation ran on an uncommitted working tree, as recorded by the training
provenance. Git history records subsequent commits and publication. No remote
deployment or hosted CI result is claimed by this validation record.

The host had no Python on PATH. A Python embeddable runtime and dependencies were
installed **inside ignored .tools/**, with repository path configured for imports.
The executable used here was `./.tools/python/python.exe`. Commands below use
`python` for a normal Python 3.12 environment; substitute the local executable in
this workspace. Package versions are recorded in requirements.lock.txt and
reports/reference/environment.txt. Models require matching numerical-library versions
and Python minor version.

## Executed phase gates

| Phase / command | Result |
|---|---|
| Initial Git/file/notebook/artifact/config inspection | Clean starting tree; dedicated branch created. Original audit references commit 34004c0. Legacy pickles were not deserialized. |
| P0 `python -m pytest -q` | First complete suite: **70 passed**; shared parsing, leakage controls, artifact/API/attribution and synthetic training checks. |
| Data retrieval via original Kaggle API | Real version-2 archive and public metadata downloaded; no credentials read. |
| `python scripts/download_data.py --output data/retrieval_check` | Passed; both raw-file SHA-256 hashes exactly match the experiment's original downloads. |
| Real data ingestion / validation | 300,261 input; 2 duplicates removed; 4 malformed rows excluded; 300,255 retained. Source hashes/quality reports persisted. |
| `python -m src.cli eda --data-dir data/raw --output reports/reference --config configs/experiment.json` | Passed; training-only summaries and five figures generated. All five plus holdout diagnostics visually inspected. |
| `python -m src.cli train --data-dir data/raw --output artifacts/v2 --config configs/experiment.json` | Passed on real data; all seven candidates, 3-fold grouped CV, frozen selection, final holdout, 500 cluster bootstraps, segments, permutation/SHAP and manifest saved. Selected hist_log. |
| `python -m src.cli evaluate --data-dir data/raw --model-dir artifacts/v2 --output reports/local/evaluation` | Passed; source/split hashes checked and selected metrics reproduced (MAE 3425.240250089688; RMSE 5295.105744635419; R-squared 0.9462449287173601). |
| `python -m src.cli drift --data-dir data/raw --model-dir artifacts/v2 --output reports/local/drift` | Passed; descriptive offline batch report, not production drift evidence. |
| `python scripts/publish_report.py` | Passed; curated small reports/figures copied without model binary or full holdout data. |
| Final `python -m pytest -q` | **71 passed, 2 third-party deprecation warnings**, 14.05 seconds in the final full run. Includes Streamlit successful prediction and invalid-route handling. |
| `python -m ruff check .` | Passed. |
| `python -m ruff format --check .` | Passed; 38 Python files already formatted. |
| `python -m mypy` | Passed, 7 selected contract/config/artifact/API source files. This does not claim full-repository strict typing. |
| `python -m pip check` | Passed; no broken installed requirements. |
| `python -m build --no-isolation` | Passed; sdist and wheel built. |
| Wheel installed with --no-deps into ignored .tools/package-smoke | Passed; src and api imported from installed wheel target, API version 2.0.0. This is not a clean-environment/Linux installation test. |
| Uvicorn on 127.0.0.1:8080 and real HTTP GET /health, POST /predict?explain=true | Passed: ready v2; historical prediction INR 3594.67; log(INR) attribution additivity residual 1.78e-15. Request/response saved in reports/reference/api_example.json. |
| `docker compose config --quiet` | Passed (exit 0); host emitted Docker config access warnings. |
| `docker build -t flight-fare-benchmark:local .` | **Blocked**: Docker engine named pipe unavailable / daemon stopped; build and container runtime were not verified. |
| Local Markdown link audit / git diff --check | Passed; no broken local documentation links or whitespace errors. |
| Hosted GitHub Actions / original public Streamlit demo | Not executed or verified. Workflow is configured; README does not advertise an unverified demo. |

Warnings: Starlette accesses a deprecated AnyIO alias; SHAP imports a deprecated NumPy
dtype conversion. They do not fail the suite and are not hidden. A Windows physical-core
detection warning appeared in an earlier run; LOKY_MAX_CPU_COUNT=2 avoids that host
probe. No claim of dependency vulnerability clearance is made.

The training source snapshot captures the code at experiment time. Subsequent API
logging, UI icon, report/docs and test changes do not change the fitted estimator.
Future releases should commit the reviewed tree and repeat from a clean checkout.
The report and test outputs are separate: synthetic test fixtures never supply portfolio
metrics or EDA claims.

## Fresh-checkout reproduction

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements_dev.txt
python scripts/download_data.py
python -m src.cli validate --data-dir data/raw --output reports/local/quality
python -m src.cli eda --data-dir data/raw --output reports/local/eda
python -m src.cli train --data-dir data/raw --output artifacts/v2 --config configs/experiment.json
python -m src.cli evaluate --data-dir data/raw --model-dir artifacts/v2 --output reports/local/evaluation
python scripts/publish_report.py --run artifacts/v2 --output reports/local/published
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m build --no-isolation
python -m pip check
```

For another run use a new empty output, e.g. artifacts/v2-repro, and change --model-dir
or MODEL_DIR accordingly. Training is intentionally single-threaded. Floating-point
results can vary across platforms/library builds even with fixed seeds; compare
within numerical tolerance rather than promising universal bitwise identity.

```powershell
$env:MODEL_DIR="artifacts/v2"
python -m uvicorn api:app --host 127.0.0.1 --port 8080
# Second activated terminal:
python -m streamlit run app.py
# With Docker Desktop running:
docker build -t flight-fare-benchmark .
docker compose up --build
```

On POSIX use `source .venv/bin/activate` and `export MODEL_DIR=artifacts/v2`.
API documentation is /docs, readiness /health, liveness /ping and version /version.
Only mount artifacts generated or otherwise controlled by you.

## Remaining blockers and unverified work

No dataset or credential blocker remains. Docker execution, hosted CI, Linux/fresh
environment reproduction, original demo availability, public deployment, production
monitoring, current-market predictive validity and measured business impact remain
unverified. These limits are distinct from the locally passing implementation.
