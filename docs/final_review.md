# Final review: hiring-panel perspective

This is an evidence-based self-review from three hiring perspectives, not an external
panel or a numerical certification. Findings refer to executed v2 results and current
source; the published holdout is already consumed.

| Dimension | Repository evidence | Strong | Weak / remaining risk | Concrete next improvement |
|---|---|---|---|---|
| 1. Business framing | problem_statement.md; success_criteria.md | Narrow historical budget benchmark; explicit target, timing, stakeholder and baseline. | Operational relevance to today's travel decisions is untested. | Agree a prospective decision protocol and error costs with a real stakeholder. |
| 2. Data understanding | data_dictionary.md; data_quality_report.md; analysis.py | Original source/CC0 metadata, hashes, malformed-duration quarantine, duplicate and rare-carrier counts. | Unknown sampling, observation times and fare inclusions. | Acquire recent timestamped complete offers and validate source sampling. |
| 3. Statistical rigor | model_evaluation.py; model_evaluation.md | Group-aware evaluation, original-unit metrics, paired cluster bootstrap and caveats. | Bootstrap is conditional; residual dependencies may escape grouping. | Audit itinerary/scrape IDs and assess group/metric sensitivity on a fresh study. |
| 4. Experimental methodology | configs/experiment.json; training_pipeline.py; selection.json | Prespecified small search; fold-local preprocessing; selection frozen before test. | One dataset/split; limited hyperparameter exploration and no external replication. | Prospectively register a follow-up split and selection/complexity rule. |
| 5. Features and leakage | schema.py; data_ingestion.py; feature_engineering.md | Clock-independent canonical features; flight and identical-input components; leakage tests. | Omitted true booking timing limits identifiability; unknown multi-leg dependence. | Obtain real observation timestamps and full itinerary identifiers before adding timing. |
| 6. Model evaluation | results.json; segment_errors.csv; largest_errors.csv | Seven candidates, realistic median baseline, residuals, segment counts and expensive failures. | Weak within-economy R-squared, high relative error, almost no rare-carrier test support. | Design a new stratified-by-business-scope grouped/temporal evaluation with adequate support. |
| 7. Explainability | explainability.py; explanations.json; serving tests | Exact encoded names, checked additivity and explicit log(INR) units; permutation separated. | Correlated features and tree-path dependence; small global sample. | Assess attribution stability across refits/reference choices and aggregate original-field contributions carefully. |
| 8. Engineering quality | pyproject.toml; API/UI; tests; workflow | Typed contracts, modular code, clean errors, caching, tests/lint/static/build checks. | Static typing currently targets contracts/config/artifacts/API, not all pandas-heavy modules; third-party deprecations remain. | Extend types at module boundaries and refresh/test dependencies. |
| 9. Reproducibility | downloader; sources; split hash; source snapshot; environment | Real data retrieval verified by hashes; deterministic fixtures; versioned run outputs. | Historical source can disappear; training ran on a dirty working tree rather than a release commit. | Commit/release the reviewed tree and repeat from a clean checkout on CI infrastructure. |
| 10. Architecture | architecture.md; Dockerfile; artifacts.py | One fitted pipeline; trusted versioned artifact; process/resource caches; non-root container design. | Docker build/runtime, Linux execution and hosted CI are unverified locally. | Run the configured CI and container smoke/load checks on a host with Docker. |
| 11. Business impact / communication | business_impact.md; executive_summary.md; README | Error reduction separated from savings; scenarios and asymmetric cost formulation are explicit. | No actual procurement outcomes or decision study. | Measure outcomes against a realistic quote-based baseline prospectively. |
| 12. Technical judgment | technical_decisions.md; migration.md | Rejects unsupported forecasting/infrastructure; records model/runtime/explanation trade-offs and breaking changes. | Alternative grouping and cost-sensitive target choices remain unexplored. | Test alternatives on new data, with decision costs fixed before evaluation. |

## Role-specific readiness

- **Senior Data Scientist:** credible evidence of problem framing, leakage control,
  baseline comparison, segment interpretation and honest communication. Interview-ready
  as a historical benchmarking case; incomplete as evidence of deployed business value.
- **Senior ML Engineer:** meaningful local packaging, serving, artifact and test work.
  Shows sound scope judgment; deployment, workload behavior, security maintenance,
  observability and real lifecycle operation remain unproven.
- **Applied Scientist:** defensible conventional experimental design and clear
  uncertainty limitations. Does not demonstrate a novel method, causal identification,
  longitudinal forecasting or broad external validity.

The strongest evidence is the correction of unsupported timing and evaluation leakage,
not the high aggregate R-squared. No readiness label should obscure the segment failures or
the absence of current-market and operational evidence.
