# Remaining roadmap

## P0: preserve validity in any extension

- Keep the published holdout consumed; register a new evaluation before new tuning.
- Require genuine observation timestamps before restoring booking lead time or making
  forecasting claims. Audit inventory/fare inclusions and related multi-leg offers.
- Recheck parser/schema/feature/explanation invariants after any data/model change.
- Continue to reject incompatible or untrusted artifacts.

## P1: stronger analytical and decision evidence

- Acquire recent timestamped data with full itinerary and observation identifiers.
- Evaluate later observations and adequate airline/cabin/route support prospectively.
- Address high-fare underprediction and weak economy relative accuracy. Compare
  quantile/asymmetric losses only after stakeholder costs are defined.
- Calibrate individual uncertainty using a valid separate calibration design and
  report coverage by segment; average-error confidence intervals are insufficient.
- Assess grouping, attribution and model-selection sensitivity on fresh data.
- Run a real budget/quote decision study before claiming impact or ROI.

## P2: engineering evidence still to obtain

- Run GitHub Actions and build/run Docker on an available daemon; check the mounted
  trusted model, readiness, non-root access and API contract inside the container.
- Repeat from a clean commit/checkout, verify the Linux dependency lock and install
  package artifacts in a fresh environment.
- Measure representative latency/throughput and failure behavior before defining SLOs.
- Refresh dependency pins, review vulnerabilities and pin a verified base-image digest
  before a public deployment.
- Extend static typing at data/evaluation boundaries, establish telemetry retention
  and prospective outcome joins, and exercise version rollback.

No Kubernetes, cloud resource, distributed processing or feature-store work is needed
to answer these open questions.
