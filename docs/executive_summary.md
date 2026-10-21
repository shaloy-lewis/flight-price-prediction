# Executive summary

This project now estimates **historical Indian domestic fare benchmarks** from listed
2022 offers. It makes no current-price, future-booking or verified savings claim.

The audit removed an invented booking-lead-time feature that differed between training
and inference, replaced row splitting with related-itinerary grouping, and corrected
encoded-feature explanation labels and log-price units.

The real-data workflow retains **300,255** validated offers after two duplicate removals
and four malformed-duration exclusions. A fixed **65,324-offer / 699-group** holdout
is excluded from model selection. Seven prespecified model families/target variants
are developed with grouped cross-validation.

The selected log-target histogram boosting model has **INR 3,425 MAE** versus
**5,164** for a route/class median reference, a **33.67% historical error reduction**.
Its MAE group-bootstrap 95% interval is **INR 3,208-3,620**. This supports a more
informative historical benchmark, not a cost-saving claim.

Material limitations remain: economy MAPE is **33.4%**, business MAE **INR 5,865**,
some expensive offers are underpredicted by more than **INR 47,000**, and rare-carrier
test evidence is absent or negligible. Current-market adoption requires recent
timestamped data and a prospective decision-impact evaluation.

The repository supplies tested pipelines, provenance/manifest checks, FastAPI and
Streamlit interfaces, actual reports and charts, CI configuration and lifecycle
guidance. Local Python/application checks are recorded; Docker execution and hosted
deployment remain unverified in this environment.
