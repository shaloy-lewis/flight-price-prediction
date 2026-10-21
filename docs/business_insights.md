# Business insights and decision framework

Primary stakeholder: a travel-budget analyst using a historical reference, not a
traveler selecting when to book in today's market.

Evidence-supported findings:

1. Cabin is the dominant scale separator: training medians are **INR 5,742 economy**
   and **53,164 business**. Keep cabin-specific budgets and segment reporting.
2. A route/class median is a strong, cheap reference (holdout MAE **INR 5,164.26**).
   The selected model reduces this historical error to **3,425.24**, justifying
   comparison with ML while retaining the baseline as an audit reference.
3. Large percentage economy errors (**33.45% MAPE**) and expensive business misses
   make automatic fare approval or budget commitment inappropriate.
4. Carrier and route associations reflect product mix and sampling. They do not
   establish savings from switching airline or adding connections.

Decision flow: define a supported configuration -> obtain its historical model estimate
and route/class reference -> check support and historical segment error -> compare
with an independently obtained current quote -> escalate unusual discrepancies for
human review. The current UI shows the model estimate and support warnings; the
route/class baseline and segment diagnostics are offline reports, not a live quote
comparison service.

Underprediction can underfund travel or falsely characterize an actual quote as
expensive. Overprediction can reserve excess funds or approve a costly quote.
Track signed errors as well as absolute error. Mean error is **-635.50 INR** even
though underprediction occurs for only **43.24%** of test offers, demonstrating the
effect of large negative tails.

No booking-date optimization, causal pricing recommendation, realized savings,
conversion uplift or ROI is measured. For an operational extension, collect recent
timestamped offers and actual procurement decisions before evaluating whether better
benchmarks change outcomes.
