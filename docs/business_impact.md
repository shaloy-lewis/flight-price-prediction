# Business impact: explicit scenarios, not measured savings

Measured evidence is a reduction in historical holdout MAE from **INR 5,164.26**
(route/class reference) to **3,425.24** (selected model), difference **INR 1,739.02**
per offer. The paired group-bootstrap difference interval is **-2,065.01 to
-1,415.15 INR** (model minus baseline). These are predictive errors on historical
offers, not procurement savings.

A transparent budgeting scenario defines **gross absolute allocation discrepancy** as

`N comparable allocations x historical MAE x error-scale multiplier`.

Assumptions: each allocation is equally represented by the held-out 2022 offers;
configuration and observation conditions are comparable; historical MAE transports;
there is one allocation per offer; no selection or behavioral response occurs.
Those assumptions are unverified for current corporate travel. Gross absolute
discrepancy adds errors without cancellation and is not the signed total budget error.

| Hypothetical allocations | Baseline gross discrepancy (INR) | Model gross discrepancy (INR) | Reduction in this scenario (INR) |
|---:|---:|---:|---:|
| 10 | 51,642.63 | 34,252.40 | 17,390.23 |
| 100 | 516,426.35 | 342,524.02 | 173,902.32 |
| 1,000 | 5,164,263.46 | 3,425,240.25 | 1,739,023.21 |

Sensitivity for **100 hypothetical allocations**: if error scale is 0.5x, 1x or 2x
the historical sample, the modeled gross discrepancy is **INR 171,262 / 342,524 /
685,048**, respectively. This simple multiplier expresses transportability risk;
it is not a fitted future error model.

A second decision-cost formulation separates asymmetric costs:

`loss = c_under x sum(max(actual - estimate, 0)) + c_over x sum(max(estimate - actual, 0))`.

The stakeholder must specify these costs and downstream actions before choosing a
buffer, quantile model or approval threshold. This implementation selects symmetric
MAE and reports signed/segment errors; it does not claim to optimize procurement cost.
Do not convert an MAE reduction directly into savings, assume errors cancel independently,
or use a bootstrap average-error interval as an individual contingency.

Prospective impact study: establish a quote-based baseline and decision protocol;
record timestamps, configurations, model versions, accepted quotes and outcomes;
compare budget shortfalls, unnecessary reserves and decision time against agreed
costs, controlling for request mix. Only then estimate net benefit after maintenance
and data-acquisition costs.
