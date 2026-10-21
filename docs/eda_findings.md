# EDA findings

Executed by `python -m src.cli eda` on **234,931 training offers only**, using the same
locked split/configuration as model development. Full-source validity counts are
reported separately. Exact grouped summaries are in
[eda_summary.json](../reports/reference/eda_summary.json).

| Figure | Observed finding | Modeling / business relevance |
|---|---|---|
| [Fare distributions](../reports/reference/figures/price_distribution.png) | Training economy median INR 5,742; business median INR 53,164. Raw price is strongly separated by cabin, and logging does not remove that separation. | Include cabin explicitly; compare raw/log targets empirically. Aggregate R-squared can be dominated by class differences. |
| [Airline and cabin](../reports/reference/figures/airline_cabin.png) | Air India and Vistara are the only business carriers represented. Economy medians include AirAsia 3,090, Indigo 4,453, Air India 6,048 and Vistara 6,461 INR. Missing bars mean absent combinations, not zero fares. | Carrier comparisons require class/route controls; do not interpret a marginal carrier difference as a saving from switching airline. |
| [Duration and stops](../reports/reference/figures/duration_stops.png) | Duration-price Pearson correlation is 0.203; duration-stops correlation 0.471 and stops-price 0.124. Scatter has substantial within-cabin variability. | Simple monotonic assumptions are insufficient. Duration/stops are correlated; importance can be shared or unstable. Connections have convenience costs not captured by fare. |
| [Economy route medians](../reports/reference/figures/routes.png) | Route medians range from INR 4,325 for Mumbai to Hyderabad to 6,489 for Kolkata to Bangalore in training. The diagonal is absent by design. | A directional route/class median is a realistic and inexpensive non-ML reference. Route mix confounds comparisons. |
| [Travel-date profile](../reports/reference/figures/travel_dates.png) | Median listed fares vary across travel dates within the short source window. | These are travel-date associations, not repeated quote trajectories. They do not establish booking timing, annual seasonality or a forecastable temporal process. Travel date is excluded from predictors. |

No booking lead-time chart is produced: an observation/booking timestamp is not present
in the raw files. There is no evidential basis for an optimal purchase-date claim.

The data-quality findings matter more than cosmetic plots: four malformed durations
are excluded; repeated feature vectors and conflicting fares motivate grouped validation
and an explicit irreducible-information limitation. Rare carriers and unsupported
airline/cabin combinations need visible support counts. Full-source quality and
training-only descriptive EDA do not establish current-market representativeness.

The original notebook is archived with outputs cleared and its deployment conclusion
withdrawn. The CLI analysis is the maintained reproducible workflow.
