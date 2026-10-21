# Problem statement

Estimate the **observed listed fare in INR for a historical Indian domestic flight
configuration**, under this dataset's sampling process. A listed offer is not evidence
of a completed purchase or an all-inclusive final payment.

The primary stakeholder is a corporate travel analyst exploring retrospective budget
benchmarks. At the comparison decision, airline, origin/destination city, cabin, scheduled
departure/arrival time categories, itinerary duration and stop category are known.
The output helps compare a configuration with historical offers. A current supplier
quote remains necessary for any present-day purchase or budget.

This is cross-sectional historical fare estimation. It is not a live quote, a forecast
of a future fare, a probability that prices will rise, or an optimal booking-date tool.
The raw files contain travel dates but no per-offer observation/booking timestamp.
The cleaned file's derived `days_left` does not supply independently verifiable
timestamps. The former fixed-date subtraction is removed.

The non-ML decision is the training-only median fare for the same directional
route and cabin, with a global median fallback. The model must improve that reference
on unseen itinerary groups to justify additional complexity. Success is lower and
more stable historical budgeting error; no measured savings or current-market benefit
is asserted. Underestimation can leave a budget short; overestimation can tie up budget
or incorrectly flag a reasonable quote.

Constraints: offers cover six cities and a short 2022 travel window, with uneven
airline/cabin representation, no inventory, purchase outcomes, fare conditions,
observation timestamps, or longitudinal trajectories. Taxes and inclusions are not
established. A future operational pilot requires legally obtainable recent data,
observed timestamps, prospectively defined decision costs, and evaluation on later
observations.
