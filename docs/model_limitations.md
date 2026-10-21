# Model limitations and failure modes

This model describes 2022 listed domestic fares under the source sampling process.
It cannot verify a current price, recommend waiting to book, or establish causal
effects of carrier, class, stops, travel date or duration.

Reliability is weak or unestablished for rare carriers, unsupported routes/cabins,
extreme fares and durations, unusual connections, fare inclusions, and all future
market regimes. Even known category combinations may have little joint training
support; marginal/category warnings are not a complete joint-density detector.
Historical carrier labels are not a current operating-carrier directory.

Omitting unverified booking timing is necessary but limits identifiability: identical
canonical configurations have different fares. Increasing tree depth cannot recover
unobserved inventory, observation time or fare rules. Segment price buckets are
retrospective error diagnostics; a user's true fare is unknown before inference.

The test targets previously unseen related-itinerary components within one historical
window. It provides no temporal transportability evidence. Grouping cannot capture
every shared leg or scrape session. Bootstrap confidence intervals are conditional
average-error summaries, not per-itinerary intervals, and may understate dependence
not represented by the groups.

Use the estimate for historical comparison and educational budgeting scenarios. Obtain
a current quote and independent judgment for purchases, commitments, reimbursements
or high-cost exceptions. A credible operational extension needs timestamped recent
observations, complete itinerary identifiers, fare inclusions, and a prospective
evaluation with agreed error costs.
