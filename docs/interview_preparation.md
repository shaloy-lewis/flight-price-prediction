# Ten challenging interview questions

1. **What exactly does the model predict, and why is it not a forecasting model?**

   A listed INR fare conditional on a historical configuration. The raw files have
   travel dates but no per-offer booking/observation timestamps or repeated quote
   trajectories; they cannot establish a future-price forecasting task.

2. **Why remove a strong booking-lead-time feature instead of fixing the reference date?**

   A fixed 10 February 2022 reference is not an observed booking event, and today's
   date at inference creates skew. Shared invented timing is still conceptually wrong.
   Both derived lead time and travel date are excluded from predictors.

3. **How do you prevent duplicate and near-duplicate leakage?**

   Connected components link airline/route/flight IDs across dates/cabins and identical
   canonical model inputs across IDs. No fares determine groups. Fixed group holdout
   and GroupKFold prevent these components crossing evaluation boundaries.

4. **What does the holdout test, and what does it fail to test?**

   It tests new itinerary components within the historical sampling frame. It does
   not test future markets or every dependency between multi-leg offers. Genuine
   observation timestamps and better itinerary IDs are needed for temporal evaluation.

5. **How was the final model selected without test-set tuning?**

   A prespecified seven-candidate set and small grids use three-fold grouped-CV INR
   MAE. selection.json freezes the minimum-CV-MAE winner before holdout predictions.
   All final test tables are diagnostic; further tuning requires a new holdout.

6. **Why use a log target, and how do you interpret retransformation?**

   Raw/log tree and boosting variants were compared by original-INR MAE. The selected
   log histogram model is inverse-transformed by TransformedTargetRegressor. Exp of a
   fitted log conditional response is generally not an unbiased arithmetic mean;
   no holdout-fitted correction was applied.

7. **Why is R-squared 0.946 not enough to call the model highly accurate?**

   Cabin strongly separates fare scales. Economy R-squared is only 0.192, economy MAPE is
   33.45%, business MAE is INR 5,865 and one miss exceeds INR 57,000. Aggregate fit
   cannot establish segment usefulness or current-market transportability.

8. **How are the SHAP explanations technically valid?**

   The exact fitted preprocessor supplies both encoded inputs and feature names.
   Additivity is checked against the underlying model output. Selected-model SHAP
   reconstructs log(INR); exponentiating individual contributions does not produce
   additive INR effects. Correlation and tree-path assumptions limit interpretation.

9. **What does the bootstrap interval mean?**

   Resampling 699 holdout components 500 times gives an average-MAE interval of
   INR 3,208-3,620 conditional on this trained model and cluster definition. It is
   neither an individual prediction interval nor uncertainty over training/model selection.

10. **What evidence supports business value and what would you require before deployment?**

    Historical MAE improves 33.67% versus route/class median, not verified savings.
    Require recent legally usable timestamped offers, agreed decision costs, fresh
    prospective evaluation, calibrated risk estimates where needed, and container/CI,
    workload, monitoring and rollback checks. Only trusted versioned models may be loaded.
