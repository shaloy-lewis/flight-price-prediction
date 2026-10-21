# Feature engineering and leakage controls

All serving and training share one canonical schema. The model artifact includes the
fitted ColumnTransformer and estimator; serving never reconstructs learned preprocessing.

| Feature | Definition and rationale |
|---|---|
| airline / origin / destination / cabin | Historical known offer configuration; one-hot encoded. Routes remain directional. |
| departure / arrival bucket | 00:00-05:59 Late_Night; 06:00-07:59 Early_Morning; 08:00-11:59 Morning; 12:00-16:59 Afternoon; 17:00-18:59 Evening; 19:00-23:59 Night. Same mapping in ingestion and legacy adapter. |
| duration_minutes | Explicit total duration including layovers. Accept integer hours/minutes, including >24 hours. No subtraction of clock times to invent multi-day duration. |
| stops | Ordinal 0, 1, 2; 2 is the source's two-or-more category. This is a coarse encoding, not an exact count above two. |

Rejected predictors: synthetic booking lead time, cleaned `days_left`, travel date,
flight number and carrier prefix. No observation timestamp independently supports a
booking lead-time feature. Travel date would confound short-window departure timing
with the former artificial reference-date feature. Flight number invites memorization;
carrier prefix duplicates airline. No seasonality or current-date calculation enters
the model.

The raw date is checked for a valid DD-MM-YYYY date, and carrier prefix must match
airline. Clock time plus explicit duration is checked modulo 24 hours in the legacy
adapter and Streamlit. The canonical API takes time buckets and total duration, so it
cannot check precise arrival timing. Raw source inconsistencies are counted. A 72-hour
duration ceiling is an explicit service scope rule, not a learned outlier threshold.

Transformations create new frames and never mutate callers' inputs. Numeric medians,
scaling parameters, categorical modes and encoder categories are fit inside each CV
training fold. Unknown but schema-valid categories use all-zero one-hot columns with
support warnings; missing request fields are rejected. Imputers provide controlled
behavior for internal model/fixture use, not permission to submit incomplete requests.
No percentile duration caps are used: long connecting itineraries can be legitimate,
and held-out data must not determine thresholds.

A fixed GroupShuffleSplit locks final groups. GroupKFold on development groups tunes
hyperparameters; no fare or target statistic determines grouping. Identical model inputs
and repeated flight identities cannot cross either boundary. Constructing identity
groups before splitting is deliberate split design, not learned target preprocessing.
Residual dependence beyond these group definitions remains a limitation.
