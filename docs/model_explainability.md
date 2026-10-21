# Explainability

The former implementation processed local explanations differently from predictions,
zipped encoded SHAP contributions against original columns, and omitted the log scale.
Those outputs are withdrawn.

The v2 Explainer obtains the exact fitted preprocessor from the deployed estimator,
transforms the same canonical request, and uses get_feature_names_out() for labels.
It verifies dimensions and additive reconstruction before returning results.
Tests cover one-hot labels, prediction/explanation equality, tree/histogram boosting,
raw/log targets, linear contributions and median baselines.

| Method | Output and interpretation |
|---|---|
| Offline permutation importance | Increase in original-INR MAE after shuffling each **raw canonical column** on a fixed 2,000-offer holdout sample, five repeats. Measures dependence of predictive accuracy on that column under this perturbation. |
| Tree SHAP | Local tree_path_dependent SHAP contributions on the **encoded model features**. For the selected model, base + contributions reconstructs **log(INR)**, not INR. |
| Linear contributions | Exact centered coefficients times transformed-feature deviations from training means. Explicitly labeled **not SHAP**. |
| Route/class reference | Difference between the route/class lookup and global training median, in INR. It is a lookup decomposition, not a causal effect or SHAP. |
| Impurity importance | Not used as the authoritative global report; it is not equivalent to permutation importance or SHAP. |

Executed permutation MAE increases: cabin **17,740.33 INR**, duration **1,645.18**,
airline **545.20**, destination **410.45**, origin **328.04**, stops **140.01**,
arrival bucket **94.22**, departure bucket **4.92**. Repeat SD for departure is
**13.99**, so its tiny estimated contribution is unstable. Values are descriptive
and sample-dependent; the reported repeat SD is not a population confidence interval.
See [permutation report](../reports/reference/permutation_importance.json).

[explanations.json](../reports/reference/explanations.json) includes one local example
and mean absolute attributions across a deterministic sample of 100 holdout offers.
Those global descriptive values are in **log(INR)** on encoded features. The local
example's additive residual is approximately -5.3e-15. A real HTTP example is
[recorded separately](../reports/reference/api_example.json).

For a log-target model, exp(base + sum(contributions)) reconstructs the currency
prediction. Exponentiating each contribution separately does **not** create additive
INR contributions: nonlinear retransformation has interactions and depends on the
reference. Neither the API nor UI labels these values as INR effects.

One-hot fields retain their exact encoded names; category contributions may be summed
for a single original field in the same model-output scale, but no raw-label zip or
silent truncation is permitted. Duration/stops/time categories are correlated;
one-hot columns are constrained and carriers/classes/routes are not independent.
Permutation can create implausible combinations; tree-path SHAP uses a different
reference dependence assumption. Both can change across retraining and background
assumptions. Predictive attribution is not causal evidence or a booking recommendation.

Global work runs offline once per model version. The API only computes a local
explanation when requested, and reuses its initialized explainer. No global importance
is recomputed for each user request. An individual uncertainty interval is not supplied.
