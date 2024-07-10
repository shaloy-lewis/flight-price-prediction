from scipy.stats import randint, uniform

CATEGORICAL_FEATURES=['airline', 'from', 'to', 'class', 'ch_code','departure_time','arrival_time']
NUMERIC_FEATURES=['days_prior_booked','flight_duration','number_of_stops']
TARGET_FEATURE='price'

OUTLIER_COL = ['flight_duration']
OUTLIER_CAPPING_LOWER_THRESHOLD=1
OUTLIER_CAPPING_UPPER_THRESHOLD=99

# PARAM_SPACE = {
#     'criterion': ['friedman_msea', 'poisson', 'squared_error', 'absolute_error'],
#     'max_depth': [None, 10, 20, 30, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': [None, 'sqrt', 'log2'],
#     'splitter': ['best', 'random']
# }