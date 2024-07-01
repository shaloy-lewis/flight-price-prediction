CATEGORICAL_FEATURES=['airline', 'from', 'to', 'class', 'flight_code','departure_time','arrival_time']
NUMERIC_FEATURES=['days_prior_booked','flight_duration','number_of_stops']
TARGET_FEATURE='price'

OUTLIER_COL = ['flight_duration']
OUTLIER_CAPPING_LOWER_THRESHOLD=1
OUTLIER_CAPPING_UPPER_THRESHOLD=99

PARAM_GRID = {
    'n_estimators': [100, 200, 500, 1000],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}