"""Versioned, clock-independent historical fare contract."""

SCHEMA_VERSION = "2.0"
PROJECT_VERSION = "2.0.0"
SCOPE = "Historical Indian domestic fare benchmark; not a current quote or forecast."
AIRLINE_CODES = {
    "SpiceJet": "SG",
    "AirAsia": "I5",
    "Vistara": "UK",
    "GO FIRST": "G8",
    "Indigo": "6E",
    "Air India": "AI",
    "Trujet": "2T",
    "StarAir": "S5",
}
CITIES = ("Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai")
TIME_BUCKETS = ("Late_Night", "Early_Morning", "Morning", "Afternoon", "Evening", "Night")
CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "destination_city",
    "cabin_class",
    "departure_time",
    "arrival_time",
]
NUMERIC_FEATURES = ["duration_minutes", "stops"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_FEATURE = "price"
