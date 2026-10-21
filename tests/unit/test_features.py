from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.components.data_transformation import make_preprocessor
from src.schema import FlightRequest, LegacyFlightRequest
from src.utils.constants import FEATURES
from src.utils.utils import (
    categorize_time,
    convert_to_minutes,
    parse_stops,
    parse_travel_date,
    validate_clock_duration,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("02h 20m", 140),
        ("2h", 120),
        ("45m", 45),
        ("25h 10m", 1510),
        (" 00h 05m ", 5),
    ],
)
def test_duration(value, expected):
    assert convert_to_minutes(value) == expected


@pytest.mark.parametrize("value", ["1.03h m", "2h 60m", "0h 0m", "-1h", "2h junk", "", "73h"])
def test_invalid_duration(value):
    with pytest.raises(ValueError):
        convert_to_minutes(value)


@pytest.mark.parametrize(
    ("clock", "bucket"),
    [
        ("00:00", "Late_Night"),
        ("06:00", "Early_Morning"),
        ("08:00", "Morning"),
        ("12:00", "Afternoon"),
        ("17:00", "Evening"),
        ("19:00", "Night"),
        ("23:59", "Night"),
    ],
)
def test_bucket_edges(clock, bucket):
    assert categorize_time(clock) == bucket


@pytest.mark.parametrize("clock", ["24:00", "12:60", "-1:00", "6:20", "12:20x"])
def test_invalid_clocks(clock):
    with pytest.raises(ValueError):
        categorize_time(clock)


@pytest.mark.parametrize(
    ("text", "expected"), [("non-stop ", 0), ("1-stop\n\t", 1), ("2+stop", 2), ("2+-stop", 2)]
)
def test_stops(text, expected):
    assert parse_stops(text) == expected


def test_invalid_stops_and_dates():
    with pytest.raises(ValueError):
        parse_stops("11-stop")
    with pytest.raises(ValueError):
        parse_travel_date("31-02-2022")
    assert parse_travel_date("29-02-2024").year == 2024


def test_overnight_and_multiday():
    validate_clock_duration("23:00", "01:00", 120)
    validate_clock_duration("23:00", "01:00", 1560)
    with pytest.raises(ValueError):
        validate_clock_duration("23:00", "01:00", 90)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("duration_minutes", 0),
        ("duration_minutes", float("nan")),
        ("duration_minutes", "120"),
        ("stops", True),
        ("stops", 1.0),
        ("stops", 3),
        ("source_city", "Unknown"),
        ("cabin_class", None),
        ("airline", "New Airline"),
        ("departure_time", "25:00"),
    ],
)
def test_schema_rejects_invalid(request_payload, key, value):
    request_payload[key] = value
    with pytest.raises(ValidationError):
        FlightRequest.model_validate(request_payload)


def test_distinct_cities_and_extra_fields(request_payload):
    request_payload["destination_city"] = request_payload["source_city"]
    with pytest.raises(ValidationError):
        FlightRequest.model_validate(request_payload)
    request_payload["date"] = "01-01-2030"
    with pytest.raises(ValidationError):
        FlightRequest.model_validate(request_payload)


def test_preprocessor_fold_local_missing_unseen_and_immutable(fixture_frame):
    train = fixture_frame[FEATURES].iloc[:30].copy()
    original = deepcopy(train)
    processor = make_preprocessor().fit(train)
    means = processor.named_transformers_["numeric"].named_steps["scale"].mean_.copy()
    future = train.iloc[:2].copy()
    future.loc[future.index[0], "duration_minutes"] = np.nan
    future.loc[future.index[0], "airline"] = np.nan
    future.loc[future.index[1], "airline"] = "UNSEEN"
    future.loc[future.index[1], "duration_minutes"] = 4000
    encoded = processor.transform(future)
    assert np.isfinite(encoded).all()
    np.testing.assert_array_equal(
        means, processor.named_transformers_["numeric"].named_steps["scale"].mean_
    )
    pd.testing.assert_frame_equal(train, original)
    assert encoded.shape[1] == len(processor.get_feature_names_out())


def test_legacy_ignores_reference_date():
    payload = dict(
        airline="AirAsia",
        from_="Delhi",
        to_="Mumbai",
        class_="economy",
        ch_code="I5",
        dep_time="23:00",
        arr_time="01:00",
        time_taken="26h",
        stop="2+stop",
    )
    old = LegacyFlightRequest(**payload, date="12-02-2022").to_canonical()
    new = LegacyFlightRequest(**payload, date="12-02-2030").to_canonical()
    assert old == new
    payload["ch_code"] = "SG"
    with pytest.raises(ValueError):
        LegacyFlightRequest(**payload, date="12-02-2022").to_canonical()
