"""Pure parsing helpers shared by ingestion and the legacy request adapter."""

import re
from datetime import date, datetime
from functools import lru_cache

from src.utils.constants import TIME_BUCKETS


@lru_cache(maxsize=2048)
def clock_minutes(value: str) -> int:
    if not re.fullmatch(r"\d{2}:\d{2}", value.strip()):
        raise ValueError("Time must use HH:MM")
    parsed = datetime.strptime(value.strip(), "%H:%M")
    return 60 * parsed.hour + parsed.minute


@lru_cache(maxsize=2048)
def categorize_time(value: str) -> str:
    minutes = clock_minutes(value)
    boundaries = (360, 480, 720, 1020, 1140, 1440)
    return next(bucket for limit, bucket in zip(boundaries, TIME_BUCKETS) if minutes < limit)


@lru_cache(maxsize=4096)
def convert_to_minutes(value: str) -> int:
    match = re.fullmatch(r"\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*", value)
    if not match or not any(match.groups()):
        raise ValueError("Duration must use hours/minutes, e.g. 02h 20m")
    hours, minutes = (int(part or 0) for part in match.groups())
    total = hours * 60 + minutes
    if minutes >= 60 or not 0 < total <= 4320:
        raise ValueError("Duration must be positive, at most 72h, with minutes below 60")
    return total


@lru_cache(maxsize=256)
def parse_stops(value: str) -> int:
    # Raw source sometimes appends connection location after a tab/newline.
    token = value.strip().split("\n")[0].split("\t")[0].strip().lower()
    mapping = {
        "non-stop": 0,
        "1-stop": 1,
        "2+-stop": 2,
        "2+stop": 2,
        "zero": 0,
        "one": 1,
        "two_or_more": 2,
    }
    if token not in mapping:
        raise ValueError("Stops must be non-stop, 1-stop or 2+-stop")
    return mapping[token]


@lru_cache(maxsize=256)
def parse_travel_date(value: str) -> date:
    if not re.fullmatch(r"\d{2}-\d{2}-\d{4}", value.strip()):
        raise ValueError("Travel date must use DD-MM-YYYY")
    return datetime.strptime(value.strip(), "%d-%m-%Y").date()


def validate_clock_duration(departure: str, arrival: str, duration: int) -> None:
    if (clock_minutes(arrival) - clock_minutes(departure)) % 1440 != duration % 1440:
        raise ValueError("Duration and clock times disagree, including overnight/multi-day travel")
