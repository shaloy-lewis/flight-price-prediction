"""Strict public inference schema. Raw dates are metadata, never booking lead time."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.utils.constants import AIRLINE_CODES, CITIES, TIME_BUCKETS
from src.utils.utils import (
    categorize_time,
    convert_to_minutes,
    parse_stops,
    parse_travel_date,
    validate_clock_duration,
)


class FlightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    airline: str
    source_city: str
    destination_city: str
    cabin_class: Literal["economy", "business"]
    departure_time: str
    arrival_time: str
    duration_minutes: float = Field(gt=0, le=4320, allow_inf_nan=False)
    stops: Literal[0, 1, 2]

    @field_validator("airline")
    @classmethod
    def airline_known(cls, value: str) -> str:
        if value not in AIRLINE_CODES:
            raise ValueError("Airline is outside the historical service scope")
        return value

    @field_validator("source_city", "destination_city")
    @classmethod
    def city_known(cls, value: str) -> str:
        if value not in CITIES:
            raise ValueError("City is outside the historical service scope")
        return value

    @field_validator("departure_time", "arrival_time")
    @classmethod
    def bucket_known(cls, value: str) -> str:
        if value not in TIME_BUCKETS:
            raise ValueError("Unrecognized time bucket")
        return value

    @field_validator("stops", mode="before")
    @classmethod
    def stops_integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Stops must be an integer: 0, 1 or 2 (two or more)")
        return value

    @model_validator(mode="after")
    def distinct_cities(self) -> "FlightRequest":
        if self.source_city == self.destination_city:
            raise ValueError("Departure and destination must differ")
        return self


class LegacyFlightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    airline: str
    from_: str
    to_: str
    class_: str
    ch_code: str
    dep_time: str
    arr_time: str
    date: str
    time_taken: str
    stop: str

    def to_canonical(self) -> FlightRequest:
        parse_travel_date(self.date)
        if AIRLINE_CODES.get(self.airline) != self.ch_code:
            raise ValueError("Airline code does not match airline")
        duration = convert_to_minutes(self.time_taken)
        validate_clock_duration(self.dep_time, self.arr_time, duration)
        return FlightRequest.model_validate(
            {
                "airline": self.airline,
                "source_city": self.from_,
                "destination_city": self.to_,
                "cabin_class": self.class_,
                "departure_time": categorize_time(self.dep_time),
                "arrival_time": categorize_time(self.arr_time),
                "duration_minutes": float(duration),
                "stops": parse_stops(self.stop),
            }
        )
