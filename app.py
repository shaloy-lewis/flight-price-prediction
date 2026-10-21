"""Streamlit interface for historical configuration comparisons."""

from datetime import time

import pandas as pd
import streamlit as st

from src.artifacts import ArtifactError
from src.pipeline.prediction_pipeline import PredictPipeline
from src.schema import FlightRequest
from src.utils.constants import AIRLINE_CODES, CITIES, SCOPE
from src.utils.utils import categorize_time, validate_clock_duration

st.set_page_config(page_title="Historical Fare Benchmark", page_icon=":airplane:")
st.title("Historical Flight Fare Benchmark")
st.info(SCOPE)
st.caption(
    "Based on Indian domestic offers for February-March 2022. No live airline or booking data is connected."
)


@st.cache_resource
def load_predictor():
    return PredictPipeline()


page = st.sidebar.selectbox("Page", ["Problem statement", "Predictor"])
if page == "Problem statement":
    st.write(
        "Estimate a historical fare benchmark for a specified itinerary to explore travel-budget assumptions. "
        "Compare the estimate with a route/class median and obtain a current quote before making a purchase."
    )
    st.write(
        "This dataset cannot identify the best future booking date or measure cost savings. "
        "Differences between airlines, classes and routes are associations, not causal effects."
    )
else:
    try:
        predictor = load_predictor()
    except ArtifactError:
        st.error(
            "A compatible model is unavailable. Follow the README training instructions and set MODEL_DIR."
        )
        st.stop()
    reference = predictor.manifest["reference"]
    st.caption(
        f"Model version: {predictor.manifest['model_version']}; "
        "individual prediction intervals have not been calibrated."
    )
    with st.form("itinerary"):
        cols = st.columns(2)
        origin = cols[0].selectbox("From", CITIES)
        destination = cols[1].selectbox("To", CITIES, index=1)
        cols = st.columns(2)
        airline = cols[0].selectbox(
            "Historical airline", sorted(reference["categorical"]["airline"])
        )
        cabin = cols[1].selectbox("Cabin", ["economy", "business"])
        cols = st.columns(3)
        departure = cols[0].time_input("Departure time", value=time(19, 0))
        arrival = cols[1].time_input("Arrival time", value=time(21, 0))
        duration = cols[2].number_input(
            "Total duration (minutes, including connections)",
            min_value=1,
            max_value=4320,
            value=120,
        )
        stops = st.selectbox("Stops (2 means two or more)", [0, 1, 2])
        st.caption(
            f"Airline code: {AIRLINE_CODES[airline]}. Enter the full duration for overnight or multi-day itineraries."
        )
        submitted = st.form_submit_button("Estimate historical fare")
    if submitted:
        try:
            validate_clock_duration(
                departure.strftime("%H:%M"), arrival.strftime("%H:%M"), duration
            )
            request = FlightRequest(
                airline=airline,
                source_city=origin,
                destination_city=destination,
                cabin_class=cabin,
                departure_time=categorize_time(departure.strftime("%H:%M")),
                arrival_time=categorize_time(arrival.strftime("%H:%M")),
                duration_minutes=float(duration),
                stops=stops,
            )
            frame = pd.DataFrame([request.model_dump()])
        except ValueError as exc:
            st.error(f"Please check the itinerary: {exc}")
        else:
            for warning in predictor.warnings(frame):
                st.warning(warning)
            try:
                estimate = predictor.predict(frame)[0]
                st.metric("Estimated historical fare (INR)", f"{estimate:,.2f}")
                st.caption(
                    "This is a model estimate of a historical offer, not a verified present-day fare. "
                    "Compare a current quote before budgeting or booking."
                )
            except Exception:
                st.error("The estimate could not be generated. Check the model and server logs.")
