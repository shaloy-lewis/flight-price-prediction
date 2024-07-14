import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime, time
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

# Initialize prediction pipeline
predict_pipeline = PredictPipeline()

# Streamlit app title
st.title("Flight Price Prediction")

page = st.sidebar.selectbox('Page Navigation', ["Problem statement", "Predictor",])

st.sidebar.markdown("""---""")
st.sidebar.write("Created by [Shaloy Lewis](https://www.linkedin.com/in/shaloy-lewis/)")
available_destinations = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']

if page=="Problem statement":
    st.write("""Predicting flight prices provides significant advantages for both businesses and individuals. For businesses, such predictions enable better budget management and planning of business travel, conferences, and events by anticipating cost fluctuations. It also allows companies in the travel industry to adjust pricing strategies dynamically, potentially gaining a competitive edge.
             
For individuals, predicting flight prices helps in saving money by booking tickets during periods of lower prices and allows for better planning of travel dates and destinations.

Overall, this capability enhances customer satisfaction, operational efficiency, and ensures more informed decision-making, benefiting both travelers and businesses involved in the travel sector.
            """)

else:
    st.subheader("Enter travel details:")
    date=st.date_input("Date of travel", format="DD-MM-YYYY", min_value=date.today()+timedelta(days=1), max_value=date.today()+timedelta(days=49))
    col1 = st.columns(2)
    from_=col1[0].selectbox('From', options=available_destinations)
    if from_:
        available_destinations.remove(from_)
    to_ = col1[1].selectbox('To', options=available_destinations) if from_ else col1.empty()
    col2=st.columns(3)
    airline=col2[0].selectbox('Airline', options=['SpiceJet', 'AirAsia', 'Vistara', 'GO FIRST', 'Indigo','Air India', 'Trujet', 'StarAir'])
    class_=col2[1].selectbox('Flight class', options=['economy', 'business'])
    ch_code=col2[2].selectbox('Airline character code', options=['SG', 'I5', 'UK', 'G8', '6E', 'AI', '2T', 'S5'])
    col3=st.columns(4)
    dep_time=col3[0].time_input("Departure time HH MM",value=time(19, 00))
    arr_time=col3[1].time_input("Arrival time HH MM",value=time(21, 00))
    if dep_time and arr_time:
        # Convert to string format
        dep_time_str = dep_time.strftime("%H:%M")
        arr_time_str = arr_time.strftime("%H:%M")

        # Calculate flight duration
        departure = datetime.combine(date.today(), dep_time)
        arrival = datetime.combine(date.today(), arr_time)
        flight_duration = arrival - departure
        hours, remainder = divmod(flight_duration.seconds, 3600)
        minutes = remainder // 60
        time_taken = f"{hours}h {minutes}m"

        time_taken=col3[2].text_input("Flight duration",value=time_taken)
    stop=col3[3].selectbox('Number of stops', options=['non-stop','1-stop','2+stop'])

    # Button to submit data and get predictions
    if st.button('Predict'):
        custom_data = CustomData(
            airline=airline,
            from_=from_,
            to_=to_,
            class_=class_,
            ch_code=ch_code,
            dep_time=dep_time.strftime("%H:%M"),
            arr_time=arr_time.strftime("%H:%M"),
            date=date,
            time_taken=time_taken,
            stop=stop
        )

        # Convert to DataFrame
        features_df = custom_data.get_data_as_dataframe()

        # Initialize the prediction pipeline and make predictions
        pipeline = PredictPipeline()
        prediction = pipeline.predict(features_df)
        
        # Display the prediction results
        st.subheader("Prediction Results:")
        st.write(f"Estimated flight price: INR {prediction[0]:.2f}")