<h1 align="center"> Flight Price Prediction</h1>

![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Overview
- This repository hosts a **Decision Tree Regressor** model, served via **FastAPI**, that predicts the price of the flight based on different flight parameters, and booking type.
- The app is deployed on streamlit. Try it out <a href="https://flight-price-prediction-shaloy-lewis.streamlit.app/"> here </a>
- Dataset obtained from <a href="https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction/"> Kaggle </a>

![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### A. Run with Docker
1. Clone the repository
```
git clone https://github.com/shaloy-lewis/flight-price-prediction.git
cd flight-price-prediction
```
2. Build and run the Docker container
```
docker-compose build
docker-compose up
```
3. Access the application
```
http://localhost:8080
```

### B. Run Locally Without Docker
1. Clone the repository
```
git clone https://github.com/shaloy-lewis/flight-price-prediction.git
cd flight-price-prediction
```
2. Create and activate virtualenv
```
pip install virtualenv
python3.12 -m venv venv
```
*For windows*
```
venv/Scripts/activate.bat
```
*For linux*
```
source venv/bin/activate
```
3. Install all the required packages and dependencies
```
pip install -r requirements.txt
```
5. Run the server
```
uvicorn api:app --reload --port 8080 --host 0.0.0.0
```
6. Access the application
```
http://localhost:8080
```
![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Getting Predictions
```
curl -X 'POST' \
  'http://localhost:[hostname]/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "airline": "AirAsia",
  "from_": "Mumbai",
  "to_": "Delhi",
  "class_": "economy",
  "ch_code": "SG",
  "dep_time": "06:20",
  "arr_time": "08:40",
  "date": "26-07-2024",
  "time_taken": "02h 20m",
  "stop": "non-stop"
}'
```
Change the hostname with the hostname given on your environment

![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Sample Response
```
{
    "flight_price_prediction": 5943.0,
    "instance_feature_importance": {
        "airline": 0.5287879073385162,
        "class": 0.016441714728573843,
        "dep_time": 0.005849110273980766,
        "arrival_time": 0.0035834280095936905,
        "departure_time": 0.0032865818800380748,
        "date": 0.0031789825306639784,
        "number_of_stops": 0.001178653839270607,
        "ch_code": 0.000325844048253295,
        "stop": 2.703823361866759e-05,
        "time_taken": -1.4638277106594559e-05,
        "arr_time": -0.002341867410095442,
        "flight_duration": -0.0024083451790196725,
        "to": -0.005528481238562058,
        "days_prior_booked": -0.00744738610425843,
        "from": -0.5412209047247729
    },
    "global_feature_importance": {
        "catategorical_pipeline__class_economy": 0.825996096511394,
        "numeric_pipeline__days_prior_booked": 0.06965641182230739,
        "numeric_pipeline__flight_duration": 0.04889857169871642,
        "catategorical_pipeline__ch_code_I5": 0.010095919275278187,
        "catategorical_pipeline__ch_code_6E": 0.007379716167384243,
        "numeric_pipeline__number_of_stops": 0.002994974850902845,
        "catategorical_pipeline__from_Kolkata": 0.0027897883857408577,
        "catategorical_pipeline__to_Kolkata": 0.002605955922011959,
        "catategorical_pipeline__from_Delhi": 0.002418425972594556,
        "catategorical_pipeline__to_Delhi": 0.001970927583297048,
        "catategorical_pipeline__airline_GO FIRST": 0.001886015914578589,
        "catategorical_pipeline__from_Bangalore": 0.0018575331278740477,
        "catategorical_pipeline__from_Mumbai": 0.0015535805138020263,
        "catategorical_pipeline__airline_Air India": 0.0015238011597790556,
        "catategorical_pipeline__to_Hyderabad": 0.0014726464979581235,
        "catategorical_pipeline__to_Bangalore": 0.001306112231917684,
        "catategorical_pipeline__from_Chennai": 0.001263652113157942,
        "catategorical_pipeline__to_Mumbai": 0.00117482920515208,
        "catategorical_pipeline__to_Chennai": 0.0010908135165285783,
        "catategorical_pipeline__from_Hyderabad": 0.0010425139847594083,
        "catategorical_pipeline__departure_time_Afternoon": 0.0009455244755585916,
        "catategorical_pipeline__arrival_time_Afternoon": 0.0009327818282345858,
        "catategorical_pipeline__ch_code_UK": 0.0008809027874694206,
        "catategorical_pipeline__departure_time_Morning": 0.0008721034674493003,
        "catategorical_pipeline__airline_SpiceJet": 0.0008561417609649325,
        "catategorical_pipeline__arrival_time_Evening": 0.0008312479337044401,
        "catategorical_pipeline__departure_time_Night": 0.000826880953641222,
        "catategorical_pipeline__arrival_time_Night": 0.0007688211435331222,
        "catategorical_pipeline__departure_time_Evening": 0.0006515337190245183,
        "catategorical_pipeline__arrival_time_Morning": 0.0005651629357754848,
        "catategorical_pipeline__departure_time_Early_Morning": 0.0005304081450221916,
        "catategorical_pipeline__departure_time_Late_Night": 0.0005183919235368542,
        "catategorical_pipeline__arrival_time_Late_Night": 0.00038010003753011364,
        "catategorical_pipeline__arrival_time_Early_Morning": 0.0003764973402790061,
        "catategorical_pipeline__ch_code_AI": 0.0003564414927913605,
        "catategorical_pipeline__airline_Vistara": 0.0003121531672676206,
        "catategorical_pipeline__ch_code_G8": 0.00017089357618411095,
        "catategorical_pipeline__airline_Indigo": 0.00010462350155769459,
        "catategorical_pipeline__ch_code_SG": 8.691803348846706e-05,
        "catategorical_pipeline__airline_StarAir": 1.8663626452627498e-05,
        "catategorical_pipeline__airline_Trujet": 1.5529963323632574e-05,
        "catategorical_pipeline__ch_code_2T": 1.2459730892936362e-05,
        "catategorical_pipeline__airline_AirAsia": 7.532000423168445e-06,
        "catategorical_pipeline__ch_code_S5": 7.596259130254306e-13
    }
}
```
