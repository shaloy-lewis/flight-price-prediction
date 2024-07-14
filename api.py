from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date, timedelta
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = FastAPI(
    title='Flight Price Prediction',
    description="Predicts price of flight using the travel details input by the user"
)

@app.get("/")
async def root():
    return {"message": "Flight price prediction api"}

@app.get("/ping", summary='Health check')
def healthcheck():
    return {"message": "Health check successful!"}

class FlightData(BaseModel):
    airline: str = "AirAsia"
    from_: str = "Mumbai"
    to_: str = "Delhi"
    class_: str = "economy"
    ch_code: str = "SG"
    dep_time: str = "06:20"
    arr_time: str = "08:40"
    date: str = (date.today() + timedelta(days=4)).strftime("%d-%m-%Y")
    time_taken: str = "02h 20m"
    stop: str = "non-stop"
    
@app.post("/predict")
def predict_default(data: FlightData):
    try:
        custom_data = CustomData(
            airline=data.airline,
            from_=data.from_,
            to_=data.to_,
            class_=data.class_,
            ch_code=data.ch_code,
            dep_time=data.dep_time,
            arr_time=data.arr_time,
            date=data.date,
            time_taken=data.time_taken,
            stop=data.stop
        )

        # Convert to DataFrame
        features_df = custom_data.get_data_as_dataframe()

        # Initialize the prediction pipeline and make predictions
        pipeline = PredictPipeline()
        prediction = pipeline.predict(features_df)
        
        # Compute global feature importance
        global_importance = pipeline.get_global_feature_importance()
        global_importance_dict = {col: imp for col, imp in zip(pipeline.preprocessor.get_feature_names_out(), global_importance)}
        global_importance_dict = dict(sorted(global_importance_dict.items(), reverse=True, key=lambda item: item[1]))

        # Compute instance-specific feature importance using SHAP
        shap_values = pipeline.get_instance_feature_importance(features_df)
        instance_importance_dict = {col: imp for col, imp in zip(features_df.columns, shap_values[0])}
        instance_importance_dict = dict(sorted(instance_importance_dict.items(), reverse=True, key=lambda item: item[1]))

        # Return the prediction result
        return {
            "flight_price_prediction": round(prediction[0],2),
            "instance_feature_importance": instance_importance_dict,
            "global_feature_importance": global_importance_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))