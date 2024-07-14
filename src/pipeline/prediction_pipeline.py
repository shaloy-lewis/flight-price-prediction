import os
import sys
import pandas as pd
import numpy as np 
import json
from src.exception.exception import customexception
from src.logger.logging import logging
from datetime import date 
import shap
from sklearn.tree import DecisionTreeRegressor

from src.utils.utils import load_object
from src.utils.constants import OUTLIER_COL 
from src.utils.utils import preprocess_data

class PredictPipeline:
    def __init__(self):
        logging.info("Initializing the prediction pipeline")
        self.preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
        self.model_path=os.path.join("artifacts","model.pkl")
        self.outlier_threshold_path=os.path.join("artifacts","outlier_threshold.json")

        self.preprocessor=load_object(self.preprocessor_path)
        self.model=load_object(self.model_path)
        with open(self.outlier_threshold_path,'r') as file:
            self.outlier_threshold=json.load(file)

    def predict(self,features):
        try:    
            low_perc = pd.Series(self.outlier_threshold['low_perc'])
            high_perc = pd.Series(self.outlier_threshold['high_perc'])
            
            features=preprocess_data(features,loginday=date.today().strftime("%d-%m-%Y"))
            features[OUTLIER_COL] = features[OUTLIER_COL].clip(lower=low_perc, upper=high_perc, axis=1)
            
            scaled_features=self.preprocessor.transform(features)
            rawpred=self.model.predict(scaled_features)
            pred=np.exp(rawpred)
            logging.info('Predictions obtained successfully')
            logging.info(f'Flight price prediction: {pred}')

            return pred

        except Exception as e:
            raise customexception(e,sys)
    
    def get_global_feature_importance(self):
        try:
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            else:
                raise ValueError("Model does not support feature_importances_ attribute.")
        except Exception as e:
            raise customexception(e, sys)

    def get_instance_feature_importance(self, features):
        try:
            scaled_features = self.preprocessor.transform(features)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(scaled_features)
            logging.info('SHAP values computed successfully')
            
            return shap_values
        
        except Exception as e:
            raise customexception(e, sys)

class CustomData:
    def __init__(self,
                 airline: str,
                 from_: str,
                 to_: str,
                 class_: str,
                 ch_code: str,
                 dep_time: str,
                 arr_time: str,
                 date: str,
                 time_taken: str,
                 stop: str):
        self.airline = airline
        self.from_ = from_
        self.to_ = to_
        self.class_ = class_
        self.ch_code = ch_code
        self.dep_time = dep_time
        self.arr_time = arr_time
        self.date = date
        self.time_taken = time_taken
        self.stop = stop 
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'airline': [self.airline],
                'from': [self.from_],
                'to': [self.to_],
                'class': [self.class_],
                'ch_code': [self.ch_code],
                'dep_time': [self.dep_time],
                'arr_time': [self.arr_time],
                'date': [self.date],
                'time_taken': [self.time_taken],
                'stop': [self.stop]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Data received for inference prediction')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise customexception(e,sys)