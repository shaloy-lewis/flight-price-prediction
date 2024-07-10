import os
import sys
import pandas as pd
import json
from src.exception.exception import customexception
from src.logger.logging import logging
import shap
from sklearn.ensemble import RandomForestRegressor

from src.utils.utils import load_object
from src.utils.constants import OUTLIER_COL 

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
            
            features[OUTLIER_COL] = features[OUTLIER_COL].clip(lower=low_perc, upper=high_perc, axis=1)
            features['BILL_AMT_AVG_6M'] = features[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].mean(axis=1).values
            features.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis=1,inplace=True)
        
            # features=preprocess_data(features)
            
            scaled_features=self.preprocessor.transform(features)
            pred=self.model.predict_proba(scaled_features)
            logging.info('Predictions obtained successfully')
            logging.info(f'Probability of default: {pred}')

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

            pool = Pool(scaled_features)
            shap_values = self.model.get_feature_importance(data=pool, fstr_type='ShapValues')
            logging.info('SHAP values computed successfully')

            # SHAP values array has an extra column for base value; we exclude it
            shap_values = shap_values[:, :-1]

            return shap_values
        
        except Exception as e:
            raise customexception(e, sys)

class CustomData:
    def __init__(self,
                 LIMIT_BAL: float,
                 AGE: float,
                 BILL_AMT1: float,
                 BILL_AMT2: float,
                 BILL_AMT3: float,
                 BILL_AMT4: float,
                 BILL_AMT5: float,
                 BILL_AMT6: float,
                 PAY_AMT1: float,
                 PAY_AMT2: float,
                 PAY_AMT3: float,
                 PAY_AMT4: float,
                 PAY_AMT5: float,
                 PAY_AMT6: float,
                 EDUCATION: str,
                 MARRIAGE: str,
                 SEX: str,
                 PAY_0: str,
                 PAY_2: str,
                 PAY_3: str,
                 PAY_4: str,
                 PAY_5: str,
                 PAY_6: str):
        self.LIMIT_BAL = LIMIT_BAL
        self.AGE = AGE
        self.BILL_AMT1 = BILL_AMT1
        self.BILL_AMT2 = BILL_AMT2
        self.BILL_AMT3 = BILL_AMT3
        self.BILL_AMT4 = BILL_AMT4
        self.BILL_AMT5 = BILL_AMT5
        self.BILL_AMT6 = BILL_AMT6
        self.PAY_AMT1 = PAY_AMT1
        self.PAY_AMT2 = PAY_AMT2
        self.PAY_AMT3 = PAY_AMT3
        self.PAY_AMT4 = PAY_AMT4
        self.PAY_AMT5 = PAY_AMT5
        self.PAY_AMT6 = PAY_AMT6
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.SEX = SEX
        self.PAY_0 = PAY_0
        self.PAY_2 = PAY_2
        self.PAY_3 = PAY_3
        self.PAY_4 = PAY_4
        self.PAY_5 = PAY_5
        self.PAY_6 = PAY_6
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'LIMIT_BAL': [self.LIMIT_BAL],
                'AGE': [self.AGE],
                'BILL_AMT1': [self.BILL_AMT1],
                'BILL_AMT2': [self.BILL_AMT2],
                'BILL_AMT3': [self.BILL_AMT3],
                'BILL_AMT4': [self.BILL_AMT4],
                'BILL_AMT5': [self.BILL_AMT5],
                'BILL_AMT6': [self.BILL_AMT6],
                'PAY_AMT1': [self.PAY_AMT1],
                'PAY_AMT2': [self.PAY_AMT2],
                'PAY_AMT3': [self.PAY_AMT3],
                'PAY_AMT4': [self.PAY_AMT4],
                'PAY_AMT5': [self.PAY_AMT5],
                'PAY_AMT6': [self.PAY_AMT6],
                'EDUCATION': [self.EDUCATION],
                'MARRIAGE': [self.MARRIAGE],
                'SEX': [self.SEX],
                'PAY_0': [self.PAY_0],
                'PAY_2': [self.PAY_2],
                'PAY_3': [self.PAY_3],
                'PAY_4': [self.PAY_4],
                'PAY_5': [self.PAY_5],
                'PAY_6': [self.PAY_6]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Data received for inference prediction')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise customexception(e,sys)