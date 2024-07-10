import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
import json
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.utils.utils import save_object

from src.utils.constants import (CATEGORICAL_FEATURES,
                                 NUMERIC_FEATURES,
                                 OUTLIER_COL,
                                 OUTLIER_CAPPING_LOWER_THRESHOLD,
                                 OUTLIER_CAPPING_UPPER_THRESHOLD)
from src.utils.utils import cap_outliers, preprocess_data, target_preprocess

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    outlier_threshold_path=os.path.join('artifacts','outlier_threshold.json')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info('Data transformation initiated')
            logging.info('Pipeline Initiated')
            
            numeric_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer()),
                    ("scaler",StandardScaler())
                ]
            )
            
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder(sparse_output=False,drop='if_binary'))
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ("numeric_pipeline",numeric_pipeline,NUMERIC_FEATURES),
                    ("catategorical_pipeline",categorical_pipeline,CATEGORICAL_FEATURES),
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in get_data_transformation")
            raise customexception(e,sys)
    
    def initiate_data_transformation(self,X_train_path,X_test_path,y_train_path,y_test_path):
        try:
            X_train=pd.read_csv(X_train_path)
            X_test=pd.read_csv(X_test_path)
            y_train=pd.read_csv(y_train_path)
            y_test=pd.read_csv(y_test_path)
            
            logging.info("Read train and test data for data transformation")
            logging.info(f'Train Dataframe shape : {X_train.shape}')
            logging.info(f'Test Dataframe shape : {X_test.shape}')
            
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            y_train = target_preprocess(y_train)
            y_test = target_preprocess(y_test)
            
            logging.info('Capping outliers in some numeric columns')  
            X_train, low_perc, high_perc = cap_outliers(X_train
                                            , percentile_low=OUTLIER_CAPPING_LOWER_THRESHOLD
                                            , percentile_high=OUTLIER_CAPPING_UPPER_THRESHOLD
                                            , req_columns=OUTLIER_COL)

            # Apply the same cap to test sets
            X_test[OUTLIER_COL] = X_test[OUTLIER_COL].clip(lower=low_perc, upper=high_perc, axis=1)
            
            logging.info("Saving outlier thresholds in JSON format")
            thresholds = {
                'low_perc': low_perc.to_dict(),
                'high_perc': high_perc.to_dict()
            }
            with open(self.data_transformation_config.outlier_threshold_path, 'w') as file:
                json.dump(thresholds, file)
            logging.info("Outlier thresholds saved in JSON format")
                        
            preprocessor = self.get_data_transformation()
            
            logging.info('Applying preprocessing object on training and testing datasets')
            X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
            X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())
            
            logging.info("Data transformattion done")
            logging.info(f"Transformed train shape: {X_train.shape}")
            logging.info(f"Transformed test shape: {X_test.shape}")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessing file saved in pickle format")
            
            return (
                X_train,
                X_test,
                y_train,
                y_test
            )
            
        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise customexception(e,sys)
