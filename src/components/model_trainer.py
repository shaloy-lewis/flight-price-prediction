from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object
from src.utils.constants import PARAM_GRID

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,train_target):
        try:
            logging.info("Initiating model training")
            cv_method = RepeatedKFold(n_splits=3, random_state=33)
            grid_search = GridSearchCV(estimator=RandomForestRegressor()
                                       , param_grid=PARAM_GRID
                                       , cv=cv_method
                                       , scoring='neg_root_mean_squared_error')
            
            logging.info("Model training in progress")
            grid_search.fit(train_array, train_target.flatten())
            best_params = grid_search.best_params_
            
            logging.info("Model training done")
            logging.info(f"Model best parameters:\n{best_params}")
            best_model = grid_search.best_estimator_
            
            logging.info("Saving model")
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
            logging.info("Model saved successfully")

        except Exception as e:
            logging.info('Exception occured in model training')
            raise customexception(e,sys)