from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from src.utils.utils import save_object
# from skopt import BayesSearchCV
# from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# from src.utils.constants import PARAM_SPACE

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,train_target):
        try:
            logging.info("Initiating model training")
            model=DecisionTreeRegressor()
            # cv_method=RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
            
            # search = RandomizedSearchCV(
            #     estimator=model,
            #     param_distributions=PARAM_SPACE,
            #     n_iter=50, 
            #     cv=3,
            #     random_state=42,
            #     n_jobs=1
            # )
            
            logging.info("Model training in progress")
            model.fit(train_array, train_target)
            
            logging.info("Model training done")
            # logging.info(f"Best params: {search.best_params_}")
            # best_model = search.best_estimator_
            
            logging.info("Saving model")
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=model 
            )
            logging.info("Model saved successfully")

        except Exception as e:
            logging.info('Exception occured in model training')
            raise customexception(e,sys)