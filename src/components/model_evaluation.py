import os
import sys
import numpy as np
from src.utils.utils import load_object
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from src.logger.logging import logging
from src.exception.exception import customexception

class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation initiated")
        
    def evaluate_model(self, true, pred):
        true=np.exp(true)
        pred=np.exp(pred)
        
        mse = mean_squared_error(true, pred)
        rmse = mean_squared_error(true, pred, squared=False)
        r2 = r2_score(true, pred)
        mape = mean_absolute_percentage_error(true, pred)
        
        return mse, rmse, r2, mape

    def initiate_model_evaluation(self,train_array,test_array,train_target,test_target):
        try:
            logging.info("Model loading in progress")
            model_path=os.path.join("artifacts","model.pkl")
            best_model=load_object(model_path)

            logging.info("Calculating model predictions")
            y_train_pred=best_model.predict(train_array)
            y_pred=best_model.predict(test_array)

            logging.info("Evaluating model")
            mse_train, rmse_train, r2_train, mape_train=self.evaluate_model(train_target,y_train_pred)
            mse_test, rmse_test, r2_test, mape_test=self.evaluate_model(test_target,y_pred)
            
            logging.info("Model report:")
            logging.info(f"Train MSE: {mse_train}\t Test MSE: {mse_test}")
            logging.info(f"Train RMSE: {rmse_train}\t Test RMSE: {rmse_test}")
            logging.info(f"Train R2: {r2_train}\t Test R2: {r2_test}")
            logging.info(f"Train MAPE: {mape_train}\t Test MAPE: {mape_test}")
            
        except Exception as e:
            logging.info("Exception occured in model evaluation")
            raise customexception(e,sys)
    
    
            