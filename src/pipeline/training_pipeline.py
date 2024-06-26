from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

data_ingestion=DataIngestion()
train_data_path,test_data_path,train_target_path,test_target_path=data_ingestion.initiate_data_ingestion()

data_transformation=DataTransformation()
train_arr,test_arr,train_target,test_target=data_transformation.initiate_data_transformation(train_data_path,test_data_path,train_target_path,test_target_path)

model_trainer=ModelTrainer()
model_trainer.initate_model_training(train_arr,train_target)

model_eval=ModelEvaluation()
model_eval.initiate_model_evaluation(train_arr,test_arr,train_target,test_target)