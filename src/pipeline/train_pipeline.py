import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass
    
    def run_pipeline(self):
        try:
            # Data Ingestion
            logging.info("Starting the training pipeline")
            
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            
            # Model Training
            model_trainer = ModelTrainer()
            accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info("Training pipeline completed successfully")
            return accuracy
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # This allows running the pipeline directly
    pipeline = TrainPipeline()
    result = pipeline.run_pipeline()
    print(f"Pipeline completed with accuracy: {result:.4f}")