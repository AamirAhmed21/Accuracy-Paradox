import sys
from venv import logger

from Accuracyparadox.exception.exception import CustomException
from Accuracyparadox.logging import logging

from Accuracyparadox.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from Accuracyparadox.Components.synthetic_data_generator import SyntheticDataGenerator
from Accuracyparadox.Components.data_ingestion import DataIngestion
from Accuracyparadox.Components.data_validation import DataValidation
from Accuracyparadox.Components.data_tranformation import DataTransformation
from Accuracyparadox.Components.Model import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self) -> tuple:
        try:
            logging.info("─────────────────────────────────────────")
            logging.info("Starting Training Pipeline")
            logging.info("─────────────────────────────────────────")
            
            ## Step 1: Generate synthetic data
            synthetic_data_generator = SyntheticDataGenerator()
            raw_data_path = synthetic_data_generator.generate_data()
            logging.info(f"Synthetic data generated at: {raw_data_path}")
            
            ## Step 2: Data Ingestion
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            
            ## override raw data path in config with generated data path
            data_ingestion_config.raw_data_path = raw_data_path
            ## RUN data ingestion
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logging.info(f"DataIngestionArtifact: {data_ingestion_artifact}")
            logging.info("========== Training Pipeline Completed ==========")
            
            # 2 Data validation
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            
            data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"DataValidationArtifact: {data_validation_artifact}")
            
            # 3 Data Transformation
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"DataTransformationArtifact: {data_transformation_artifact}")
            
            # 4 Model Training
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            logging.info("========== Training Pipeline Completed ==========")
            return (
                data_ingestion_artifact,
                data_validation_artifact,
                data_transformation_artifact,
                model_trainer_artifact
            )
            
        except Exception as e:
            raise CustomException(e, sys) from e
