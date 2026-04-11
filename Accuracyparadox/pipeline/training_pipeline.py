import sys
import os
import pandas as pd

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
            
            # Step 1: Load real Kaggle dataset when available, else fallback to synthetic
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            kaggle_data_path = os.path.join(project_root, "Data", "creditcard", "creditcard.csv")

            if os.path.exists(kaggle_data_path):
                logging.info(f"Found Kaggle dataset at: {kaggle_data_path}")
                real_df = pd.read_csv(kaggle_data_path)

                if "Class" in real_df.columns and "target" not in real_df.columns:
                    real_df = real_df.rename(columns={"Class": "target"})

                if "target" not in real_df.columns:
                    raise CustomException(
                        "Input dataset must contain either 'Class' or 'target' column.",
                        sys,
                    )

                prepared_raw_dir = os.path.join(project_root, "Data", "raw")
                os.makedirs(prepared_raw_dir, exist_ok=True)
                raw_data_path = os.path.join(prepared_raw_dir, "creditcard_prepared.csv")
                real_df.to_csv(raw_data_path, index=False)
                logging.info(f"Prepared real dataset saved to: {raw_data_path}")
                logging.info(f"Prepared dataset shape: {real_df.shape}")
                logging.info(
                    f"Target distribution: {real_df['target'].value_counts(normalize=True).to_dict()}"
                )
            else:
                logging.warning(
                    f"Kaggle dataset not found at {kaggle_data_path}. Falling back to synthetic data."
                )
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
