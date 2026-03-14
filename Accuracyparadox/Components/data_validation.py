import os 
import pandas as pd
import numpy as np
import sys
import json
from Accuracyparadox.entity.config_entity import DataValidationConfig
from Accuracyparadox.exception.exception import CustomException
from Accuracyparadox.logging.logging import logging
from Accuracyparadox.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact)
from Accuracyparadox.constant.trainingpipeline import TARGET_COLUMN

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
    def validate_file_existence(self) -> bool:
        try:
            train_exists = os.path.exists(self.data_ingestion_artifact.train_file_path)
            test_exists = os.path.exists(self.data_ingestion_artifact.test_file_path)
            return train_exists and test_exists
        except Exception as e:
            raise CustomException(e, sys)
    def validate_dataset(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            
            validation_status = True
            validation_report = {}
            
            if not self.validate_file_existence():
                validation_status = False
                validation_report["file_existence"] = False
            else:
                validation_report["file_existence"] = True
                logging.info("Files exist.")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            validation_report["train_shape"] = list(train_df.shape)
            validation_report["test_shape"] = list(test_df.shape)
            
            validation_report["target_column_exists"] = (
                TARGET_COLUMN in train_df.columns and TARGET_COLUMN in test_df.columns
            )
            if not validation_report["target_column_exists"]:
                validation_status = False
            
            validation_report["train_missing_values"] = train_df.isnull().sum().to_dict()
            validation_report["test_missing_values"] = test_df.isnull().sum().to_dict()
            
            validation_report['test_duplicates'] = int(test_df.duplicated().sum())
            validation_report['train_duplicates'] = int(train_df.duplicated().sum())
            
            if TARGET_COLUMN in train_df.columns:
                validation_report["train_target_distribution"] = (train_df[TARGET_COLUMN].value_counts(normalize=True).to_dict())
                
            if TARGET_COLUMN in test_df.columns:
                validation_report["test_target_distribution"] = (test_df[TARGET_COLUMN].value_counts(normalize=True).to_dict())
                
            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            
            with open(self.data_validation_config.validation_report_file_path, 'w') as f:
                json.dump(validation_report, f, indent=4)
                
            logging.info(f"Validation report saved at: {self.data_validation_config.validation_report_file_path}")
            logging.info("Data validation completed")
            
            return DataValidationArtifact(
                validation_status=validation_status,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            return self.validate_dataset()
        except Exception as e:
            raise CustomException(e, sys)