import os 
import sys 
import pandas as pd
from sklearn.model_selection import train_test_split

from Accuracyparadox.logging.logging import logging 
from Accuracyparadox.exception.exception import CustomException
from Accuracyparadox.entity.config_entity import DataIngestionConfig
from Accuracyparadox.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys) from e

    def read_data(self) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from file: {self.data_ingestion_config.raw_data_path}")
            raw_data_path = self.data_ingestion_config.raw_data_path
            df = pd.read_csv(raw_data_path)  
            logging.info(f"Data loaded successfully from: {raw_data_path}")
            logging.info(f"Shape: {df.shape}")
            logging.info(f"Columns: {list(df.columns)}")
            return df

        except Exception as e:
            raise CustomException(e, sys) from e
    def split_data(self, df: pd.DataFrame):
        try:
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.test_size,
                random_state=self.data_ingestion_config.random_state,
                stratify=df['target']) ## keep class ratio in both splits
            logging.info(f"Train set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")
            logging.info(f"Train set class distribution:\n{train_set['target'].value_counts(normalize=True)}")
            logging.info(f"Test set class distribution:\n{test_set['target'].value_counts(normalize=True)}")
            return train_set, test_set
        except Exception as e:
            raise CustomException(e, sys) from e
    def save_data(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> DataIngestionArtifact:
        try:
            logging.info("Saving train and test sets to files")
            
            ## create directories if they don't exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            
            ## save train and test sets
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False)
            
            logging.info(f"Train set saved to: {self.data_ingestion_config.train_data_path}")
            logging.info(f"Test set saved to: {self.data_ingestion_config.test_data_path}")
            
        except Exception as e:
            raise CustomException(e, sys) from e
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("─────────────────────────────────────────")
            logging.info("Starting Data Ingestion")
            logging.info("─────────────────────────────────────────")
            
             # Step 1: read data
            df = self.read_data()
            
            
            # Step 2: split data
            train_df, test_df = self.split_data(df)
            
            # Step 3: save data
            self.save_data(train_df, test_df)
            
            # Step 4: return artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
                raw_data_path=self.data_ingestion_config.raw_data_path
            )

            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            logging.info("Data Ingestion Completed Successfully")
            logging.info("─────────────────────────────────────────")

            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e