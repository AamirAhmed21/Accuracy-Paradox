import os 
import sys 
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Accuracyparadox.exception.exception import CustomException
from Accuracyparadox.logging.logging import logging
from Accuracyparadox.entity.config_entity import DataTransformationConfig
from Accuracyparadox.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
    def get_data_transformer_object(self, input_df: pd.DataFrame):
        try:
            numerical_columns = input_df.columns.tolist()
            
            numerical_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', numerical_pipeline, numerical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Obtaining training and testing file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("Loading training and testing data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            target_column_name = "target"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object(input_df=input_feature_train_df)

            logging.info("Applying preprocessing object on training and testing dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            transformed_train_dir = os.path.dirname(self.data_transformation_config.transformed_train_data_path)
            transformed_test_dir = os.path.dirname(self.data_transformation_config.transformed_test_data_path)

            os.makedirs(transformed_train_dir, exist_ok=True)
            os.makedirs(transformed_test_dir, exist_ok=True)

            logging.info("Saving transformed training and testing array.")
            np.save(self.data_transformation_config.transformed_train_data_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_data_path, test_arr)
            
            preprocessor_dir = os.path.dirname(self.data_transformation_config.preprocessor_object_path)
            os.makedirs(preprocessor_dir, exist_ok=True)

            with open(self.data_transformation_config.preprocessor_object_path, 'wb') as file_obj:
                pickle.dump(preprocessing_obj, file_obj)
            logging.info("Saved preprocessing object.")
            logging.info("Data transformation completed")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_data_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_data_path,
                preprocessor_object_file_path=self.data_transformation_config.preprocessor_object_path
            )
        except Exception as e:
            raise CustomException(e, sys) from e
            