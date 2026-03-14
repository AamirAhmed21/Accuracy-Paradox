import os 
from datetime import datetime

# ─────────────────────────────────────────
# General
# ─────────────────────────────────────────
PIPELINE_NAME: str = "accuracy_paradox"
ARTIFACT_DIR: str = "artifacts"

# ─────────────────────────────────────────
# Data Ingestion
# ─────────────────────────────────────────
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR: str = "raw_data"
DATA_INGESTION_INGESTED_DIR: str = "ingested_data"
DATA_INGESTION_TEST_DATA_DIR: str = "test_data"
FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ─────────────────────────────────────────
# Data Validation createing the variable for data validation
# ─────────────────────────────────────────
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_SUBDIR: str = "validated_data"
VALIDATION_REPORT_FILE_NAME: str = "validation_report.json"


# ─────────────────────────────────────────
# Data Transformation createing the variable for data validation
# ─────────────────────────────────────────

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
TRANSFORMED_DATA_DIR_NAME: str = "transformed_data"
TRANSFORMED_TRAIN_FILE_NAME: str = "train.npy"
TRANSFORMED_TEST_FILE_NAME: str = "test.npy"
PREPROCESSOR_OBJECT_FILE_NAME: str = "preprocessor.pkl"

TARGET_COLUMN: str = "target"

# ─────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────
MODEL_TRAINER_DIR_NAME: str = 'model_trainer'
MODEL_OBJECT_FILE_NAME: str = 'model.pkl'
MODEL_METRICS_FILE_NAME: str = 'metrics.json'
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MLFLOW_EXPERIMENT_NAME: str = 'AccuracyParadox_Experiment'
BENTOML_MODEL_NAME: str = "accuracy_paradox_model"
