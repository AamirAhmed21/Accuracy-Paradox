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
TRANSFORMED_DIR_NAME: str = "transformed_data"
PREPROCESSOR_DIR_NAME: str = "preprocessor"
PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"
TRANSFORMED_TRAIN_FILE_NAME: str = "transformed_train.csv"
TRANSFORMED_TEST_FILE_NAME: str = "transformed_test.csv"

TARGET_COLUMN: str = "target"
