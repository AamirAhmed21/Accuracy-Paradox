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