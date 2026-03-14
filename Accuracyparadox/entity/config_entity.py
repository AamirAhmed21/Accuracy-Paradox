from datetime import datetime
import sys 
import os 
from dataclasses import dataclass, field
from Accuracyparadox.constant.trainingpipeline import *

# ─────────────────────────────────────────
# 1. Training Pipeline Config
# ─────────────────────────────────────────

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = field(default=PIPELINE_NAME)
    artifact_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), ARTIFACT_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))

# ─────────────────────────────────────────
# 2. Data Ingestion Config
# ─────────────────────────────────────────
@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    data_ingestion_dir: str = field(default="")
    raw_data_path: str = field(default="")
    train_data_path: str = field(default="")
    test_data_path: str = field(default="")
    test_size: float = field(default=TEST_SIZE)
    random_state: int = field(default=RANDOM_STATE)
    
    def __post_init__(self):
        self.data_ingestion_dir = os.path.join(self.training_pipeline_config.artifact_dir,  DATA_INGESTION_DIR_NAME)
        self.raw_data_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_RAW_DATA_DIR, FILE_NAME)
        self.train_data_path = os.path.join(self.data_ingestion_dir,   DATA_INGESTION_INGESTED_DIR,
            TRAIN_FILE_NAME)
        self.test_data_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_TEST_DATA_DIR, TEST_FILE_NAME)
# ─────────────────────────────────────────
# 2. Data Validation Config
# ─────────────────────────────────────────
@dataclass
class DataValidationConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    data_validation_dir: str = field(default="")
    validated_data_dir: str = field(default="")
    
    def __post_init__(self):
        self.data_validation_dir = os.path.join(self.training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
        self.validation_report_file_path = os.path.join(
            self.data_validation_dir,
            VALIDATION_REPORT_FILE_NAME
        )
# ─────────────────────────────────────────
# 3. Data Transformation Config
# ─────────────────────────────────────────
@dataclass
class DataTransformationConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    data_transformation_dir: str = field(default="")
    transformed_train_data_path: str = field(default="")
    transformed_test_data_path: str = field(default="")
    preprocessor_object_path: str = field(default="")
    
    def __post_init__(self):
        self.data_transformation_dir = os.path.join(self.training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_data_path = os.path.join(
            self.data_transformation_dir,
            TRANSFORMED_DATA_DIR_NAME,
            TRANSFORMED_TRAIN_FILE_NAME,
        )
        self.transformed_test_data_path = os.path.join(
            self.data_transformation_dir,
            TRANSFORMED_DATA_DIR_NAME,
            TRANSFORMED_TEST_FILE_NAME,
        )
        self.preprocessor_object_path = os.path.join(
            self.data_transformation_dir,
            TRANSFORMED_DATA_DIR_NAME,
            PREPROCESSOR_OBJECT_FILE_NAME,
        )
# ─────────────────────────────────────────
# 4. Model Config
# ───────────────────────────────────────── 
@dataclass
class ModelTrainerConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    model_trainer_dir: str = field(default="")
    trained_model_file_path: str = field(default="")
    model_metrics_file_path: str = field(default="")
    expected_score: float = field(default=MODEL_TRAINER_EXPECTED_SCORE)
    mlflow_experiment_name: str = field(default=MLFLOW_EXPERIMENT_NAME)
    bentoml_model_name: str = field(default=BENTOML_MODEL_NAME)
    
    def __post_init__(self):
        self.model_trainer_dir = os.path.join(self.training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path = os.path.join(self.model_trainer_dir, MODEL_OBJECT_FILE_NAME)
        self.model_metrics_file_path = os.path.join(self.model_trainer_dir, MODEL_METRICS_FILE_NAME)
