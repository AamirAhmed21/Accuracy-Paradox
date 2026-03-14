from dataclasses import dataclass
from typing import List, Dict, Any


# ─────────────────────────────────────────
# 1) Data Ingestion Output
# ─────────────────────────────────────────
@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str
    raw_data_path: str
    
# ─────────────────────────────────────────
# 2) Data Validation Output
# ─────────────────────────────────────────
    
@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_file_path: str
# ─────────────────────────────────────────
# 3) Data Transformation Output
# ───────────────────────────────────────── 
@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_file_path: str
# ─────────────────────────────────────────
# 4) Model Training Output
# ───────────────────────────────────────── 
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    model_metrics_file_path: str
    best_model_name: str
    best_model_f1: float
    mlflow_run_id: str
    bentoml_model_tag: str