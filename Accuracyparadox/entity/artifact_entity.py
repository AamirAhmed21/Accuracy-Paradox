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
    
@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_file_path: str
    
@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_file_path: str