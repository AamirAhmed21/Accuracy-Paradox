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