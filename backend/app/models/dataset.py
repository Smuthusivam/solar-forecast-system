from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


# Represents a dataset object with metadata about uploaded files
@dataclass
class Dataset:
    dataset_id:     int
    session_id:     str
    filename:       str
    file_path:      str
    file_size:      int
    file_hash:      str
    row_count:      int
    column_map:     str
    detection_mode: str
    created_at:     datetime
