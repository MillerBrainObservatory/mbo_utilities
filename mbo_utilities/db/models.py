"""
database models for mbo_db.

uses sqlite with dataclasses for simplicity and portability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class DatasetStatus(str, Enum):
    """processing status of a dataset."""
    RAW = "raw"
    REGISTERED = "registered"
    SEGMENTED = "segmented"
    COMPLETE = "complete"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class Dataset:
    """represents a discovered dataset in the database."""

    # primary key (auto-generated)
    id: Optional[int] = None

    # file system info
    path: str = ""
    name: str = ""
    size_bytes: int = 0
    modified_at: Optional[datetime] = None
    scanned_at: Optional[datetime] = None

    # pipeline info (from pipeline_registry)
    pipeline: str = ""  # e.g. "suite2p", "zarr", "tiff"
    category: str = ""  # e.g. "reader", "segmentation"

    # dataset structure
    status: DatasetStatus = DatasetStatus.UNKNOWN
    num_frames: Optional[int] = None
    num_zplanes: Optional[int] = None
    num_rois: Optional[int] = None
    shape: str = ""  # stored as string "(T, Z, Y, X)"
    dtype: str = ""

    # voxel size
    dx: Optional[float] = None
    dy: Optional[float] = None
    dz: Optional[float] = None
    fs: Optional[float] = None  # frame rate

    # user annotations
    tags: str = ""  # comma-separated
    notes: str = ""

    # parent dataset (for linking raw -> processed)
    parent_id: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = DatasetStatus(self.status)
        if isinstance(self.modified_at, str):
            self.modified_at = datetime.fromisoformat(self.modified_at)
        if isinstance(self.scanned_at, str):
            self.scanned_at = datetime.fromisoformat(self.scanned_at)

    @property
    def path_obj(self) -> Path:
        return Path(self.path)

    @property
    def tag_list(self) -> list[str]:
        if not self.tags:
            return []
        return [t.strip() for t in self.tags.split(",") if t.strip()]

    def add_tag(self, tag: str) -> None:
        tags = self.tag_list
        if tag not in tags:
            tags.append(tag)
            self.tags = ",".join(tags)

    def remove_tag(self, tag: str) -> None:
        tags = self.tag_list
        if tag in tags:
            tags.remove(tag)
            self.tags = ",".join(tags)

    @property
    def size_human(self) -> str:
        """human-readable file size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"

    def to_dict(self) -> dict:
        """convert to dictionary for display."""
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "pipeline": self.pipeline,
            "status": self.status.value,
            "shape": self.shape,
            "size": self.size_human,
            "modified": self.modified_at.isoformat() if self.modified_at else "",
            "tags": self.tag_list,
        }


@dataclass
class DatasetLink:
    """represents a relationship between datasets (e.g. raw -> processed)."""

    id: Optional[int] = None
    source_id: int = 0  # parent dataset
    target_id: int = 0  # child dataset
    link_type: str = ""  # e.g. "processed_from", "registered_from"
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)


# sql schema for creating tables
SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    size_bytes INTEGER DEFAULT 0,
    modified_at TEXT,
    scanned_at TEXT,
    pipeline TEXT,
    category TEXT,
    status TEXT DEFAULT 'unknown',
    num_frames INTEGER,
    num_zplanes INTEGER,
    num_rois INTEGER,
    shape TEXT,
    dtype TEXT,
    dx REAL,
    dy REAL,
    dz REAL,
    fs REAL,
    tags TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    parent_id INTEGER,
    FOREIGN KEY (parent_id) REFERENCES datasets(id)
);

CREATE TABLE IF NOT EXISTS dataset_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    link_type TEXT,
    created_at TEXT,
    FOREIGN KEY (source_id) REFERENCES datasets(id),
    FOREIGN KEY (target_id) REFERENCES datasets(id),
    UNIQUE(source_id, target_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_datasets_path ON datasets(path);
CREATE INDEX IF NOT EXISTS idx_datasets_pipeline ON datasets(pipeline);
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);
CREATE INDEX IF NOT EXISTS idx_datasets_parent ON datasets(parent_id);
"""
