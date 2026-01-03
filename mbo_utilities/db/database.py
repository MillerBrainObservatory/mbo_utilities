"""
database operations for mbo_db.

uses sqlite for persistent storage at ~/mbo/mbo_db.sqlite
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from mbo_utilities import log
from mbo_utilities.db.models import Dataset, DatasetStatus, SCHEMA

logger = log.get("db.database")

# default database path
_DB_PATH: Path | None = None


def get_db_path() -> Path:
    """Get the database file path (~/mbo/mbo_db.sqlite)."""
    global _DB_PATH
    if _DB_PATH is None:
        from mbo_utilities import get_mbo_dirs
        dirs = get_mbo_dirs()
        _DB_PATH = Path(dirs["base"]) / "mbo_db.sqlite"
    return _DB_PATH


def set_db_path(path: Path) -> None:
    """Set a custom database path (for testing)."""
    global _DB_PATH
    _DB_PATH = Path(path)


@contextmanager
def get_connection():
    """Get a database connection."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the database schema."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    logger.info(f"initialized database at {get_db_path()}")


def _row_to_dataset(row: sqlite3.Row) -> Dataset:
    """Convert a database row to a Dataset object."""
    return Dataset(
        id=row["id"],
        path=row["path"],
        name=row["name"],
        size_bytes=row["size_bytes"] or 0,
        modified_at=row["modified_at"],
        scanned_at=row["scanned_at"],
        pipeline=row["pipeline"] or "",
        category=row["category"] or "",
        status=DatasetStatus(row["status"]) if row["status"] else DatasetStatus.UNKNOWN,
        num_frames=row["num_frames"],
        num_zplanes=row["num_zplanes"],
        num_rois=row["num_rois"],
        shape=row["shape"] or "",
        dtype=row["dtype"] or "",
        dx=row["dx"],
        dy=row["dy"],
        dz=row["dz"],
        fs=row["fs"],
        tags=row["tags"] or "",
        notes=row["notes"] or "",
        parent_id=row["parent_id"],
    )


def upsert_dataset(dataset: Dataset) -> int:
    """Insert or update a dataset, returns the id."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO datasets (
                path, name, size_bytes, modified_at, scanned_at,
                pipeline, category, status,
                num_frames, num_zplanes, num_rois, shape, dtype,
                dx, dy, dz, fs, tags, notes, parent_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                name = excluded.name,
                size_bytes = excluded.size_bytes,
                modified_at = excluded.modified_at,
                scanned_at = excluded.scanned_at,
                pipeline = excluded.pipeline,
                category = excluded.category,
                status = excluded.status,
                num_frames = excluded.num_frames,
                num_zplanes = excluded.num_zplanes,
                num_rois = excluded.num_rois,
                shape = excluded.shape,
                dtype = excluded.dtype,
                dx = excluded.dx,
                dy = excluded.dy,
                dz = excluded.dz,
                fs = excluded.fs
            """,
            (
                dataset.path,
                dataset.name,
                dataset.size_bytes,
                dataset.modified_at.isoformat() if dataset.modified_at else None,
                dataset.scanned_at.isoformat() if dataset.scanned_at else None,
                dataset.pipeline,
                dataset.category,
                dataset.status.value,
                dataset.num_frames,
                dataset.num_zplanes,
                dataset.num_rois,
                dataset.shape,
                dataset.dtype,
                dataset.dx,
                dataset.dy,
                dataset.dz,
                dataset.fs,
                dataset.tags,
                dataset.notes,
                dataset.parent_id,
            ),
        )
        # get the id
        if cursor.lastrowid:
            return cursor.lastrowid
        # if updated, get existing id
        row = conn.execute(
            "SELECT id FROM datasets WHERE path = ?", (dataset.path,)
        ).fetchone()
        return row["id"] if row else 0


def scan_directory(
    path: str | Path,
    recursive: bool = True,
    progress_callback=None,
) -> int:
    """
    Scan a directory and add datasets to the database.

    Parameters
    ----------
    path : str or Path
        directory to scan
    recursive : bool
        scan subdirectories
    progress_callback : callable, optional
        callback(current, total, path) for progress

    Returns
    -------
    int
        number of datasets added/updated
    """
    from mbo_utilities.db.scanner import scan_for_datasets

    init_db()

    path = Path(path).resolve()
    count = 0

    for dataset in scan_for_datasets(path, recursive, progress_callback):
        upsert_dataset(dataset)
        count += 1
        logger.debug(f"indexed: {dataset.path}")

    logger.info(f"scanned {path}: {count} datasets indexed")
    return count


def get_datasets(
    pipeline: str | None = None,
    status: DatasetStatus | None = None,
    tag: str | None = None,
    search: str | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> list[Dataset]:
    """
    Get datasets from the database with optional filters.

    Parameters
    ----------
    pipeline : str, optional
        filter by pipeline name
    status : DatasetStatus, optional
        filter by status
    tag : str, optional
        filter by tag (partial match)
    search : str, optional
        search in path/name
    limit : int
        max results
    offset : int
        offset for pagination

    Returns
    -------
    list[Dataset]
        matching datasets
    """
    init_db()

    query = "SELECT * FROM datasets WHERE 1=1"
    params = []

    if pipeline:
        query += " AND pipeline = ?"
        params.append(pipeline)

    if status:
        query += " AND status = ?"
        params.append(status.value)

    if tag:
        query += " AND tags LIKE ?"
        params.append(f"%{tag}%")

    if search:
        query += " AND (path LIKE ? OR name LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])

    query += " ORDER BY scanned_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [_row_to_dataset(row) for row in rows]


def get_dataset(dataset_id: int) -> Dataset | None:
    """Get a single dataset by id."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        return _row_to_dataset(row) if row else None


def get_dataset_by_path(path: str | Path) -> Dataset | None:
    """Get a dataset by path."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE path = ?", (str(path),)
        ).fetchone()
        return _row_to_dataset(row) if row else None


def delete_dataset(dataset_id: int) -> bool:
    """Delete a dataset from the database."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        return cursor.rowcount > 0


def link_datasets(
    source_id: int,
    target_id: int,
    link_type: str = "processed_from",
) -> int:
    """
    Create a link between datasets.

    Parameters
    ----------
    source_id : int
        parent dataset id
    target_id : int
        child dataset id
    link_type : str
        type of link (e.g. "processed_from", "registered_from")

    Returns
    -------
    int
        link id
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO dataset_links (source_id, target_id, link_type, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, target_id, link_type, datetime.now().isoformat()),
        )
        return cursor.lastrowid or 0


def get_children(dataset_id: int) -> list[Dataset]:
    """Get child datasets (processed from this dataset)."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT d.* FROM datasets d
            JOIN dataset_links l ON d.id = l.target_id
            WHERE l.source_id = ?
            """,
            (dataset_id,),
        ).fetchall()
        return [_row_to_dataset(row) for row in rows]


def get_parents(dataset_id: int) -> list[Dataset]:
    """Get parent datasets (this was processed from)."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT d.* FROM datasets d
            JOIN dataset_links l ON d.id = l.source_id
            WHERE l.target_id = ?
            """,
            (dataset_id,),
        ).fetchall()
        return [_row_to_dataset(row) for row in rows]


def add_tag(dataset_id: int, tag: str) -> None:
    """Add a tag to a dataset."""
    dataset = get_dataset(dataset_id)
    if dataset:
        dataset.add_tag(tag)
        with get_connection() as conn:
            conn.execute(
                "UPDATE datasets SET tags = ? WHERE id = ?",
                (dataset.tags, dataset_id),
            )


def remove_tag(dataset_id: int, tag: str) -> None:
    """Remove a tag from a dataset."""
    dataset = get_dataset(dataset_id)
    if dataset:
        dataset.remove_tag(tag)
        with get_connection() as conn:
            conn.execute(
                "UPDATE datasets SET tags = ? WHERE id = ?",
                (dataset.tags, dataset_id),
            )


def update_notes(dataset_id: int, notes: str) -> None:
    """Update notes for a dataset."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE datasets SET notes = ? WHERE id = ?",
            (notes, dataset_id),
        )


def get_stats() -> dict:
    """Get database statistics."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        by_pipeline = dict(
            conn.execute(
                "SELECT pipeline, COUNT(*) FROM datasets GROUP BY pipeline"
            ).fetchall()
        )
        by_status = dict(
            conn.execute(
                "SELECT status, COUNT(*) FROM datasets GROUP BY status"
            ).fetchall()
        )
        total_size = conn.execute(
            "SELECT SUM(size_bytes) FROM datasets"
        ).fetchone()[0] or 0

        return {
            "total_datasets": total,
            "by_pipeline": by_pipeline,
            "by_status": by_status,
            "total_size_bytes": total_size,
        }
