from __future__ import annotations

import psycopg

from app.models.dataset import Dataset


def save_dataset(conn: psycopg.Connection, **kwargs) -> Dataset:
    row = conn.execute(
        """
        INSERT INTO datasets
            (session_id, filename, file_path, file_size, file_hash,
             row_count, column_map, detection_mode)
        VALUES
            (%(session_id)s, %(filename)s, %(file_path)s, %(file_size)s, %(file_hash)s,
             %(row_count)s, %(column_map)s, %(detection_mode)s)
        RETURNING *
        """,
        kwargs,
    ).fetchone()
    conn.commit()
    return Dataset(**row)
