#!/usr/bin/env python3
"""Deduplicate Qdrant local storage by payload signature.

This script works directly on Qdrant's local SQLite storage. It keeps one
point per unique payload signature and removes the others. For duplicate
groups that contain both legacy integer IDs and newer UUID IDs, it prefers
the UUID-backed point.
"""

from __future__ import annotations

import base64
import os
import pickle
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config import COLLECTION_NAME, QDRANT_PATH


STORAGE_PATH = Path(QDRANT_PATH) / "collection" / COLLECTION_NAME / "storage.sqlite"


@dataclass
class PointRow:
    row_id: str
    point_id: object
    payload: Dict

    @property
    def is_uuid(self) -> bool:
        return isinstance(self.point_id, str)


def load_rows(conn: sqlite3.Connection) -> List[PointRow]:
    rows: List[PointRow] = []
    cur = conn.execute("select id, point from points")
    for row_id, blob in cur:
        point = pickle.loads(blob)
        payload = getattr(point, "payload", {}) or {}
        rows.append(PointRow(row_id=row_id, point_id=point.id, payload=payload))
    return rows


def signature(row: PointRow) -> Tuple:
    p = row.payload
    return (
        p.get("book_id"),
        p.get("filename"),
        p.get("page_start"),
        p.get("page_end"),
        p.get("chunk_idx"),
        p.get("text"),
    )


def choose_keep(rows: List[PointRow]) -> PointRow:
    uuid_rows = [r for r in rows if r.is_uuid]
    if uuid_rows:
        return uuid_rows[0]
    return rows[0]


def delete_rows(conn: sqlite3.Connection, row_ids: List[str]) -> None:
    batch_size = 200
    for i in range(0, len(row_ids), batch_size):
        batch = row_ids[i : i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        conn.execute(f"delete from points where id in ({placeholders})", batch)


def main() -> None:
    if not STORAGE_PATH.exists():
        raise SystemExit(f"Storage file not found: {STORAGE_PATH}")

    backup_path = STORAGE_PATH.with_suffix(".sqlite.bak")
    if not backup_path.exists():
        shutil.copy2(STORAGE_PATH, backup_path)
        print(f"Backup created: {backup_path}")

    conn = sqlite3.connect(STORAGE_PATH)
    conn.execute("begin immediate")

    rows = load_rows(conn)
    groups: Dict[Tuple, List[PointRow]] = defaultdict(list)
    for row in rows:
        groups[signature(row)].append(row)

    duplicate_groups = {sig: members for sig, members in groups.items() if len(members) > 1}
    delete_ids: List[str] = []
    for members in duplicate_groups.values():
        keep = choose_keep(members)
        for row in members:
            if row.row_id != keep.row_id:
                delete_ids.append(row.row_id)

    delete_rows(conn, delete_ids)
    conn.commit()
    conn.close()

    print(f"Total rows before: {len(rows)}")
    print(f"Unique signatures: {len(groups)}")
    print(f"Duplicate groups : {len(duplicate_groups)}")
    print(f"Deleted rows     : {len(delete_ids)}")
    print(f"Remaining rows   : {len(rows) - len(delete_ids)}")


if __name__ == "__main__":
    main()
