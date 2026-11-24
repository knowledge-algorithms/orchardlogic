"""Resumable-state checkpoints (docs/DATA_ORCHARD_V4.md section 5.1)."""

from __future__ import annotations

import json
from typing import Any

import duckdb

from orchardlogic.treedb.fixtures import canonical_json
from orchardlogic.treedb.tx import transaction


def put_checkpoint(con: duckdb.DuckDBPyConnection, namespace: str, key: str, payload: Any) -> None:
    with transaction(con):
        con.execute("INSERT OR REPLACE INTO treedb.checkpoint (namespace, key, payload, updated_at) "
                    "VALUES (?, ?, ?, CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP))",
                    [namespace, key, canonical_json(payload)])


def get_checkpoint(con: duckdb.DuckDBPyConnection, namespace: str, key: str) -> Any | None:
    row = con.execute("SELECT payload FROM treedb.checkpoint WHERE namespace = ? AND key = ?",
                      [namespace, key]).fetchone()
    return None if row is None else json.loads(row[0])


def list_checkpoints(con: duckdb.DuckDBPyConnection, namespace: str) -> dict[str, Any]:
    """``{key: payload}`` of a namespace, sorted by key."""
    rows = con.execute("SELECT key, payload FROM treedb.checkpoint WHERE namespace = ? ORDER BY key",
                       [namespace]).fetchall()
    return {k: json.loads(p) for k, p in rows}


def delete_checkpoint(con: duckdb.DuckDBPyConnection, namespace: str, key: str) -> None:
    with transaction(con):
        con.execute("DELETE FROM treedb.checkpoint WHERE namespace = ? AND key = ?", [namespace, key])


__all__ = ["delete_checkpoint", "get_checkpoint", "list_checkpoints", "put_checkpoint"]
