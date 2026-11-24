"""Ledger write API (docs/DATA_ORCHARD_V4.md section 5; logging policy is a later to-do)."""

from __future__ import annotations

import json
import uuid
from typing import Any

import duckdb

from orchardlogic.treedb.documents import timestamp_param
from orchardlogic.treedb.fixtures import canonical_json
from orchardlogic.treedb.tx import transaction

_FIELDS = ("session_id", "actor", "tree_id", "node_key", "item_id", "cohort_id")


def append(con: duckdb.DuckDBPyConnection, action: str, **fields: Any) -> uuid.UUID:
    """Append one event. ``fields``: session_id, actor, tree_id, node_key, item_id, cohort_id, payload."""
    if not action:
        raise ValueError("action must be non-empty")
    unknown = set(fields) - set(_FIELDS) - {"payload"}
    if unknown:
        raise TypeError(f"unknown ledger field(s): {sorted(unknown)}")
    event_id = uuid.uuid4()
    payload = fields.get("payload")
    tree_id = fields.get("tree_id")
    with transaction(con):
        con.execute(
            "INSERT INTO treedb.ledger (event_id, action, session_id, actor, tree_id, node_key, item_id, "
            "cohort_id, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [str(event_id), action, fields.get("session_id"), fields.get("actor"),
             None if tree_id is None else str(tree_id), fields.get("node_key"), fields.get("item_id"),
             fields.get("cohort_id"), None if payload is None else canonical_json(payload)])
    return event_id


def events(con: duckdb.DuckDBPyConnection, *, action: str | None = None, since: Any = None, until: Any = None,
           limit: int | None = None, **filters: Any) -> list[dict[str, Any]]:
    """Events oldest first. Equality filters on session_id, actor, tree_id, node_key, item_id, cohort_id."""
    unknown = set(filters) - set(_FIELDS)
    if unknown:
        raise TypeError(f"unknown ledger filter(s): {sorted(unknown)}")
    where, params = [], []
    if action is not None:
        where.append("action = ?")
        params.append(action)
    for k, v in filters.items():
        where.append(f"{k} = ?")
        params.append(str(v) if k == "tree_id" else v)
    if since is not None:
        where.append("occurred_at >= CAST(? AS TIMESTAMP)")
        params.append(timestamp_param(since))
    if until is not None:
        where.append("occurred_at < CAST(? AS TIMESTAMP)")
        params.append(timestamp_param(until))
    cols = ("event_id", "occurred_at", "session_id", "actor", "action", "tree_id", "node_key", "item_id",
            "cohort_id", "payload")
    sql = (f"SELECT {', '.join(cols)} FROM treedb.ledger" + (" WHERE " + " AND ".join(where) if where else "")
           + " ORDER BY occurred_at, event_id" + (f" LIMIT {int(limit)}" if limit is not None else ""))
    out = []
    for row in con.execute(sql, params).fetchall():
        rec = dict(zip(cols, row))
        rec["payload"] = None if rec["payload"] is None else json.loads(rec["payload"])
        out.append(rec)
    return out


__all__ = ["append", "events"]
