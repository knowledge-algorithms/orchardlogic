"""Precomputed cut rows (docs/DATA_ORCHARD_V4.md section 4.3 and 5.1, A6).

``cut_node`` rows are the packed presentation tree. node_key is opaque after import
(docs/NODE_KEY_OPAQUE.md): never re-validate a cut against membership hashes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import duckdb

from orchardlogic.treedb.documents import normalize_timestamp, timestamp_param
from orchardlogic.treedb.fixtures import canonical_json
from orchardlogic.treedb.tx import transaction

CUT_SCHEMA_VERSION = "orchard_cultivar_cut_v1"
_KINDS = ("root", "cluster", "item")


def put_cut(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID, cut_doc: Mapping[str, Any]) -> None:
    """Replace the rows of one cut from an ``orchard_cultivar_cut_v1`` document.

    ``cut_doc["display"]`` (``{node_key: {title, summary, tags}}``) fills the title, summary and
    tags columns. ``revision`` is 0 for a new cut and +1 when an existing cut is replaced.
    """
    if cut_doc.get("schema_version") != CUT_SCHEMA_VERSION:
        raise ValueError(f"unsupported cut schema_version: {cut_doc.get('schema_version')!r}")
    name = cut_doc["name"]
    nodes = list(cut_doc["nodes"])
    display = cut_doc.get("display") or {}
    keys = [n["node_key"] for n in nodes]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate node_key in cut")
    for n in nodes:
        if n["kind"] not in _KINDS:
            raise ValueError(f"bad node kind {n['kind']!r}")
    tid = str(tree_id)
    cols = {c: [] for c in ("node_key", "parent_key", "kind", "depth", "position", "item_id",
                            "descendant_count", "path", "title", "summary", "tags")}
    for n in nodes:
        d = display.get(n["node_key"]) or {}
        cols["node_key"].append(n["node_key"])
        cols["parent_key"].append(n.get("parent_key"))
        cols["kind"].append(n["kind"])
        cols["depth"].append(int(n["depth"]))
        cols["position"].append(int(n["position"]))
        cols["item_id"].append(n.get("item_id"))
        cols["descendant_count"].append(int(n["descendant_count"]))
        cols["path"].append(list(n.get("path") or []))
        cols["title"].append(d.get("title"))
        cols["summary"].append(d.get("summary"))
        cols["tags"].append(None if d.get("tags") is None else list(d["tags"]))
    with transaction(con):
        row = con.execute("SELECT revision FROM treedb.cut WHERE tree_id = ? AND cut_name = ?",
                          [tid, name]).fetchone()
        revision = 0 if row is None else row[0] + 1
        con.execute("DELETE FROM treedb.cut_node WHERE tree_id = ? AND cut_name = ?", [tid, name])
        con.execute("INSERT OR REPLACE INTO treedb.cut (tree_id, cut_name, settings, revision, updated_at) "
                    "VALUES (?, ?, ?, ?, CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP))",
                    [tid, name, canonical_json(cut_doc.get("settings") or {}), revision])
        if nodes:
            con.execute(
                "INSERT INTO treedb.cut_node (tree_id, cut_name, node_key, parent_key, kind, depth, position, "
                "item_id, descendant_count, path, title, summary, tags) "
                "SELECT ?::UUID, ?, unnest(?::VARCHAR[]), unnest(?::VARCHAR[]), unnest(?::VARCHAR[]), "
                "unnest(?::INTEGER[]), unnest(?::INTEGER[]), unnest(?::VARCHAR[]), unnest(?::INTEGER[]), "
                "unnest(?::VARCHAR[][]), unnest(?::VARCHAR[]), unnest(?::VARCHAR[]), unnest(?::VARCHAR[][])",
                [tid, name, cols["node_key"], cols["parent_key"], cols["kind"], cols["depth"], cols["position"],
                 cols["item_id"], cols["descendant_count"], cols["path"], cols["title"], cols["summary"],
                 cols["tags"]])


def cut_revision(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID, cut_name: str) -> int | None:
    row = con.execute("SELECT revision FROM treedb.cut WHERE tree_id = ? AND cut_name = ?",
                      [str(tree_id), cut_name]).fetchone()
    return None if row is None else int(row[0])


def read_cut(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID, cut_name: str, *,
             since: Any = None, until: Any = None) -> list[dict[str, Any]]:
    """Rows of a cut, ordered by (depth, parent_key, position).

    With ``since`` (inclusive) and/or ``until`` (exclusive; UTC, any form ``normalize_timestamp``
    accepts) the cut is pruned to the items whose ``captured_at`` (``v_item_time``) is in the window
    plus all their ancestors (via ``path``); items without a ``captured_at`` are then excluded.
    Every row has ``window_count``: items in the window below it (``descendant_count`` when not
    windowed; 1 for an item). An empty window returns ``[]``. ``captured_at`` is a UTC ``Z`` string.
    """
    tid = str(tree_id)
    windowed = since is not None or until is not None
    cols = ("node_key, parent_key, kind, depth, position, item_id, descendant_count, path, title, summary, tags")
    if not windowed:
        rows = con.execute(
            f"SELECT {cols}, 1, NULL FROM treedb.cut_node WHERE tree_id = ? AND cut_name = ? "
            "ORDER BY depth, parent_key NULLS FIRST, position", [tid, cut_name]).fetchall()
        # captured_at for items
        times = dict(con.execute(
            "SELECT i.item_id, i.captured_at FROM treedb.v_cut_items i WHERE i.tree_id = ? AND i.cut_name = ?",
            [tid, cut_name]).fetchall())
        return [_rec(r, times.get(r[5]), r[6]) for r in rows]
    conds, params = [], [tid, cut_name]
    if since is not None:
        conds.append("captured_at >= CAST(? AS TIMESTAMP)")
        params.append(timestamp_param(since))
    if until is not None:
        conds.append("captured_at < CAST(? AS TIMESTAMP)")
        params.append(timestamp_param(until))
    sql = f"""
        WITH inwin AS (
            SELECT item_id, node_key, path, captured_at FROM treedb.v_cut_items
            WHERE tree_id = ? AND cut_name = ? AND {' AND '.join(conds)}),
        cnt AS (
            SELECT node_key, count(*) AS n FROM (
                SELECT unnest(path) AS node_key FROM inwin
                UNION ALL SELECT node_key FROM inwin) GROUP BY node_key)
        SELECT {', '.join('c.' + x.strip() for x in cols.split(','))}, cnt.n, w.captured_at
        FROM cnt JOIN treedb.cut_node c ON c.node_key = cnt.node_key AND c.tree_id = ? AND c.cut_name = ?
        LEFT JOIN inwin w ON w.node_key = c.node_key
        ORDER BY c.depth, c.parent_key NULLS FIRST, c.position"""
    rows = con.execute(sql, params + [tid, cut_name]).fetchall()
    return [_rec(r[:11], r[12], r[11]) for r in rows]


def _rec(r: tuple, captured: Any, window_count: int) -> dict[str, Any]:
    return {"node_key": r[0], "parent_key": r[1], "kind": r[2], "depth": r[3], "position": r[4],
            "item_id": r[5], "descendant_count": r[6], "path": list(r[7]), "title": r[8],
            "summary": r[9], "tags": None if r[10] is None else list(r[10]),
            "captured_at": None if captured is None else normalize_timestamp(captured),
            "window_count": int(window_count)}


def add_items(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID, cut_name: str,
              placements: Sequence[Mapping[str, Any]]) -> int:
    """Incremental insert of item rows. Returns the number added; ``revision`` += 1 when any was.

    Placement: ``{"item_id", "parent_key"}`` plus optional ``node_key`` (default
    ``item_<sha256(item_id)[:32]>``) and ``position`` (default: after the parent's last child).
    Depth = parent depth + 1, ``path`` = parent path + parent key, ``descendant_count`` of every
    ancestor += 1. An item already in the cut is skipped (a re-capture adds no second row).
    """
    tid = str(tree_id)
    added = 0
    with transaction(con):
        if cut_revision(con, tree_id, cut_name) is None:
            raise KeyError(f"no cut {cut_name!r} for tree {tid}")
        for p in placements:
            item_id, parent_key = p["item_id"], p["parent_key"]
            if con.execute("SELECT 1 FROM treedb.cut_node WHERE tree_id = ? AND cut_name = ? AND item_id = ?",
                           [tid, cut_name, item_id]).fetchone():
                continue
            parent = con.execute("SELECT depth, path, kind FROM treedb.cut_node WHERE tree_id = ? "
                                 "AND cut_name = ? AND node_key = ?", [tid, cut_name, parent_key]).fetchone()
            if parent is None:
                raise KeyError(f"unknown parent_key {parent_key!r}")
            if parent[2] == "item":
                raise ValueError(f"parent {parent_key!r} is an item")
            path = list(parent[1]) + [parent_key]
            key = p.get("node_key") or "item_" + hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:32]
            pos = p.get("position")
            if pos is None:
                pos = con.execute("SELECT coalesce(max(position) + 1, 0) FROM treedb.cut_node WHERE tree_id = ? "
                                  "AND cut_name = ? AND parent_key = ?", [tid, cut_name, parent_key]).fetchone()[0]
            con.execute(
                "INSERT INTO treedb.cut_node (tree_id, cut_name, node_key, parent_key, kind, depth, position, "
                "item_id, descendant_count, path) VALUES (?, ?, ?, ?, 'item', ?, ?, ?, 1, ?)",
                [tid, cut_name, key, parent_key, parent[0] + 1, int(pos), item_id, path])
            con.execute("UPDATE treedb.cut_node SET descendant_count = descendant_count + 1 WHERE tree_id = ? "
                        "AND cut_name = ? AND node_key IN (SELECT unnest(?::VARCHAR[]))", [tid, cut_name, path])
            added += 1
        if added:
            con.execute("UPDATE treedb.cut SET revision = revision + 1, "
                        "updated_at = CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP) WHERE tree_id = ? AND cut_name = ?",
                        [tid, cut_name])
    return added


def update_display(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID, cut_name: str,
                   display: Mapping[str, Mapping[str, Any]]) -> None:
    """Set title / summary / tags for the given node_keys (only keys present in each entry change)."""
    tid = str(tree_id)
    with transaction(con):
        if cut_revision(con, tree_id, cut_name) is None:
            raise KeyError(f"no cut {cut_name!r} for tree {tid}")
        for key, d in display.items():
            for col in ("title", "summary", "tags"):
                if col in d:
                    v = d[col]
                    if col == "tags" and v is not None:
                        v = list(v)
                    con.execute(f"UPDATE treedb.cut_node SET {col} = ? WHERE tree_id = ? AND cut_name = ? "
                                "AND node_key = ?", [v, tid, cut_name, key])
        con.execute("UPDATE treedb.cut SET revision = revision + 1, "
                    "updated_at = CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP) WHERE tree_id = ? AND cut_name = ?",
                    [tid, cut_name])


def cut_settings(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID, cut_name: str) -> dict[str, Any]:
    row = con.execute("SELECT settings FROM treedb.cut WHERE tree_id = ? AND cut_name = ?",
                      [str(tree_id), cut_name]).fetchone()
    if row is None:
        raise KeyError(f"no cut {cut_name!r} for tree {tree_id}")
    return json.loads(row[0])


__all__ = ["add_items", "cut_revision", "cut_settings", "put_cut", "read_cut", "update_display"]
