"""Item and node enrichment outputs (docs/DATA_ORCHARD_V4.md section 5.1).

``enricher_id_or_handle`` is a 64-hex enricher id or ``name@version``.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import duckdb

from orchardlogic.treedb.fixtures import canonical_json
from orchardlogic.treedb.tx import transaction

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def resolve_enricher(con: duckdb.DuckDBPyConnection, id_or_handle: str) -> str:
    """Enricher id for an id or ``name@version``; ``KeyError`` when unknown."""
    if _HEX64.match(id_or_handle):
        row = con.execute("SELECT enricher_id FROM treedb.enricher WHERE enricher_id = ?", [id_or_handle]).fetchone()
    else:
        name, _, version = id_or_handle.partition("@")
        row = con.execute("SELECT enricher_id FROM treedb.enricher WHERE name = ? AND version = ?",
                          [name, version]).fetchone()
    if row is None:
        raise KeyError(f"unknown enricher {id_or_handle!r}")
    return row[0]


def _check_outputs(outputs: Mapping[str, Any]) -> None:
    for k, v in outputs.items():
        if not isinstance(v, Mapping):
            raise ValueError(f"output for {k!r} must be a JSON object")


def put_item_outputs(con: duckdb.DuckDBPyConnection, enricher_id: str, outputs: Mapping[str, dict],
                     input_sha: Mapping[str, str] | None = None) -> int:
    """Upsert ``{item_id: output}`` for an enricher id or handle. Returns the number of rows."""
    _check_outputs(outputs)
    if not outputs:
        return 0
    ids = list(outputs)
    with transaction(con):
        eid = resolve_enricher(con, enricher_id)
        con.execute(
            "INSERT OR REPLACE INTO treedb.item_enrichment (item_id, enricher_id, output, input_sha256) "
            "SELECT unnest(?::VARCHAR[]), ?, unnest(?::JSON[]), unnest(?::VARCHAR[])",
            [ids, eid, [canonical_json(outputs[i]) for i in ids], [(input_sha or {}).get(i) for i in ids]])
    return len(ids)


def get_item_outputs(con: duckdb.DuckDBPyConnection, enricher_id_or_handle: str,
                     item_ids: Sequence[str] | None = None) -> dict[str, dict]:
    eid = resolve_enricher(con, enricher_id_or_handle)
    if item_ids is None:
        rows = con.execute("SELECT item_id, output FROM treedb.item_enrichment WHERE enricher_id = ? "
                           "ORDER BY item_id", [eid]).fetchall()
    else:
        rows = con.execute("SELECT item_id, output FROM treedb.item_enrichment WHERE enricher_id = ? "
                           "AND item_id IN (SELECT unnest(?)) ORDER BY item_id", [eid, list(item_ids)]).fetchall()
    return {i: json.loads(o) for i, o in rows}


def put_node_outputs(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID, enricher_id: str,
                     outputs: Mapping[str, dict]) -> int:
    """Upsert ``{node_key: output}`` for a tree."""
    _check_outputs(outputs)
    if not outputs:
        return 0
    keys = list(outputs)
    with transaction(con):
        eid = resolve_enricher(con, enricher_id)
        con.execute(
            "INSERT OR REPLACE INTO treedb.node_enrichment (tree_id, node_key, enricher_id, output) "
            "SELECT ?::UUID, unnest(?::VARCHAR[]), ?, unnest(?::JSON[])",
            [str(tree_id), keys, eid, [canonical_json(outputs[k]) for k in keys]])
    return len(keys)


def get_node_outputs(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID,
                     enricher_id_or_handle: str) -> dict[str, dict]:
    eid = resolve_enricher(con, enricher_id_or_handle)
    rows = con.execute("SELECT node_key, output FROM treedb.node_enrichment WHERE tree_id = ? AND enricher_id = ? "
                       "ORDER BY node_key", [str(tree_id), eid]).fetchall()
    return {k: json.loads(o) for k, o in rows}


__all__ = ["get_item_outputs", "get_node_outputs", "put_item_outputs", "put_node_outputs", "resolve_enricher"]
