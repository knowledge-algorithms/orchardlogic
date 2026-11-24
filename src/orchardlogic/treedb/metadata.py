"""Labels and cards (architecture 8.9, INV-11): set one on a node, change nothing else.

Labels and cards are stored against node identity (``treedb.node_labels`` and
``treedb.node_cards``). Changing one never touches ``orchard``, ``tree_branches``,
``tree_z`` or any cohort, and does **not** change the cultivar's ``state_version``:
the version counts changes of topology and lineage (D-007), which these edits are not.
Labels therefore follow a node through attach and detach, which also keep node identity.

A node that has no label or card yet gets ``source_node_key`` ``"set:<node id>"``
(there is no Orchard source key for it); an existing row keeps its key.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import duckdb

from orchardlogic.treedb import read
from orchardlogic.treedb.fixtures import canonical_json


def _node(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, node_id: Any) -> tuple[uuid.UUID, uuid.UUID, str]:
    """``(tree_id, node_id, source_key)``; raises ``read.NodeNotFoundError`` for a node of another cultivar."""
    tree_id, _, _ = read.cultivar_version(con, cultivar)
    nid = node_id if isinstance(node_id, uuid.UUID) else uuid.UUID(str(node_id))
    read.get_node(con, tree_id, nid)  # raises NodeNotFoundError
    row = con.execute(
        "SELECT source_node_key FROM treedb.node_labels WHERE tree_id = $t AND node_id = $n "
        "UNION ALL SELECT source_node_key FROM treedb.node_cards WHERE tree_id = $t AND node_id = $n",
        {"t": tree_id, "n": nid}).fetchone()
    return tree_id, nid, row[0] if row else f"set:{nid}"


def set_label(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, node_id: Any,
              label_set: str, label: str) -> None:
    """Set (insert or replace) the node's label in ``label_set``."""
    if not isinstance(label_set, str) or not label_set or not isinstance(label, str):
        raise ValueError("label_set must be a non-empty string and label a string")
    tree_id, nid, key = _node(con, cultivar, node_id)
    con.execute(
        "INSERT INTO treedb.node_labels (tree_id, node_id, label_set, label, source_node_key) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT (tree_id, node_id, label_set) DO UPDATE SET label = excluded.label",
        [tree_id, nid, label_set, label, key])


def set_card(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, node_id: Any, card: Any) -> None:
    """Set (insert or replace) the node's card. ``card`` is any JSON-serialisable value, stored as canonical JSON."""
    text = canonical_json(card)
    json.loads(text)
    tree_id, nid, key = _node(con, cultivar, node_id)
    con.execute(
        "INSERT INTO treedb.node_cards (tree_id, node_id, card, source_node_key) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (tree_id, node_id) DO UPDATE SET card = excluded.card",
        [tree_id, nid, text, key])


__all__ = ["set_card", "set_label"]
