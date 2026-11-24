"""DuckDB side of the UUID canonicalization (INV-5, INV-22, INV-23).

Mirrors P1's definition; it does not define a new one:

* node identity across backends is ``structural_key(tree_name, item_ids)``
  (INV-22), computed with ``treedb.fixtures.structural_key`` itself;
* a leaf UUID inverts INV-5 (``item_uuid``) through ``treedb.item_registry``,
  whose rows are checked against ``treedb.fixtures.item_uuid``;
* a tree UUID is replaced by the cultivar name (INV-2);
* "Branch UUIDs, orchard row ids, ``branch_ix`` and ``l/r`` order are all
  excluded from the comparison key" (INV-22). Generated ids that are not node
  identities (``orchard.id``, ``tree_branches.graft_id``) and timestamps are
  therefore dropped; ``EXCLUDED_COLUMNS`` lists every dropped column and why.

The item set of a node is computed from the **stored Orchard edges** of the tree
(``treedb.orchard`` + ``treedb.tree_branches``), never from staging, so a wrong
edge shows up as a wrong key rather than being hidden. A node is a branch when
its UUID is a ``tree_branches.branch_id`` of the tree, and a leaf when it is a
registered item UUID; ``edge_ix`` is not used to decide, so INV-4 coding errors
stay visible in the canonical rows. Unresolvable UUIDs raise ``CanonicalError``.

Canonical rows are dicts with the original column names; a table is a list of
rows sorted by their ``canonical_json`` text (views and tables are unordered).
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

import duckdb

from orchardlogic.treedb.fixtures import canonical_json, item_uuid, structural_key

#: Tables and views compared at ingest (P3A deliverable 5).
INGEST_TABLES = ("tree_z", "tree_branches", "orchard", "v_tree_z", "v_tree_map", "v_orchard")

#: table -> {column: reason it is not part of the canonical row}.
EXCLUDED_COLUMNS: dict[str, dict[str, str]] = {
    "tree_branches": {
        "graft_id": "generated gen_random_uuid() token, not a node identity (INV-22)",
        "created_at": "timestamp of the load",
    },
    "orchard": {"id": "orchard row id (INV-22 excludes orchard row ids)"},
    "v_orchard": {
        "graft_id": "generated gen_random_uuid() token, not a node identity (INV-22)",
        "orchard_id": "orchard row id (INV-22 excludes orchard row ids)",
        "vault_id": "VaultBubbles-specific; NULL in TreeDB, a stub vault in the PG harness (PORTING_NOTES TR-1)",
    },
}

#: table -> columns that hold node UUIDs (replaced by structural keys).
NODE_COLUMNS: dict[str, tuple[str, ...]] = {
    "tree_z": ("l_doc_id", "r_doc_id"),
    "tree_branches": ("branch_id",),
    "orchard": ("branch_id", "node_id"),
    "v_tree_z": ("l_doc_id", "r_doc_id"),
    "v_tree_map": ("doc_id",),
    "v_orchard": ("branch_id", "node_id"),
}

_ORDER_COLUMNS: dict[str, str] = {
    "tree_z": "branch_id, l_node, r_node, coph_distance, doc_count, l_doc_id, r_doc_id, tree_id, l_leaf, r_leaf",
    "tree_branches": "branch_id, tree_id",
    "orchard": "branch_id, node_id, edge_ix, coph_distance",
    "v_tree_z": "branch_id, l_node, r_node, coph_distance, doc_count, l_doc_id, r_doc_id, tree_id, "
                "n_docs, l_mod, r_mod, l_leaf, r_leaf",
    "v_tree_map": "tree_id, doc_id, z_id, leaf, branch_id, z_mod, b_side, coph_distance, doc_count",
    "v_orchard": "branch_id, node_id, edge_ix, coph_distance, tree_id",
}


class CanonicalError(ValueError):
    """A UUID could not be mapped to a structural key."""


def tree_id_for(con: duckdb.DuckDBPyConnection, tree_name: str) -> uuid.UUID:
    row = con.execute("SELECT id FROM treedb.tree_registry WHERE name = ?", [tree_name]).fetchone()
    if row is None:
        raise CanonicalError(f"no cultivar named {tree_name!r}")
    return row[0]


def item_for_uuid(con: duckdb.DuckDBPyConnection) -> dict[uuid.UUID, str]:
    """Inverse of INV-5 over the items registry; checks every row against ``item_uuid``."""
    out: dict[uuid.UUID, str] = {}
    for item, u in con.execute("SELECT item_id, item_uuid FROM treedb.item_registry").fetchall():
        if u != item_uuid(item):
            raise CanonicalError(f"item_registry: {item!r} has {u}, CONTRACTS UUID is {item_uuid(item)}")
        out[u] = item
    return out


def node_members(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID) -> dict[uuid.UUID, tuple[str, ...]]:
    """Node UUID -> sorted descendant item_ids, from the stored edges of one tree."""
    items = item_for_uuid(con)
    branches = {r[0] for r in con.execute(
        "SELECT branch_id FROM treedb.tree_branches WHERE tree_id = ?", [tree_id]).fetchall()}
    children: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
    for b, nd in con.execute(
        "SELECT o.branch_id, o.node_id FROM treedb.orchard o "
        "JOIN treedb.tree_branches b USING (branch_id) WHERE b.tree_id = ?", [tree_id]
    ).fetchall():
        children[b].append(nd)

    members: dict[uuid.UUID, tuple[str, ...]] = {}

    def resolve(node: uuid.UUID) -> None:
        # Iterative post-order (deep caterpillar trees exceed Python recursion).
        stack = [node]
        visiting: set[uuid.UUID] = set()
        while stack:
            cur = stack[-1]
            if cur in members:
                stack.pop()
            elif cur in branches:
                pending = [c for c in children.get(cur, []) if c not in members]
                if pending:
                    if cur in visiting:
                        raise CanonicalError(f"cycle through branch {cur}")
                    visiting.add(cur)
                    stack.extend(pending)
                    continue
                acc = [i for c in children.get(cur, []) for i in members[c]]
                if len(acc) != len(set(acc)):
                    raise CanonicalError(f"branch {cur}: an item appears twice below it")
                members[cur] = tuple(sorted(acc))
                visiting.discard(cur)
                stack.pop()
            elif cur in items:
                members[cur] = (items[cur],)
                stack.pop()
            else:
                raise CanonicalError(f"node {cur} is neither a branch of tree {tree_id} nor a registered item")

    for b in branches:
        resolve(b)
    for kids in children.values():
        for ch in kids:
            resolve(ch)
    return members


def key_map(con: duckdb.DuckDBPyConnection, tree_name: str) -> dict[uuid.UUID, str]:
    """Node UUID -> structural key (INV-22) for one cultivar."""
    tid = tree_id_for(con, tree_name)
    return {u: structural_key(tree_name, m) for u, m in node_members(con, tid).items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def canonical_table(con: duckdb.DuckDBPyConnection, table: str, tree_name: str,
                    keys: dict[uuid.UUID, str] | None = None) -> list[dict[str, Any]]:
    """Canonical rows of one ingest table or view, restricted to one cultivar."""
    if table not in INGEST_TABLES:
        raise ValueError(f"not an ingest table: {table}")
    tid = tree_id_for(con, tree_name)
    keys = key_map(con, tree_name) if keys is None else keys
    cols = [c.strip() for c in _ORDER_COLUMNS[table].split(",")]
    if table == "orchard":
        # graph.orchard has no tree_id; a row belongs to the tree of its branch
        # (rows with a dangling branch_id are reported by store.integrity_violations).
        sql = ("SELECT o.branch_id, o.node_id, o.edge_ix, o.coph_distance FROM treedb.orchard o "
               "JOIN treedb.tree_branches b USING (branch_id) WHERE b.tree_id = ?")
    else:
        sql = f"SELECT {', '.join(cols)} FROM treedb.{table} WHERE tree_id = ?"
    cur = con.execute(sql, [tid])
    out: list[dict[str, Any]] = []
    for raw in cur.fetchall():
        row = dict(zip(cols, raw))
        for col in NODE_COLUMNS[table]:
            v = row[col]
            if v is None:
                continue
            if v not in keys:
                raise CanonicalError(f"{table}.{col}: {v} has no structural key in {tree_name!r}")
            row[col] = keys[v]
        if "tree_id" in row:
            row["tree_id"] = tree_name
        out.append({k: _jsonable(v) for k, v in row.items()})
    out.sort(key=canonical_json)
    return out


def canonical_snapshot(con: duckdb.DuckDBPyConnection, tree_name: str) -> dict[str, list[dict[str, Any]]]:
    """Canonical rows of every ingest table/view for one cultivar."""
    keys = key_map(con, tree_name)
    if len(set(keys.values())) != len(keys):
        raise CanonicalError(f"{tree_name!r}: two node UUIDs share a structural key (not bijective)")
    return {t: canonical_table(con, t, tree_name, keys) for t in INGEST_TABLES}
