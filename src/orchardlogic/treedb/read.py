"""Read operations: ancestry, compressed Steiner subtree, tree and node retrieval.

Phase P3B. Port of the installed PostgreSQL read functions (D-002, D-009):

======================================  ==================================================
TreeDB                                  PostgreSQL reference
======================================  ==================================================
``ancestry_paths``                      ``graph.get_ancestry_paths(uuid[])``
                                        ``reference/pg_installed/graph/function/
                                        get_ancestry_paths_uuid_arr.sql:1-65``
``compressed_subtree``                  ``graph.get_compressed_subtree(uuid[])``
                                        ``reference/pg_installed/graph/function/
                                        get_compressed_subtree_uuid_arr.sql:1-121``
======================================  ==================================================

Both are DuckDB table macros in ``src/treedb/sql/read/`` (``treedb.get_ancestry_paths``,
``treedb.get_compressed_subtree``), callable from SQL as
``SELECT * FROM treedb.get_compressed_subtree([...]::UUID[])``. The Python functions
here only install the macros, bind the argument and fetch the rows. They return the
**raw PostgreSQL contract** (INV-14): the PG columns in PG order, 1-based numbering,
the unary stem row, NULL indexes for non-leaf entries. Nothing is normalized here;
the proposed SciPy normalization (OD-1) is the separate module ``treedb.normalize``.

Inspectable stages (Architecture §8 "Comparison policy"): ``path_closure``,
``full_linkage`` (grouping by contributing leaves, ``attached_docs``),
``compressed_linkage`` (unary-chain compression) and ``indexed_linkage`` (Z-like
numbering), plus ``inspect_compressed_subtree`` for all of them at once. They are
the same macros ``get_compressed_subtree`` is composed of, so they cannot drift.

Determinism: where PostgreSQL leaves an order open (no tie-break), TreeDB appends a
tie-break after the PG keys (DET-1..DET-5, ``docs/porting/P3B.md``). The rules are a
proposal for OD-1 at G3 (D-015).

Item-level convenience (INV-6, INV-7, INV-20): ``resolve_items`` maps item ids plus a
cultivar to orchard entry rows and reports, never drops, unknown items (not in the
item registry) and items missing from the cultivar. The raw functions themselves keep
PG's behaviour of silently ignoring unknown orchard ids (PG_BASELINE §10, D-016d).
The item-level reads also report the ``components`` the present items fall into
(G3 D-024 Q8: a cohort over a split cultivar gives a forest plus this notice, never an
error).

Cycles (G3 D-024 Q7): the recursive ``get_ancestry_paths`` macro carries a depth guard
(CYC-2, ``docs/porting/P3B.md``) that raises ``AncestryCycleError`` on a cycle instead of
recursing forever; on rows without a cycle its output is unchanged.

Tree retrieval (Architecture §7 "Retrieve tree/node"): ``tree_info``, ``get_node``,
``node_children``, ``node_edges``, ``node_labels``, ``node_card``, ``node_ancestry``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Iterable, Sequence

import duckdb

#: Output columns of ``graph.get_ancestry_paths`` (get_ancestry_paths_uuid_arr.sql:2).
ANCESTRY_COLUMNS: tuple[str, ...] = (
    "orchard_id", "tree_id", "branch_id", "node_id", "edge_ix", "coph_distance",
    "depth", "start_orchard_id", "start_node_id",
)

#: Output columns of ``graph.get_compressed_subtree`` (get_compressed_subtree_uuid_arr.sql:2; INV-14).
COMPRESSED_COLUMNS: tuple[str, ...] = (
    "tree_id", "branch_id", "branch_ix", "l_node_ix", "r_node_ix", "l_node_id", "r_node_id",
    "l_edge_ix", "r_edge_ix", "doc_count", "coph_distance",
)

#: DuckDB types of the raw outputs (the PG declared types, mapped as in PORTING_NOTES TBL-1).
ANCESTRY_TYPES: tuple[str, ...] = (
    "UUID", "UUID", "UUID", "UUID", "INTEGER", "DOUBLE", "INTEGER", "UUID", "UUID",
)
COMPRESSED_TYPES: tuple[str, ...] = (
    "UUID", "UUID", "BIGINT", "BIGINT", "BIGINT", "UUID", "UUID", "INTEGER", "INTEGER", "BIGINT", "DOUBLE",
)

#: Stage name -> table macro (``sql/read/get_compressed_subtree.sql``).
STAGE_MACROS: dict[str, str] = {
    "path_closure": "treedb.cs_paths",
    "full_linkage": "treedb.cs_full_linkage",
    "compressed_linkage": "treedb.cs_compressed_linkage",
    "indexed_linkage": "treedb.cs_indexed_linkage",
}

_SQL_FILES = ("sql/read/get_ancestry_paths.sql", "sql/read/get_compressed_subtree.sql")


class ReadError(ValueError):
    """A read request that cannot be answered as asked."""


class MissingItemsError(ReadError):
    """Raised with ``on_missing="error"`` when items are unknown or absent from the cultivar."""

    def __init__(self, cultivar: str, unknown: Sequence[str], missing: Sequence[str]):
        self.cultivar = cultivar
        self.unknown_items = tuple(unknown)
        self.missing_items = tuple(missing)
        super().__init__(
            f"cultivar {cultivar!r}: {len(self.unknown_items)} unknown item(s) {list(self.unknown_items)[:20]}, "
            f"{len(self.missing_items)} item(s) not in the cultivar {list(self.missing_items)[:20]}"
        )


class NodeNotFoundError(ReadError, KeyError):
    """The node id is not a node of the cultivar."""


class AncestryCycleError(ReadError):
    """A recursive read met a cycle in the stored rows (a node is its own ancestor).

    The depth guard of ``treedb.get_ancestry_paths`` (CYC-2, G3 D-024 Q7) raises
    instead of recursing forever. PostgreSQL has no guard and does not terminate.
    ``treedb.mutate`` refuses to commit a cycle: on attach (CYC-1, G3 D-024 Q7,
    ``CyclicAttachError``) and on detach, whose cross-tree GRAFT-1 rewire could close
    a loop in another tree (CYC-3, G6 D-038, ``CyclicDetachError``; PG commits both).
    This read guard is therefore for rows written outside ``treedb.mutate`` (direct
    SQL, another tool). The query layer (``treedb.query`` execute and explain)
    raises this error on a cycle as well.
    """


# ---------------------------------------------------------------------------
# Installation of the read macros
# ---------------------------------------------------------------------------

def _sql_texts() -> list[str]:
    return [resources.files("orchardlogic.treedb").joinpath(p).read_text(encoding="utf-8") for p in _SQL_FILES]


def read_sql_version() -> str:
    """Hash of the read SQL texts; stored in the ``treedb.read_sql_version()`` macro."""
    h = hashlib.sha256()
    for text in _sql_texts():
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def install(con: duckdb.DuckDBPyConnection) -> None:
    """(Re)create the read macros. Idempotent (``CREATE OR REPLACE``)."""
    for text in _sql_texts():
        con.execute(text)
    con.execute(f"CREATE OR REPLACE MACRO treedb.read_sql_version() AS '{read_sql_version()}'")


def _ensure(con: duckdb.DuckDBPyConnection) -> None:
    """Install the macros when missing or written by another version of this module.

    A file store keeps macros across reopen, so a stale definition is replaced.
    """
    try:
        found = con.execute("SELECT treedb.read_sql_version()").fetchone()[0]
    except duckdb.Error:
        found = None
    if found != read_sql_version():
        install(con)


ensure_installed = _ensure
ensure_installed.__doc__ = """Public name of ``_ensure``: install the read macros when missing or outdated."""


def _uuids(ids: Iterable[Any]) -> list[uuid.UUID]:
    out = []
    for v in ids:
        out.append(v if isinstance(v, uuid.UUID) else uuid.UUID(str(v)))
    return out


#: Text of the depth guard's error() in ``sql/read/get_ancestry_paths.sql`` (CYC-2).
_CYCLE_MARKER = "treedb.get_ancestry_paths: ancestry cycle"


def _fetch(con: duckdb.DuckDBPyConnection, macro: str, orchard_ids: Iterable[Any],
           order_all: bool = False) -> tuple[list[str], list[tuple]]:
    _ensure(con)
    # Stage macros are unordered sets, like the PG CTEs; inspection sorts them by every column.
    order = " ORDER BY ALL" if order_all else ""
    try:
        cur = con.execute(f"SELECT * FROM {macro}(CAST(? AS UUID[])){order}", [_uuids(orchard_ids)])
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall()
    except duckdb.Error as exc:
        if _CYCLE_MARKER in str(exc):
            raise AncestryCycleError(str(exc)) from exc
        raise


# ---------------------------------------------------------------------------
# Raw PostgreSQL-contract functions
# ---------------------------------------------------------------------------

def ancestry_paths(con: duckdb.DuckDBPyConnection, orchard_ids: Iterable[Any]) -> list[tuple]:
    """``graph.get_ancestry_paths``: every orchard row from each start row up to its root.

    Rows are tuples in ``ANCESTRY_COLUMNS`` order. Unknown orchard ids are ignored,
    as in PostgreSQL (use ``resolve_items`` to get them reported). Row order: DET-1
    (depth, start node, start row), a refinement of PG's unspecified order.
    """
    return _fetch(con, "treedb.get_ancestry_paths", orchard_ids)[1]


def compressed_subtree(con: duckdb.DuckDBPyConnection, orchard_ids: Iterable[Any]) -> list[tuple]:
    """``graph.get_compressed_subtree``: the raw compressed Steiner subtree (INV-14, D-004).

    Tuples in ``COMPRESSED_COLUMNS`` order, ordered by ``coph_distance`` (ties: DET-5).
    1-based numbering, the unary stem row when the entries' LCA is below the root,
    NULL indexes for non-leaf entries (D-015). Unknown orchard ids are ignored, as
    in PostgreSQL.

    **D-019 (G3, a divergence; P3B MIX-2).** Entries from more than one cultivar are
    refused with ``MixedCultivarError``, which names the cultivars. PostgreSQL's
    function accepts them and mixes the trees in one numbering (MIX-1), which has no
    meaning. The SQL macro ``treedb.get_compressed_subtree`` stays literal; the check
    and the read run in one transaction snapshot (the caller's, if one is open).
    """
    ids = _uuids(orchard_ids)
    _ensure(con)
    with _Snapshot(con):
        _refuse_mixed_cultivars(con, ids, "compressed_subtree")
        return _fetch(con, "treedb.get_compressed_subtree", ids)[1]


class MixedCultivarError(ReadError):
    """D-019: compressed-subtree entries belong to more than one cultivar.

    ``cultivars`` are the cultivar names (sorted), ``tree_ids`` their registry ids and
    ``entry_counts`` the number of distinct entries per cultivar. PostgreSQL raises
    nothing (its output mixes the trees, MIX-1); TreeDB refuses (G3, D-019).
    """

    def __init__(self, function: str, found: Sequence[tuple[str, uuid.UUID, int]]):
        self.cultivars = tuple(name for name, _, _ in found)
        self.tree_ids = {name: tid for name, tid, _ in found}
        self.entry_counts = {name: n for name, _, n in found}
        shown = ", ".join(f"{name!r} ({n} entr{'y' if n == 1 else 'ies'})" for name, _, n in found)
        super().__init__(
            f"{function}: the entries belong to {len(found)} cultivars: {shown}. A compressed subtree is "
            "defined within one cultivar; call it once per cultivar (D-019; PostgreSQL's function would "
            "mix the trees in one numbering)")


def _refuse_mixed_cultivars(con: duckdb.DuckDBPyConnection, ids: list[uuid.UUID], function: str) -> None:
    """D-019: raise ``MixedCultivarError`` when the known entries span more than one cultivar.
    Unknown ids are ignored here as in the macro (UNK-1)."""
    found = [(name, tid, int(n)) for name, tid, n in con.execute(
        """
        SELECT r.name, v.tree_id, count(DISTINCT v.orchard_id)
        FROM (SELECT unnest(CAST(? AS UUID[])) AS orchard_id) q
        JOIN treedb.v_orchard v ON v.orchard_id = q.orchard_id
        JOIN treedb.tree_registry r ON r.id = v.tree_id
        GROUP BY r.name, v.tree_id
        ORDER BY r.name
        """, [ids]).fetchall()]
    if len(found) > 1:
        raise MixedCultivarError(function, found)


@dataclass(frozen=True)
class Stage:
    """Rows of one inspectable stage."""

    name: str
    macro: str
    columns: tuple[str, ...]
    rows: list[tuple]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]


def _stage(con: duckdb.DuckDBPyConnection, name: str, orchard_ids: Iterable[Any]) -> Stage:
    macro = STAGE_MACROS[name]
    cols, rows = _fetch(con, macro, orchard_ids, order_all=True)
    return Stage(name, macro, tuple(cols), rows)


def path_closure(con: duckdb.DuckDBPyConnection, orchard_ids: Iterable[Any]) -> Stage:
    """Stage 1, CTEs ``doc_list`` + ``paths`` (gcs:19-29): the ancestry of the distinct entries."""
    return _stage(con, "path_closure", orchard_ids)


def full_linkage(con: duckdb.DuckDBPyConnection, orchard_ids: Iterable[Any]) -> Stage:
    """Stage 2, CTE ``full_linkage`` (gcs:31-45): each edge with the sorted start nodes below it
    (``attached_docs``), their start rows (``attached_grafts``), ``doc_count``, plus TreeDB's
    ``min_depth`` (DET-2)."""
    return _stage(con, "full_linkage", orchard_ids)


def compressed_linkage(con: duckdb.DuckDBPyConnection, orchard_ids: Iterable[Any]) -> Stage:
    """Stage 3, CTE ``compressed_linkage`` (gcs:47-67): one row per unary chain
    (top branch, bottom node, max height, bottom edge_ix)."""
    return _stage(con, "compressed_linkage", orchard_ids)


def indexed_linkage(con: duckdb.DuckDBPyConnection, orchard_ids: Iterable[Any]) -> Stage:
    """Stage 4, CTEs ``node_ix``, ``branch_ix``, ``indexed_linkage`` (gcs:69-104): 1-based leaf
    and branch indexes and the left/right slot ``lr``."""
    return _stage(con, "indexed_linkage", orchard_ids)


def inspect_compressed_subtree(con: duckdb.DuckDBPyConnection, orchard_ids: Iterable[Any]) -> dict[str, Stage]:
    """All four stages and the final raw result, keyed ``path_closure``, ``full_linkage``,
    ``compressed_linkage``, ``indexed_linkage``, ``result``. Read in one transaction
    snapshot when the caller has none open."""
    ids = _uuids(orchard_ids)
    _ensure(con)
    own_txn = not _in_transaction(con)
    if own_txn:
        con.execute("BEGIN TRANSACTION")
    try:
        # D-019: the inspection returns the compressed result too, so it refuses the same input.
        _refuse_mixed_cultivars(con, ids, "inspect_compressed_subtree")
        out = {name: _stage(con, name, ids) for name in STAGE_MACROS}
        cols, rows = _fetch(con, "treedb.get_compressed_subtree", ids)
        out["result"] = Stage("result", "treedb.get_compressed_subtree", tuple(cols), rows)
    except BaseException:
        if own_txn:
            con.execute("ROLLBACK")
        raise
    if own_txn:
        con.execute("COMMIT")
    return out


def _in_transaction(con: duckdb.DuckDBPyConnection) -> bool:
    q = "SELECT current_transaction_id()"
    return con.execute(q).fetchone()[0] == con.execute(q).fetchone()[0]


# ---------------------------------------------------------------------------
# Cultivars and entry resolution (INV-6, INV-7, INV-20)
# ---------------------------------------------------------------------------

def _tree(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID) -> tuple[uuid.UUID, str, int]:
    if isinstance(cultivar, uuid.UUID):
        row = con.execute("SELECT id, name, state_version FROM treedb.tree_registry WHERE id = ?",
                          [cultivar]).fetchone()
    else:
        row = con.execute("SELECT id, name, state_version FROM treedb.tree_registry WHERE name = ?",
                          [cultivar]).fetchone()
    if row is None:
        raise ReadError(f"no cultivar {cultivar!r}")
    return row[0], row[1], int(row[2])


def in_transaction(con: duckdb.DuckDBPyConnection) -> bool:
    """Public: True when ``con`` is inside an explicit transaction (alias of ``_in_transaction``)."""
    return _in_transaction(con)


def cultivar_version(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID) -> tuple[uuid.UUID, str, int]:
    """Public: ``(tree_id, name, state_version)`` of a cultivar by name or id.

    Raises ``ReadError`` for an unknown cultivar. Alias of ``_tree``.
    """
    return _tree(con, cultivar)


@dataclass(frozen=True)
class EntryResolution:
    """Orchard entry rows for a list of item ids in one cultivar.

    ``orchard_ids`` follows the order of ``requested`` (first occurrence), skipping
    only the items reported in ``unknown_items`` or ``missing_items``.
    """

    cultivar: str
    tree_id: uuid.UUID
    state_version: int
    requested: tuple[str, ...]
    orchard_ids: tuple[uuid.UUID, ...]
    item_rows: dict[str, uuid.UUID]
    unknown_items: tuple[str, ...]   # not in treedb.item_registry
    missing_items: tuple[str, ...]   # registered, but no orchard row in this cultivar

    @property
    def complete(self) -> bool:
        return not self.unknown_items and not self.missing_items


_ON_MISSING = ("report", "error")


def resolve_items(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, item_ids: Iterable[str],
                  *, on_missing: str = "report") -> EntryResolution:
    """Map item ids to the cultivar's leaf orchard rows (INV-6).

    Each item id goes item_registry -> item UUID (INV-5) -> the ``treedb.orchard`` row
    whose ``node_id`` is that UUID and whose branch belongs to the cultivar. Items are
    never dropped silently (INV-20): ``unknown_items`` lists ids absent from the
    registry and ``missing_items`` ids with no row in this cultivar. ``on_missing``:
    ``"report"`` (default, the P4A OD-5 proposal) or ``"error"`` (raise
    ``MissingItemsError``). OD-5 itself is open (G3). Repeated ids are collapsed (a
    cohort is a set); that is not a drop.

    An item with more than one row in the tree violates INV-6 ("exactly one such
    row"); that raises ``ReadError`` rather than picking one.
    """
    if on_missing not in _ON_MISSING:
        raise ValueError(f"on_missing must be one of {_ON_MISSING}")
    if isinstance(item_ids, (str, bytes)):
        raise TypeError("item_ids must be a collection of item ids, not one string")
    tree_id, name, version = _tree(con, cultivar)
    requested = tuple(dict.fromkeys(str(i) for i in item_ids))
    rows = con.execute(
        """
        SELECT q.item_id, i.item_uuid, v.orchard_id
        FROM (SELECT unnest(CAST(? AS VARCHAR[])) AS item_id) q
        LEFT JOIN treedb.item_registry i ON i.item_id = q.item_id
        LEFT JOIN treedb.v_orchard v ON v.node_id = i.item_uuid AND v.tree_id = ?
        """,
        [list(requested), tree_id],
    ).fetchall()
    found: dict[str, list[uuid.UUID]] = {}
    unknown, registered = set(), set()
    for item, item_u, oid in rows:
        if item_u is None:
            unknown.add(item)
            continue
        registered.add(item)
        if oid is not None:
            found.setdefault(item, []).append(oid)
    dup = sorted(i for i, v in found.items() if len(v) > 1)
    if dup:
        raise ReadError(f"cultivar {name!r}: item(s) with more than one orchard row (INV-6): {dup[:20]}")
    unknown_t = tuple(i for i in requested if i in unknown)
    missing_t = tuple(i for i in requested if i in registered and i not in found)
    if on_missing == "error" and (unknown_t or missing_t):
        raise MissingItemsError(name, unknown_t, missing_t)
    item_rows = {i: found[i][0] for i in requested if i in found}
    return EntryResolution(name, tree_id, version, requested, tuple(item_rows.values()), item_rows,
                           unknown_t, missing_t)


@dataclass(frozen=True)
class NodeResolution:
    """Orchard entry rows for node ids (leaf item UUIDs or branch ids) in one cultivar.

    ``no_row_nodes`` lists nodes of the tree that have no incoming orchard row (a
    root, INV-7); ``unknown_nodes`` lists ids that are not nodes of this tree.
    """

    cultivar: str
    tree_id: uuid.UUID
    state_version: int
    orchard_ids: tuple[uuid.UUID, ...]
    node_rows: dict[uuid.UUID, uuid.UUID]
    no_row_nodes: tuple[uuid.UUID, ...]
    unknown_nodes: tuple[uuid.UUID, ...]


def resolve_nodes(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID,
                  node_ids: Iterable[Any]) -> NodeResolution:
    """Map node ids to their incoming orchard rows (INV-7 non-leaf entries, or leaves)."""
    tree_id, name, version = _tree(con, cultivar)
    req = list(dict.fromkeys(_uuids(node_ids)))
    rows = con.execute(
        "SELECT node_id, orchard_id FROM treedb.v_orchard WHERE tree_id = ? AND list_contains(CAST(? AS UUID[]), node_id)",
        [tree_id, req]).fetchall()
    found: dict[uuid.UUID, list[uuid.UUID]] = {}
    for nd, oid in rows:
        found.setdefault(nd, []).append(oid)
    dup = sorted(str(n) for n, v in found.items() if len(v) > 1)
    if dup:
        raise ReadError(f"cultivar {name!r}: node(s) with more than one orchard row: {dup[:20]}")
    branches = {r[0] for r in con.execute(
        "SELECT branch_id FROM treedb.tree_branches WHERE tree_id = ? AND list_contains(CAST(? AS UUID[]), branch_id)",
        [tree_id, req]).fetchall()}
    node_rows = {n: found[n][0] for n in req if n in found}
    no_row = tuple(n for n in req if n not in found and n in branches)
    unknown = tuple(n for n in req if n not in found and n not in branches)
    return NodeResolution(name, tree_id, version, tuple(node_rows.values()), node_rows, no_row, unknown)


@dataclass(frozen=True)
class Component:
    """One connected component of the cultivar that holds some of the requested items.

    ``root_id`` is the component's root: the branch at the top of the items' ancestry
    chains (it has no incoming orchard row in the tree; one of ``tree_info()["roots"]``).
    ``item_ids`` are the present requested items below it, in request order.
    """

    root_id: uuid.UUID
    item_ids: tuple[str, ...]


@dataclass(frozen=True)
class ItemReadResult:
    """Raw rows of an item-level read plus the resolution that produced their entries.

    ``components`` (G3 D-024 Q8, "forest plus notice"): the connected components of the
    cultivar that the present items fall into, in the order of each component's first
    requested item. After a tree split (D-016c) a cohort can span several components; the
    raw rows are then a forest (PG's output, unchanged) and ``component_count > 1`` is the
    notice. Never an error. Empty when no requested item is present.
    """

    resolution: EntryResolution
    columns: tuple[str, ...]
    rows: list[tuple]
    components: tuple[Component, ...] = ()

    @property
    def missing_items(self) -> tuple[str, ...]:
        return self.resolution.missing_items

    @property
    def unknown_items(self) -> tuple[str, ...]:
        return self.resolution.unknown_items

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def is_forest(self) -> bool:
        """True when the present items lie in more than one component (D-024 Q8 notice)."""
        return len(self.components) > 1


def _components(res: EntryResolution, ancestry_rows: Sequence[tuple]) -> tuple[Component, ...]:
    """Group the resolved items by the root at the top of their ancestry chain (D-024 Q8).

    ``ancestry_rows`` are ``ancestry_paths(res.orchard_ids)`` rows (``ANCESTRY_COLUMNS``):
    for each start row, the row of greatest depth hangs from the component root.
    """
    i_branch, i_depth, i_start = (ANCESTRY_COLUMNS.index(c) for c in ("branch_id", "depth", "start_orchard_id"))
    top: dict[uuid.UUID, tuple[int, uuid.UUID]] = {}
    for r in ancestry_rows:
        start, depth = r[i_start], r[i_depth]
        if start not in top or depth > top[start][0]:
            top[start] = (depth, r[i_branch])
    groups: dict[uuid.UUID, list[str]] = {}
    for item, oid in res.item_rows.items():          # request order (EntryResolution)
        groups.setdefault(top[oid][1], []).append(item)
    return tuple(Component(root, tuple(items)) for root, items in groups.items())


class _Snapshot:
    """One read transaction around several statements, unless the caller already has one."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.own = False

    def __enter__(self) -> "_Snapshot":
        self.own = not _in_transaction(self.con)
        if self.own:
            self.con.execute("BEGIN TRANSACTION")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.own:
            self.con.execute("ROLLBACK" if exc_type is not None else "COMMIT")


def snapshot(con: duckdb.DuckDBPyConnection) -> "_Snapshot":
    """Public context manager: one read transaction around several statements.

    Opens a transaction unless the caller already has one (then it is a no-op).
    Same object as the private ``_Snapshot``.
    """
    return _Snapshot(con)


def ancestry_paths_for_items(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID,
                             item_ids: Iterable[str], *, on_missing: str = "report") -> ItemReadResult:
    """``ancestry_paths`` for item ids in one cultivar; unknown/missing items are reported,
    and so are the ``components`` the present items fall into (D-024 Q8)."""
    _ensure(con)
    with _Snapshot(con):
        res = resolve_items(con, cultivar, item_ids, on_missing=on_missing)
        rows = ancestry_paths(con, res.orchard_ids)
    return ItemReadResult(res, ANCESTRY_COLUMNS, rows, _components(res, rows))


def compressed_subtree_for_items(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID,
                                 item_ids: Iterable[str], *, on_missing: str = "report") -> ItemReadResult:
    """``compressed_subtree`` for item ids in one cultivar; unknown/missing items are reported,
    and so are the ``components`` the present items fall into (D-024 Q8). Over several
    components the raw rows are a forest (one top row per component, as in PostgreSQL);
    that is reported, never raised."""
    _ensure(con)
    with _Snapshot(con):
        res = resolve_items(con, cultivar, item_ids, on_missing=on_missing)
        rows = compressed_subtree(con, res.orchard_ids)
        comps = _components(res, ancestry_paths(con, res.orchard_ids))
    return ItemReadResult(res, COMPRESSED_COLUMNS, rows, comps)


# ---------------------------------------------------------------------------
# Tree and node retrieval (Architecture §7 "Retrieve tree/node")
# ---------------------------------------------------------------------------

_V_ORCHARD_COLS = ("orchard_id", "branch_id", "node_id", "edge_ix", "coph_distance", "tree_id", "graft_id")


def _orchard_rows(con, sql: str, params: list) -> list[dict[str, Any]]:
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def tree_info(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID) -> dict[str, Any]:
    """Registry record plus current shape counts of one cultivar.

    ``root_branch_id`` is the root stored at import (ROOT-1, D-013); ``roots`` are the
    current structural roots (branches no orchard row of the tree points to), which
    can differ after a mutation (D-016c).
    """
    tree_id, name, version = _tree(con, cultivar)
    reg = _orchard_rows(con, "SELECT id, name, root_branch_id, state_version, created_at, z_sha256, item_order "
                             "FROM treedb.tree_registry WHERE id = ?", [tree_id])[0]
    counts = con.execute(
        """
        SELECT (SELECT count(*) FROM treedb.tree_branches WHERE tree_id = $t),
               (SELECT count(*) FROM treedb.v_orchard WHERE tree_id = $t),
               (SELECT count(*) FROM treedb.v_orchard WHERE tree_id = $t AND edge_ix = 1)
        """, {"t": tree_id}).fetchone()
    roots = [r[0] for r in con.execute(
        """
        SELECT b.branch_id FROM treedb.tree_branches b
        WHERE b.tree_id = ? AND NOT EXISTS (
            SELECT 1 FROM treedb.v_orchard v WHERE v.tree_id = b.tree_id AND v.node_id = b.branch_id)
        ORDER BY b.branch_id
        """, [tree_id]).fetchall()]
    return {
        "tree_id": tree_id, "name": name, "state_version": version,
        "root_branch_id": reg["root_branch_id"], "roots": roots,
        "branches": int(counts[0]), "orchard_rows": int(counts[1]), "leaf_rows": int(counts[2]),
        "created_at": reg["created_at"], "z_sha256": reg["z_sha256"],
        "item_count_at_import": len(reg["item_order"] or []),
    }


@dataclass(frozen=True)
class NodeRecord:
    """One node of a cultivar (a leaf item or a branch), as currently stored."""

    tree_id: uuid.UUID
    cultivar: str
    state_version: int
    node_id: uuid.UUID
    kind: str                         # "leaf" | "branch"
    item_id: str | None               # leaves only
    is_root: bool                     # branch with no incoming orchard row
    orchard_id: uuid.UUID | None      # incoming row (None for a root)
    parent_branch_id: uuid.UUID | None
    edge_ix: int | None
    edge_coph_distance: float | None  # height of the parent branch (stored edge distance)
    height: float | None              # branches: max coph_distance of the child rows; leaves: None
    child_count: int
    labels: dict[str, str] = field(default_factory=dict)
    card: Any = None


def _node_kind(con, tree_id: uuid.UUID, node_id: uuid.UUID) -> tuple[str, str | None]:
    if con.execute("SELECT 1 FROM treedb.tree_branches WHERE tree_id = ? AND branch_id = ?",
                   [tree_id, node_id]).fetchone():
        return "branch", None
    row = con.execute(
        "SELECT i.item_id FROM treedb.v_orchard v JOIN treedb.item_registry i ON i.item_uuid = v.node_id "
        "WHERE v.tree_id = ? AND v.node_id = ?", [tree_id, node_id]).fetchone()
    if row:
        return "leaf", row[0]
    if con.execute("SELECT 1 FROM treedb.v_orchard WHERE tree_id = ? AND node_id = ?",
                   [tree_id, node_id]).fetchone():
        return "other", None          # e.g. an external graft node (edge_ix 3)
    raise NodeNotFoundError(f"{node_id} is not a node of this cultivar")


def get_node(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, node_id: Any) -> NodeRecord:
    """Node record: kind, incoming edge, height, child count, labels and card."""
    tree_id, name, version = _tree(con, cultivar)
    nid = _uuids([node_id])[0]
    kind, item = _node_kind(con, tree_id, nid)
    inc = _orchard_rows(con, "SELECT orchard_id, branch_id, edge_ix, coph_distance FROM treedb.v_orchard "
                             "WHERE tree_id = ? AND node_id = ? ORDER BY orchard_id", [tree_id, nid])
    if len(inc) > 1:
        raise ReadError(f"node {nid} has {len(inc)} incoming rows in {name!r} (INV-6)")
    kids = con.execute("SELECT count(*), max(coph_distance) FROM treedb.v_orchard WHERE tree_id = ? AND branch_id = ?",
                       [tree_id, nid]).fetchone()
    p = inc[0] if inc else None
    return NodeRecord(
        tree_id=tree_id, cultivar=name, state_version=version, node_id=nid, kind=kind, item_id=item,
        is_root=(kind == "branch" and p is None),
        orchard_id=p["orchard_id"] if p else None, parent_branch_id=p["branch_id"] if p else None,
        edge_ix=p["edge_ix"] if p else None, edge_coph_distance=p["coph_distance"] if p else None,
        height=kids[1] if kind == "branch" else None, child_count=int(kids[0]),
        labels=node_labels(con, tree_id, nid), card=node_card(con, tree_id, nid),
    )


def node_children(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, node_id: Any) -> list[dict[str, Any]]:
    """``v_orchard`` rows hanging from a branch (empty for a leaf), ordered by edge_ix, node_id."""
    tree_id, _, _ = _tree(con, cultivar)
    nid = _uuids([node_id])[0]
    _node_kind(con, tree_id, nid)
    return _orchard_rows(con, f"SELECT {', '.join(_V_ORCHARD_COLS)} FROM treedb.v_orchard "
                              "WHERE tree_id = ? AND branch_id = ? ORDER BY edge_ix, node_id", [tree_id, nid])


def node_edges(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, node_id: Any) -> dict[str, Any]:
    """The node's branch edges: its incoming row (``parent``, None for a root) and its child rows."""
    tree_id, _, _ = _tree(con, cultivar)
    nid = _uuids([node_id])[0]
    _node_kind(con, tree_id, nid)
    parent = _orchard_rows(con, f"SELECT {', '.join(_V_ORCHARD_COLS)} FROM treedb.v_orchard "
                                "WHERE tree_id = ? AND node_id = ? ORDER BY orchard_id", [tree_id, nid])
    if len(parent) > 1:
        raise ReadError(f"node {nid} has {len(parent)} incoming rows (INV-6)")
    return {"parent": parent[0] if parent else None, "children": node_children(con, tree_id, nid)}


def node_labels(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, node_id: Any) -> dict[str, str]:
    """``{label_set: label}`` of a node (labels follow node identity, INV-11)."""
    tree_id, _, _ = _tree(con, cultivar)
    nid = _uuids([node_id])[0]
    return {s: lab for s, lab in con.execute(
        "SELECT label_set, label FROM treedb.node_labels WHERE tree_id = ? AND node_id = ? ORDER BY label_set",
        [tree_id, nid]).fetchall()}


def node_card(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, node_id: Any) -> Any:
    """The node's card (parsed JSON), or None."""
    tree_id, _, _ = _tree(con, cultivar)
    nid = _uuids([node_id])[0]
    row = con.execute("SELECT card FROM treedb.node_cards WHERE tree_id = ? AND node_id = ?", [tree_id, nid]).fetchone()
    return json.loads(row[0]) if row else None


def node_ancestry(con: duckdb.DuckDBPyConnection, cultivar: str | uuid.UUID, node_id: Any) -> list[tuple]:
    """``ancestry_paths`` from the node's incoming row; ``[]`` for a root (it has no row, INV-7)."""
    res = resolve_nodes(con, cultivar, [node_id])
    if res.unknown_nodes:
        raise NodeNotFoundError(f"{res.unknown_nodes[0]} is not a node of {res.cultivar!r}")
    return ancestry_paths(con, res.orchard_ids)


__all__ = [
    "ANCESTRY_COLUMNS", "ANCESTRY_TYPES", "COMPRESSED_COLUMNS", "COMPRESSED_TYPES", "STAGE_MACROS",
    "AncestryCycleError", "Component", "EntryResolution", "MixedCultivarError", "ItemReadResult", "MissingItemsError", "NodeNotFoundError", "NodeRecord",
    "NodeResolution", "ReadError", "Stage",
    "ancestry_paths", "ancestry_paths_for_items", "compressed_linkage", "compressed_subtree",
    "compressed_subtree_for_items", "cultivar_version", "ensure_installed", "in_transaction", "snapshot", "full_linkage", "get_node", "indexed_linkage", "inspect_compressed_subtree",
    "install", "node_ancestry", "node_card", "node_children", "node_edges", "node_labels", "path_closure",
    "read_sql_version", "resolve_items", "resolve_nodes", "tree_info",
]
