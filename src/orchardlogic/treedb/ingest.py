"""Import a cultivar (Z-array + ordered item ids) into the DuckDB Orchard store.

Phase P3A. Port of the PostgreSQL import path (architecture §7 "Import tree",
§8.1). The call sequence mirrors ``OrchardRepo`` / ``TreeLoadingService``:

=====  ==========================================  =============================================
step   TreeDB (this module)                         PostgreSQL reference
=====  ==========================================  =============================================
0      ``_register_items``                          (TreeDB addition: items registry, Arch §2)
1      ``_register_tree``                           ``repo.py:92-111`` ``register_tree``
2      ``tree_z_rows`` + ``_bulk_load_tree_z``      ``loader.py:134-159`` + ``repo.py:132-174``
3      ``_create_orchard``                          ``graph.create_orchard`` (installed text:
                                                    ``reference/pg_installed/graph/function/
                                                    create_orchard_uuid_uuid.sql:1-35``;
                                                    migration ``018:197-234``)
4      ``_insert_labels`` / ``_insert_cards``       (TreeDB addition: INV-11)
(5)    *not done*: ``repo.cleanup_tree_z``          ``repo.py:180-196``, ``loader.py:198``
=====  ==========================================  =============================================

All steps run inside ONE DuckDB transaction opened by ``import_cultivar``; the
plpgsql function body (step 3) is kept as separate SQL statements, each
commented with the installed line numbers. Divergences from PostgreSQL are
listed in ``docs/PORTING_NOTES.md`` (IDs in brackets below).

**Staging cleanup.** The original Z stays in ``treedb.tree_z`` (step 5 is not
run). A later cleanup of a tree's ``tree_z`` rows IS allowed by the Orchard
semantics: after import, ``treedb.orchard`` / ``treedb.tree_branches`` are the
source of truth (Arch §2, ``repo.py:22-26``) and no Orchard table or the root id
depends on staging (the root id lives in ``tree_registry.root_branch_id``,
[ROOT-1]). A cleanup loses the original Z rows (the internal-node ids
survive as ``tree_branches.branch_id``) and makes ``v_tree_z`` /
``v_tree_map`` empty for that tree; the registry keeps ``z_sha256`` and
``item_order``, which identify (but cannot rebuild) the original Z. TreeDB
itself never cleans up staging in P3A.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np

from orchardlogic.treedb.fixtures import FixtureError, canonical_json, item_uuid, node_keys, validate_z

# Edge type codes (graph.orchard_edge_types.ix, 014:64-71; orchard/types.py:23-25).
EDGE_LEAF = 1
EDGE_INTERNAL_BRANCH = 2
EDGE_EXTERNAL_GRAFT = 3

# INV-2: the Orchard tree_id string (orchard/identity.py:validate_tree_id).
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")


class IngestError(ValueError):
    """Invalid input to ``import_cultivar``. Nothing is written."""


@dataclass(frozen=True)
class TreeZRow:
    """One ``treedb.tree_z`` row; same fields as ``orchard/types.py:55-75``."""

    branch_id: int
    l_node: float
    r_node: float
    coph_distance: float
    doc_count: float
    l_doc_id: uuid.UUID | None
    r_doc_id: uuid.UUID | None
    tree_id: uuid.UUID
    l_leaf: bool
    r_leaf: bool


@dataclass(frozen=True)
class ImportedCultivar:
    """Result of ``import_cultivar``."""

    tree_id: uuid.UUID          # treedb.tree_registry.id
    name: str                   # treedb.tree_registry.name (INV-2)
    root_branch_id: uuid.UUID   # stored root id [ROOT-1]
    orchard_rows: int           # create_orchard's return value (GET DIAGNOSTICS ROW_COUNT)
    state_version: int          # D-007, 0 at import
    node_ids: dict[int, uuid.UUID]  # SciPy node index -> orchard node identity (leaf item UUID / branch id)


# ---------------------------------------------------------------------------
# Pure helpers (no database)
# ---------------------------------------------------------------------------

def z_sha256(z: Sequence[Sequence[float]]) -> str:
    """Provenance hash of a Z-array: sha256 of ``canonical_json`` of its rows as floats."""
    rows = [[float(v) for v in row] for row in z]
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


def tree_z_rows(
    z: Sequence[Sequence[float]], item_ids: Sequence[str], tree_id: uuid.UUID
) -> tuple[list[TreeZRow], dict[int, uuid.UUID]]:
    """Z -> ``tree_z`` rows, mirroring ``loader.py:134-159`` statement by statement.

    Doc-id convention as installed:

    * leaf ``i`` -> its document UUID. In TreeDB that is the item's CONTRACTS
      UUID ``item_uuid(item_ids[i])`` (INV-5) [IR-1].
    * every internal node that appears as a child in Z (all except the root)
      -> a fresh ``uuid.uuid4()`` (loader.py:140-142). It becomes that node's
      ``tree_branches.branch_id`` through ``v_tree_map`` / ``create_orchard``.
    * the root (index ``2n-2``) never appears in Z columns 0-1 and gets no
      staging doc id; its id comes from ``tree_registry.root_branch_id``
      [ROOT-1].
    * ``l_leaf`` / ``r_leaf`` are the loader's leaf flags (loader.py:156-157).
      ``v_tree_z`` recomputes them from ``n_docs`` and ignores these columns.

    Returns the rows and the map SciPy node index -> doc id (root excluded).
    """
    n_docs = len(item_ids)
    zarr = np.asarray(z, dtype=float)
    # loader.py:136-138
    doc_to_z: dict[int, dict[str, Any]] = {}
    for i, item in enumerate(item_ids):
        doc_to_z[i] = {"doc_id": item_uuid(item), "leaf": True}
    # loader.py:140-142
    b_unique = np.unique(zarr[:, :2])
    for zmid in b_unique[b_unique >= n_docs]:
        doc_to_z[int(zmid)] = {"doc_id": uuid.uuid4(), "leaf": False}
    # loader.py:144-159
    rows: list[TreeZRow] = []
    for branch_id, (l_node, r_node, coph_dist, doc_count) in enumerate(zarr):
        rows.append(
            TreeZRow(
                branch_id=branch_id,
                l_node=float(l_node),
                r_node=float(r_node),
                coph_distance=float(coph_dist),
                doc_count=float(doc_count),
                l_doc_id=doc_to_z[int(l_node)]["doc_id"],
                r_doc_id=doc_to_z[int(r_node)]["doc_id"],
                tree_id=tree_id,
                l_leaf=bool(doc_to_z[int(l_node)]["leaf"]),
                r_leaf=bool(doc_to_z[int(r_node)]["leaf"]),
            )
        )
    return rows, {k: v["doc_id"] for k, v in doc_to_z.items()}


def validate_cultivar(
    z: Any,
    item_ids: Sequence[str],
    name: str,
    labels: Mapping[str, Mapping[str, str]] | None,
    cards: Mapping[str, Any] | None,
) -> tuple[list[list[float]], list[str], dict[str, int]]:
    """Public pre-write validation: the first step of ``import_cultivar``, no store access.

    Input checks (INV-1, INV-2, INV-9, INV-10, INV-11). Raises ``IngestError``. Returns
    ``(z_rows, items, key_to_index)``. Callers that import several cultivars call this for
    every one first, so a bad input is found before the first write.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise IngestError(f"cultivar name must match {_NAME_RE.pattern}: {name!r}")
    items = list(item_ids)
    if not all(isinstance(i, str) and i and i.strip() == i for i in items):
        raise IngestError("item_ids must be non-empty strings without edge whitespace (INV-1)")
    if len(items) != len(set(items)):
        raise IngestError("item_ids must be unique within a cultivar (INV-1)")
    if len(items) < 2:
        raise IngestError("a cultivar needs at least 2 leaves (INV-9: one-leaf cultivars are Open)")
    try:
        validate_z(z, len(items), f"cultivar {name!r}")
    except FixtureError as exc:
        raise IngestError(str(exc)) from exc
    zarr = np.asarray(z, dtype=float)
    # is_valid_linkage accepts non-integral child indices (reproducer:
    # tests/ingest/test_ingest_atomicity.py::test_non_integral_child_passes_scipy_validation);
    # v_tree_z needs integral values.
    for col in (0, 1, 3):
        if not all(float(v).is_integer() for v in zarr[:, col]):
            raise IngestError(f"Z column {col} must hold integral values (INV-8)")
    z_rows = [[float(v) for v in row] for row in zarr]
    keys = node_keys(z_rows, items)
    key_to_index = {k: i for i, k in keys.items()}
    for set_name, mapping in (labels or {}).items():
        if not isinstance(set_name, str) or not set_name:
            raise IngestError("label set names must be non-empty strings")
        for key, label in mapping.items():
            if key not in key_to_index:
                raise IngestError(f"label set {set_name!r}: unknown node key {key!r} (INV-11)")
            if not isinstance(label, str):
                raise IngestError(f"label set {set_name!r}: label for {key!r} must be a string")
    for key in (cards or {}):
        if key not in key_to_index:
            raise IngestError(f"cards: unknown node key {key!r} (INV-11)")
    return z_rows, items, key_to_index


# ---------------------------------------------------------------------------
# Steps (each runs inside the transaction opened by import_cultivar)
# ---------------------------------------------------------------------------

def _bulk_insert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[tuple[str, str, Sequence[Any]]],
    suffix: str = "",
) -> None:
    """Insert rows given column-wise: ``[(column, duckdb_type, values), ...]``.

    One ``INSERT ... SELECT unnest(?), unnest(?), ...`` statement; DuckDB zips
    several ``unnest`` calls of one SELECT list element-wise. Same rows as the
    per-row ``executemany`` of ``repo.py:144-171``, about 20x faster for one AppWorld
    cultivar (3.5 s -> 0.15 s, measured during development) [BULK-1].
    """
    lengths = {len(v) for _, _, v in columns}
    if len(lengths) != 1:
        raise ValueError("bulk insert: columns differ in length")
    if lengths == {0}:
        return
    names = ", ".join(c for c, _, _ in columns)
    selects = ", ".join(f"unnest(CAST(? AS {t}[]))" for _, t, _ in columns)
    params = [[None if v is None else (str(v) if t == "UUID" else v) for v in vals] for _, t, vals in columns]
    con.execute(f"INSERT INTO {table} ({names}) SELECT {selects} {suffix}", params)


def _register_items(con: duckdb.DuckDBPyConnection, items: Sequence[str]) -> None:
    """Step 0 [IR-1]: add unseen items to the shared registry; existing rows are kept."""
    _bulk_insert(
        con, "treedb.item_registry",
        [("item_id", "VARCHAR", list(items)), ("item_uuid", "UUID", [item_uuid(i) for i in items])],
        suffix="ON CONFLICT (item_id) DO NOTHING",
    )
    # A pre-existing row must carry the CONTRACTS UUID (INV-5); never rewrite it silently.
    stored_rows = con.execute(
        "SELECT item_id, item_uuid FROM treedb.item_registry WHERE list_contains(?, item_id)",
        [list(items)],
    ).fetchall()
    for item, stored in stored_rows:
        if stored != item_uuid(item):
            raise IngestError(f"item_registry row for {item!r} has a non-CONTRACTS UUID {stored}")


def _register_tree(
    con: duckdb.DuckDBPyConnection,
    name: str,
    z_rows: Sequence[Sequence[float]],
    items: Sequence[str],
    source: Any,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Step 1: ``repo.py:92-111`` ``register_tree``.

    PG: ``INSERT INTO graph.tree_registry (vault_id) VALUES (:vault_id)
    RETURNING id, ...``. TreeDB has no vault (vault_id NULL [TR-1]) and adds
    the cultivar name, provenance and state_version columns [TR-2]. ``id`` and
    ``root_branch_id`` come from their ``gen_random_uuid()`` defaults.
    """
    row = con.execute(
        """
        INSERT INTO treedb.tree_registry (vault_id, name, z_sha256, item_order, source)
        VALUES (NULL, ?, ?, ?, ?)
        RETURNING id, root_branch_id
        """,
        [name, z_sha256(z_rows), list(items), None if source is None else canonical_json(source)],
    ).fetchone()
    return row[0], row[1]


def _bulk_load_tree_z(con: duckdb.DuckDBPyConnection, rows: Sequence[TreeZRow]) -> int:
    """Step 2: ``repo.py:132-174`` ``bulk_load_tree_z`` (same column list).

    ``coph_distance`` is stored exactly as in Z, NOT rounded: PG's
    ``tree_z.coph_distance`` is ``double precision`` (``pgi:table/tree_z.sql:5``)
    and keeps Z bit for bit (``precision_probe.json`` stage ``tree_z``). The
    only import rounding is in ``_create_orchard`` (co:18) [CO-4, D-012].
    """
    if not rows:
        return 0
    _bulk_insert(con, "treedb.tree_z", [
        ("branch_id", "BIGINT", [r.branch_id for r in rows]),
        ("l_node", "DOUBLE", [r.l_node for r in rows]),
        ("r_node", "DOUBLE", [r.r_node for r in rows]),
        ("coph_distance", "DOUBLE", [r.coph_distance for r in rows]),
        ("doc_count", "DOUBLE", [r.doc_count for r in rows]),
        ("l_doc_id", "UUID", [r.l_doc_id for r in rows]),
        ("r_doc_id", "UUID", [r.r_doc_id for r in rows]),
        ("tree_id", "UUID", [r.tree_id for r in rows]),
        ("l_leaf", "BOOLEAN", [r.l_leaf for r in rows]),
        ("r_leaf", "BOOLEAN", [r.r_leaf for r in rows]),
    ])
    return len(rows)


def _create_orchard(con: duckdb.DuckDBPyConnection, vault_id: uuid.UUID | None, tree_id: uuid.UUID) -> int:
    """Step 3: port of ``graph.create_orchard(_vault_id uuid, _tree_id uuid) RETURNS integer``.

    Installed text: ``reference/pg_installed/graph/function/create_orchard_uuid_uuid.sql``
    (cited as ``co:<line>``); identical to ``018:197-234`` (PG_INVENTORY §2).
    Must run inside the caller's transaction: plpgsql runs the body atomically
    in the caller's transaction, DuckDB has no procedural SQL [CO-1].
    Returns the number of orchard rows inserted (``co:32-33``).

    Edge distances are rounded to 15 significant digits exactly where PG
    rounds them, the ``co:18`` ``::numeric`` cast [CO-4, D-012 (A)].
    """
    # co:8-10   INSERT INTO graph.tree_registry (vault_id, id)
    #           VALUES (_vault_id, _tree_id) ON CONFLICT (id) DO NOTHING;
    # In TreeDB the row always exists (step 1), so this is the no-op branch;
    # a missing row would fail NOT NULL on `name` instead of inserting a bare
    # row [CO-2].
    con.execute(
        "INSERT INTO treedb.tree_registry (vault_id, id) VALUES (?, ?) ON CONFLICT (id) DO NOTHING",
        [vault_id, tree_id],
    )

    # co:12     DROP TABLE IF EXISTS tmp_orchard;
    # Qualified as temp.main so an unqualified name can never resolve to a
    # permanent table [CO-3].
    con.execute("DROP TABLE IF EXISTS temp.main.tmp_orchard")

    # co:13-24  CREATE TEMP TABLE tmp_orchard AS SELECT p.doc_id AS branch_id,
    #           c.doc_id AS node_id, CASE WHEN c.leaf THEN 1::int ELSE 2::int END AS edge_ix,
    #           c.coph_distance::numeric FROM graph.v_tree_map p INNER JOIN graph.v_tree_map c
    #           ON p.tree_id = c.tree_id AND p.z_mod = c.branch_id
    #           WHERE p.tree_id = _tree_id ORDER BY p.branch_id DESC;
    # co:18 `c.coph_distance::numeric` -> CAST(printf('%.15g', ...) AS DOUBLE)
    # [CO-4, D-012 (A), G2 2026-11-09]. This is THE ONLY rounding site of the
    # import, because it is the only place PostgreSQL's import path rounds:
    #   * PG float8 -> numeric formats with "%.*g", DBL_DIG = 15 significant
    #     digits (measured: docs/PG_BASELINE.md §2a,
    #     reference/golden/_meta/precision_probe.json, stage tmp_orchard);
    #     the co:29-30 INSERT turns the numeric back into float8 (orchard
    #     column type, pgi:table/orchard.sql:6), which DuckDB's VARCHAR ->
    #     DOUBLE cast reproduces (nearest double of the 15-digit decimal).
    #   * staging is NOT rounded: pgi:table/tree_z.sql:5 is double precision,
    #     filled by repo.py:132-174 with the loader's float (loader.py:151),
    #     and pgi:view/v_tree_z.sql:4 / v_tree_map.sql:8,17,39 pass it through
    #     as float8 (the probe's tree_z / v_tree_z / v_tree_map stages are
    #     bit-exact Z). _bulk_load_tree_z therefore stores Z unchanged.
    #   * no other import write carries a distance or height (tree_branches,
    #     tree_registry: pgi:table/tree_branches.sql, tree_registry.sql).
    # tmp_orchard keeps DOUBLE, not a decimal: DuckDB NUMERIC is DECIMAL(18,3)
    # and would truncate; the only reader of tmp_orchard is the co:29-30
    # INSERT into a DOUBLE column, so holding the already-converted double
    # gives the same orchard value. Explicit alias: PG names the cast column
    # `coph_distance`, DuckDB would not.
    # Explicit attach distances are NOT rounded (orchard_attach_node casts
    # numeric -> float8 only, pgi:function/orchard_attach_node_..._numeric.sql:28-31);
    # P3C must not reuse this expression there [CO-4].
    con.execute(
        """
        CREATE TEMP TABLE tmp_orchard AS
        SELECT
            p.doc_id                                                AS branch_id,
            c.doc_id                                                AS node_id,
            CASE WHEN c.leaf THEN 1::int ELSE 2::int END            AS edge_ix,
            CAST(printf('%.15g', c.coph_distance) AS DOUBLE)        AS coph_distance
        FROM treedb.v_tree_map p
        INNER JOIN treedb.v_tree_map c
            ON p.tree_id = c.tree_id
           AND p.z_mod   = c.branch_id
        WHERE p.tree_id = ?
        ORDER BY p.branch_id DESC
        """,
        [tree_id],
    )

    # co:26-27  INSERT INTO graph.tree_branches (branch_id, tree_id)
    #           SELECT DISTINCT branch_id, _tree_id FROM tmp_orchard;
    con.execute(
        """
        INSERT INTO treedb.tree_branches (branch_id, tree_id)
        SELECT DISTINCT branch_id, CAST(? AS UUID) FROM temp.main.tmp_orchard
        """,
        [tree_id],
    )

    # co:29-30  INSERT INTO graph.orchard (branch_id, node_id, edge_ix, coph_distance)
    #           SELECT branch_id, node_id, edge_ix, coph_distance FROM tmp_orchard;
    # co:32-33  GET DIAGNOSTICS v_rowcount = ROW_COUNT; RETURN v_rowcount;
    # DuckDB returns the inserted row count as the INSERT's result [CO-5].
    inserted = con.execute(
        """
        INSERT INTO treedb.orchard (branch_id, node_id, edge_ix, coph_distance)
        SELECT branch_id, node_id, edge_ix, coph_distance FROM temp.main.tmp_orchard
        """
    ).fetchone()[0]
    # tmp_orchard is left in place, as in PG (a session temp table) [CO-3].
    return int(inserted)


def _insert_labels(
    con: duckdb.DuckDBPyConnection,
    tree_id: uuid.UUID,
    labels: Mapping[str, Mapping[str, str]],
    key_to_index: Mapping[str, int],
    node_ids: Mapping[int, uuid.UUID],
) -> None:
    """Step 4 [LC-1]: labels keyed by node identity, source node key kept separately."""
    rows = [
        (node_ids[key_to_index[key]], set_name, label, key)
        for set_name, mapping in sorted(labels.items())
        for key, label in sorted(mapping.items())
    ]
    _bulk_insert(con, "treedb.node_labels", [
        ("tree_id", "UUID", [tree_id] * len(rows)),
        ("node_id", "UUID", [r[0] for r in rows]),
        ("label_set", "VARCHAR", [r[1] for r in rows]),
        ("label", "VARCHAR", [r[2] for r in rows]),
        ("source_node_key", "VARCHAR", [r[3] for r in rows]),
    ])


def _insert_cards(
    con: duckdb.DuckDBPyConnection,
    tree_id: uuid.UUID,
    cards: Mapping[str, Any],
    key_to_index: Mapping[str, int],
    node_ids: Mapping[int, uuid.UUID],
) -> None:
    """Step 4 [LC-1]: cards as canonical JSON text, keyed by node identity."""
    rows = [(node_ids[key_to_index[key]], canonical_json(card), key) for key, card in sorted(cards.items())]
    _bulk_insert(con, "treedb.node_cards", [
        ("tree_id", "UUID", [tree_id] * len(rows)),
        ("node_id", "UUID", [r[0] for r in rows]),
        ("card", "VARCHAR", [r[1] for r in rows]),
        ("source_node_key", "VARCHAR", [r[2] for r in rows]),
    ])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def import_cultivar(
    con: duckdb.DuckDBPyConnection,
    Z: Any,
    item_ids: Sequence[str],
    name: str,
    labels: Mapping[str, Mapping[str, str]] | None = None,
    cards: Mapping[str, Any] | None = None,
    source: Any = None,
) -> ImportedCultivar:
    """Import one cultivar atomically and return its registry identity.

    * ``Z``: SciPy linkage of shape ``(n-1, 4)`` (INV-8..10); leaf ``i`` is
      ``item_ids[i]`` (INV-9).
    * ``name``: the cultivar's human name (INV-2), unique in the store.
    * ``labels``: ``{label_set: {node_key: label}}`` and ``cards``:
      ``{node_key: json_object}``, keyed by the Orchard source node key, i.e.
      the ``member_`` key of the node in the imported tree (INV-3, INV-11).
      Stored against the node's identity (``node_id``); the key is kept in
      ``source_node_key``, never used as the branch id.
    * ``source``: optional JSON-serialisable description of the source
      artifact, stored as canonical JSON text in ``tree_registry.source``.

    The connection must not be inside a transaction: this function opens one
    (``BEGIN``) and commits it, or rolls it back on any error so that nothing
    is written. The original Z stays in ``treedb.tree_z``; a later cleanup is
    allowed (see the module docstring) but is never done here.
    """
    z_rows, items, key_to_index = validate_cultivar(Z, item_ids, name, labels, cards)
    n = len(items)
    con.execute("BEGIN TRANSACTION")
    try:
        _register_items(con, items)                                    # step 0
        tree_id, root_id = _register_tree(con, name, z_rows, items, source)  # step 1
        rows, doc_ids = tree_z_rows(z_rows, items, tree_id)            # step 2 (loader)
        _bulk_load_tree_z(con, rows)                                   # step 2 (repo)
        inserted = _create_orchard(con, None, tree_id)                 # step 3
        node_ids = dict(doc_ids)
        node_ids[2 * n - 2] = root_id                                  # INV-8 root index
        _insert_labels(con, tree_id, labels or {}, key_to_index, node_ids)  # step 4
        _insert_cards(con, tree_id, cards or {}, key_to_index, node_ids)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return ImportedCultivar(
        tree_id=tree_id,
        name=name,
        root_branch_id=root_id,
        orchard_rows=inserted,
        state_version=0,
        node_ids=node_ids,
    )


def import_fixture_cultivar(
    con: duckdb.DuckDBPyConnection, fixture: Mapping[str, Any], cultivar: str, source: Any = None
) -> ImportedCultivar:
    """Convenience: import one cultivar of a ``treedb_fixture_v1`` dict."""
    cv = next(c for c in fixture["cultivars"] if c["name"] == cultivar)
    items = list(cv.get("item_ids", fixture["item_ids"]))
    return import_cultivar(
        con, cv["z"], items, cultivar,
        labels=cv.get("label_sets") or None,
        cards=cv.get("cards") or None,
        source=source if source is not None else {"fixture": fixture["name"], "cultivar": cultivar},
    )


__all__ = [
    "EDGE_EXTERNAL_GRAFT", "EDGE_INTERNAL_BRANCH", "EDGE_LEAF",
    "ImportedCultivar", "IngestError", "TreeZRow",
    "import_cultivar", "import_fixture_cultivar", "tree_z_rows", "validate_cultivar", "z_sha256",
]
