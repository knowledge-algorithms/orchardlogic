"""DuckDB store: open a connection, install the TreeDB schema, record its version.

Phase P3A. The schema is ``src/treedb/sql/schema.sql`` (a port of the installed
``graph.*`` Orchard objects into schema ``treedb``, D-006). Porting decisions are
in ``docs/PORTING_NOTES.md``.

* ``connect(None)`` or ``connect(":memory:")`` opens an in-memory database;
  ``connect(path)`` opens (or creates) a file database. Both install the schema
  unless ``read_only=True``.
* ``install_schema(con)`` is idempotent: it can run on every open.
* ``schema_version(con)`` returns the recorded version, or ``None``.
* ``integrity_violations(con)`` checks the references that PostgreSQL enforces
  with foreign keys and that TreeDB does not declare (PORTING_NOTES "FK-1").
* ``get_documents(con, item_ids)`` reads ``treedb.documents`` (D-027), reporting
  missing ids.
* Every timestamp column is a UTC ``TIMESTAMP`` (D-017 B, PORTING_NOTES
  "TS-1"), so ``SELECT *`` on any table fetches into Python without ``pytz``.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import duckdb

#: Schema name (D-006).
SCHEMA = "treedb"

#: Version of ``sql/schema.sql``. Bump when the DDL changes incompatibly.
#:
#: * 1: P3A.
#: * 2: P3A-G2. ``created_at`` / ``installed_at`` changed from a time-zone-aware
#:   type to UTC ``TIMESTAMP`` (D-017 B, PORTING_NOTES TS-1). ``CREATE TABLE IF
#:   NOT EXISTS`` cannot change an existing column, so a v1 file is refused by
#:   ``install_schema`` rather than silently kept with the old type.
#: * 3: P6B-2 (D-027). Adds ``treedb.documents``. The change is additive, so a v2
#:   file is migrated in place by ``install_schema`` (tested); v1 is still refused.
#: * 4: Data Orchard (docs/DATA_ORCHARD_V4.md section 5, D-042). Adds the perspective, enricher,
#:   cultivar_*, vector, enrichment, cut, ledger and checkpoint tables and two views. Additive, so
#:   v2 and v3 files are migrated in place (irreversible for v3 code).
SCHEMA_VERSION = 4

#: Versions ``install_schema`` upgrades in place, because the only changes since are new tables/views.
_MIGRATABLE_FROM = (2, 3)

#: UTC wall-clock ``TIMESTAMP`` of the transaction start (D-017 B, TS-1). Same
#: expression as the ``created_at`` defaults in ``sql/schema.sql``.
UTC_NOW_SQL = "CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP)"

_VERSION_DDL = f"""
CREATE TABLE IF NOT EXISTS treedb.schema_version (
    version      INTEGER   NOT NULL,
    installed_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW_SQL},
    CONSTRAINT schema_version_pkey PRIMARY KEY (version)
)
"""


class SchemaVersionError(RuntimeError):
    """The database holds a TreeDB schema of a different version."""


def schema_sql() -> str:
    """Text of ``treedb/sql/schema.sql``."""
    return resources.files("orchardlogic.treedb").joinpath("sql/schema.sql").read_text(encoding="utf-8")


def connect(database: str | Path | None = None, *, read_only: bool = False,
            migrate: bool = False) -> duckdb.DuckDBPyConnection:
    """Open an in-memory (``None`` / ``":memory:"``) or file-backed database.

    The schema is installed (idempotently) unless ``read_only`` is set.

    A file with an older migratable schema (v2, v3) is refused with ``SchemaVersionError``
    unless ``migrate=True``; then the file (and its ``.wal``) is first copied to
    ``<file>.v<old>.bak`` and migrated in place. The migration is irreversible for old code.
    Fresh and empty files get the current schema without any flag.

    A file whose stem is ``treedb`` (any case) is refused: DuckDB names the
    default catalog after the file stem, and a catalog named ``treedb`` makes
    every ``treedb.<table>`` reference ambiguous ("Ambiguous reference to
    catalog or schema"; test ``test_file_named_like_schema_is_refused``).
    """
    if database is None or str(database) == ":memory:":
        con = duckdb.connect(":memory:", read_only=False)
    else:
        path = Path(database)
        if path.stem.lower() == SCHEMA:
            raise ValueError(
                f"database file stem {path.stem!r} collides with schema {SCHEMA!r}; "
                "choose another file name"
            )
        con = duckdb.connect(str(path), read_only=read_only)
        if not read_only:
            found = schema_version(con)
            if found in _MIGRATABLE_FROM:
                con.close()
                if not migrate:
                    raise SchemaVersionError(
                        f"{path} has treedb schema v{found}; this code expects v{SCHEMA_VERSION}. "
                        "Open it with migrate=True (connect(path, migrate=True) / TreeDB(path, migrate=True)); "
                        f"that first copies the file to {path.name}.v{found}.bak and then upgrades it in place "
                        "(irreversible for older code).")
                backup = path.with_name(f"{path.name}.v{found}.bak")
                shutil.copy2(path, backup)
                wal = path.with_name(path.name + ".wal")
                if wal.exists():
                    shutil.copy2(wal, backup.with_name(backup.name + ".wal"))
                con = duckdb.connect(str(path))
    if not read_only:
        try:
            install_schema(con)
        except BaseException:
            con.close()          # e.g. SchemaVersionError on a v1 file: do not leak the handle
            raise
    return con


class StoreBusyError(RuntimeError):
    """The database file stayed locked by another process for the whole retry window (A7)."""


def connect_with_retry(database: str | Path, *, read_only: bool = False,
                       timeout_s: float = 2.0, migrate: bool = False) -> duckdb.DuckDBPyConnection:
    """``connect`` that retries on a cross-process file lock with backoff, up to ``timeout_s`` in total.

    A DuckDB file is open read-write by one process or read-only by several, never both (A7).
    Raises ``StoreBusyError`` when the lock is not released in time; other errors pass through.
    """
    deadline = time.monotonic() + timeout_s
    delay = 0.01
    while True:
        try:
            return connect(database, read_only=read_only, migrate=migrate)
        except duckdb.IOException as exc:
            msg = str(exc).lower()
            if "lock" not in msg and "being used by another process" not in msg:
                raise
            if time.monotonic() + delay > deadline:
                raise StoreBusyError(f"{database} is locked by another process (waited {timeout_s}s)") from exc
            time.sleep(delay)
            delay = min(delay * 2, 0.25)


def schema_version(con: duckdb.DuckDBPyConnection) -> int | None:
    """Recorded schema version, or ``None`` when the schema is not installed."""
    exists = con.execute(
        "SELECT count(*) FROM duckdb_tables() "
        "WHERE database_name = current_database() AND schema_name = ? AND table_name = 'schema_version'",
        [SCHEMA],
    ).fetchone()[0]
    if not exists:
        return None
    rows = con.execute("SELECT version FROM treedb.schema_version ORDER BY version").fetchall()
    if not rows:
        return None
    if len(rows) > 1:
        raise SchemaVersionError(f"several schema versions recorded: {[r[0] for r in rows]}")
    return int(rows[0][0])


def install_schema(con: duckdb.DuckDBPyConnection) -> int:
    """Install ``schema.sql`` and record ``SCHEMA_VERSION``. Idempotent.

    Runs in one transaction. A v2 database is migrated to v3 in the same
    transaction (``documents`` is created, the version row is replaced). Any
    other recorded version raises ``SchemaVersionError`` (v1 changed column
    types, which ``CREATE TABLE IF NOT EXISTS`` cannot do).
    """
    found = schema_version(con)
    migrate = found in _MIGRATABLE_FROM
    if found is not None and found != SCHEMA_VERSION and not migrate:
        raise SchemaVersionError(f"database has treedb schema v{found}; this code expects v{SCHEMA_VERSION}")
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(schema_sql())
        con.execute(_VERSION_DDL)
        if migrate:
            con.execute("DELETE FROM treedb.schema_version WHERE version = ?", [found])
        con.execute(
            "INSERT INTO treedb.schema_version (version) VALUES (?) ON CONFLICT (version) DO NOTHING",
            [SCHEMA_VERSION],
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Integrity checks that replace the PostgreSQL foreign keys (PORTING_NOTES FK-1)
# ---------------------------------------------------------------------------

#: name -> query returning offending rows. Each mirrors one PG FK or one
#: TreeDB reference (labels, cards, items).
_INTEGRITY_QUERIES: dict[str, str] = {
    # PG orchard_branch_id_fkey (orchard.sql:7, 014:116)
    "orchard.branch_id -> tree_branches.branch_id": """
        SELECT o.id FROM treedb.orchard o
        LEFT JOIN treedb.tree_branches b USING (branch_id)
        WHERE b.branch_id IS NULL
    """,
    # PG tree_branches_tree_id_fkey (tree_branches.sql:8, 014:107)
    "tree_branches.tree_id -> tree_registry.id": """
        SELECT b.branch_id FROM treedb.tree_branches b
        LEFT JOIN treedb.tree_registry r ON r.id = b.tree_id
        WHERE r.id IS NULL
    """,
    # PG tree_registry_parent_tree_id_fkey (tree_registry.sql:8, 017:65)
    "tree_registry.parent_tree_id -> tree_registry.id": """
        SELECT c.id FROM treedb.tree_registry c
        LEFT JOIN treedb.tree_registry p ON p.id = c.parent_tree_id
        WHERE c.parent_tree_id IS NOT NULL AND p.id IS NULL
    """,
    # TreeDB: a leaf row (edge_ix 1) names a registered item.
    "orchard leaf node_id -> item_registry.item_uuid": """
        SELECT o.id FROM treedb.orchard o
        LEFT JOIN treedb.item_registry i ON i.item_uuid = o.node_id
        WHERE o.edge_ix = 1 AND i.item_uuid IS NULL
    """,
    # TreeDB: labels and cards name a node of their own tree.
    "node_labels (tree_id, node_id) -> node of that tree": """
        SELECT l.tree_id, l.node_id, l.label_set FROM treedb.node_labels l
        WHERE NOT EXISTS (SELECT 1 FROM treedb.tree_branches b
                          WHERE b.tree_id = l.tree_id AND b.branch_id = l.node_id)
          AND NOT EXISTS (SELECT 1 FROM treedb.v_orchard v
                          WHERE v.tree_id = l.tree_id AND v.node_id = l.node_id)
    """,
    "node_cards (tree_id, node_id) -> node of that tree": """
        SELECT c.tree_id, c.node_id FROM treedb.node_cards c
        WHERE NOT EXISTS (SELECT 1 FROM treedb.tree_branches b
                          WHERE b.tree_id = c.tree_id AND b.branch_id = c.node_id)
          AND NOT EXISTS (SELECT 1 FROM treedb.v_orchard v
                          WHERE v.tree_id = c.tree_id AND v.node_id = c.node_id)
    """,
    # TreeDB (D-027): a document names a registered item.
    "documents.item_id -> item_registry.item_id": """
        SELECT d.item_id FROM treedb.documents d
        LEFT JOIN treedb.item_registry i ON i.item_id = d.item_id
        WHERE i.item_id IS NULL
    """,
    # Schema v4 references (docs/DATA_ORCHARD_V4.md section 5).
    "cultivar_tree.tree_id -> tree_registry.id": """
        SELECT t.tree_id FROM treedb.cultivar_tree t
        LEFT JOIN treedb.tree_registry r ON r.id = t.tree_id WHERE r.id IS NULL
    """,
    "cultivar_node.tree_id -> tree_registry.id": """
        SELECT DISTINCT n.tree_id FROM treedb.cultivar_node n
        LEFT JOIN treedb.tree_registry r ON r.id = n.tree_id WHERE r.id IS NULL
    """,
    "cut_node (tree_id, cut_name) -> cut": """
        SELECT DISTINCT n.tree_id, n.cut_name FROM treedb.cut_node n
        LEFT JOIN treedb.cut c ON c.tree_id = n.tree_id AND c.cut_name = n.cut_name WHERE c.tree_id IS NULL
    """,
    "item_vector (space, content_sha256) -> vector": """
        SELECT v.item_id, v.space FROM treedb.item_vector v
        LEFT JOIN treedb.vector x ON x.space = v.space AND x.content_sha256 = v.content_sha256
        WHERE x.space IS NULL
    """,
    "item_enrichment.enricher_id -> enricher": """
        SELECT e.item_id, e.enricher_id FROM treedb.item_enrichment e
        LEFT JOIN treedb.enricher x ON x.enricher_id = e.enricher_id WHERE x.enricher_id IS NULL
    """,
    "node_enrichment.enricher_id -> enricher": """
        SELECT e.tree_id, e.node_key, e.enricher_id FROM treedb.node_enrichment e
        LEFT JOIN treedb.enricher x ON x.enricher_id = e.enricher_id WHERE x.enricher_id IS NULL
    """,
    "cultivar_alias.tree_id -> tree_registry.id": """
        SELECT a.alias FROM treedb.cultivar_alias a
        LEFT JOIN treedb.tree_registry r ON r.id = a.tree_id WHERE r.id IS NULL
    """,
}


def integrity_violations(con: duckdb.DuckDBPyConnection) -> dict[str, list[tuple]]:
    """Offending rows per undeclared reference; empty dict when all hold."""
    out: dict[str, list[tuple]] = {}
    for name, sql in _INTEGRITY_QUERIES.items():
        rows = con.execute(sql).fetchall()
        if rows:
            out[name] = rows
    return out


# ---------------------------------------------------------------------------
# Documents (D-027)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentsResult:
    """``get_documents`` result: the records found, and the ids that have none (D-022)."""

    documents: dict[str, dict[str, Any]]   # item_id -> {"title", "text", "metadata"}
    missing_items: tuple[str, ...]         # requested ids with no document row, in request order

    @property
    def missing_count(self) -> int:
        return len(self.missing_items)


def _record(row: tuple) -> dict[str, Any]:
    return {"title": row[1], "text": row[2], "metadata": None if row[3] is None else json.loads(row[3])}


def get_documents(
    con: duckdb.DuckDBPyConnection, item_ids: Sequence[str] | None = None, *, on_missing: str = "report"
) -> DocumentsResult:
    """Documents for ``item_ids`` (every row when ``None``). ``metadata`` is parsed JSON.

    Ids without a row are listed in ``missing_items``, never dropped silently
    (D-022); ``on_missing="error"`` raises ``KeyError`` naming them instead.
    """
    if on_missing not in ("report", "error"):
        raise ValueError("on_missing must be 'report' or 'error'")
    if item_ids is None:
        rows = con.execute("SELECT item_id, title, text, metadata FROM treedb.documents ORDER BY item_id").fetchall()
        return DocumentsResult({r[0]: _record(r) for r in rows}, ())
    wanted = list(dict.fromkeys(item_ids))
    found: dict[str, dict[str, Any]] = {}
    if wanted:
        rows = con.execute(
            "SELECT item_id, title, text, metadata FROM treedb.documents WHERE item_id IN (SELECT unnest(?))",
            [wanted],
        ).fetchall()
        found = {r[0]: _record(r) for r in rows}
    missing = tuple(i for i in wanted if i not in found)
    if missing and on_missing == "error":
        raise KeyError(f"no document for {len(missing)} item(s): {list(missing)}")
    return DocumentsResult({i: found[i] for i in wanted if i in found}, missing)
