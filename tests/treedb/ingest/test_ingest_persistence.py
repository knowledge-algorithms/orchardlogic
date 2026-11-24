"""In-memory and file-backed stores are equivalent; a file store persists (Arch §8.10).

P3A acceptance criterion 5.
"""

from __future__ import annotations

import pytest

from ingest_helpers import FIXTURES, load_all, snapshot_all
from orchardlogic.treedb import ingest, store

TABLES = ("tree_registry", "tree_z", "tree_branches", "orchard", "item_registry", "node_labels",
          "node_cards", "orchard_edge_types", "schema_version")


def _raw_counts(con):
    return {t: con.execute(f"SELECT count(*) FROM treedb.{t}").fetchone()[0] for t in TABLES}


@pytest.mark.parametrize("stem", sorted(FIXTURES))
def test_memory_and_file_give_identical_canonical_results(stem, tmp_path):
    names = [cv["name"] for cv in FIXTURES[stem]["cultivars"]]
    mem, _ = load_all(stem, store.connect())
    fil, _ = load_all(stem, store.connect(tmp_path / f"{stem}.duckdb"))
    assert snapshot_all(mem, names) == snapshot_all(fil, names)
    assert _raw_counts(mem) == _raw_counts(fil)
    fil.close()


def test_close_and_reopen_keeps_schema_and_data(tmp_path):
    path = tmp_path / "store.duckdb"
    con, results = load_all("F5_twocult8", store.connect(path))
    names = list(results)
    before = snapshot_all(con, names)
    raw_before = con.execute(
        "SELECT id, branch_id, node_id, edge_ix, coph_distance FROM treedb.orchard ORDER BY id").fetchall()
    counts = _raw_counts(con)
    con.close()

    con = store.connect(path)                      # reopen runs install_schema again (idempotent)
    assert store.schema_version(con) == store.SCHEMA_VERSION
    assert _raw_counts(con) == counts             # nothing duplicated by the re-install
    assert snapshot_all(con, names) == before
    assert con.execute(
        "SELECT id, branch_id, node_id, edge_ix, coph_distance FROM treedb.orchard ORDER BY id").fetchall() == raw_before
    # the reopened store is still writable: import one more cultivar over the shared items
    fx = FIXTURES["F5_twocult8"]
    cv = fx["cultivars"][0]
    extra = ingest.import_cultivar(con, cv["z"], fx["item_ids"], "domain_copy")
    assert extra.orchard_rows == 2 * (len(fx["item_ids"]) - 1)
    assert con.execute("SELECT count(*) FROM treedb.item_registry").fetchone() == (len(fx["item_ids"]),)
    assert store.integrity_violations(con) == {}
    con.close()

    con = store.connect(path, read_only=True)     # read-only reopen sees the committed import
    assert con.execute("SELECT count(*) FROM treedb.tree_registry").fetchone() == (3,)
    con.close()


def test_install_schema_is_idempotent():
    con = store.connect()
    store.install_schema(con)
    store.install_schema(con)
    assert con.execute("SELECT count(*) FROM treedb.orchard_edge_types").fetchone() == (3,)
    assert con.execute("SELECT version FROM treedb.schema_version").fetchall() == [(store.SCHEMA_VERSION,)]


def test_schema_version_mismatch_is_refused():
    con = store.connect()
    con.execute("UPDATE treedb.schema_version SET version = 999")
    with pytest.raises(store.SchemaVersionError):
        store.install_schema(con)


def test_schema_v1_file_is_refused(tmp_path):
    """TS-1 / D-017 (B): schema v2 retyped the timestamp columns to UTC TIMESTAMP. CREATE TABLE
    IF NOT EXISTS cannot retype a P3A (v1) file's columns, so such a file is refused, not
    silently kept with time-zone-aware columns."""
    assert store.SCHEMA_VERSION == 4   # Data Orchard v4 (was 3: documents table, D-027)
    path = tmp_path / "v1.duckdb"
    con = store.connect(path)
    con.execute("UPDATE treedb.schema_version SET version = 1")
    con.close()
    with pytest.raises(store.SchemaVersionError, match="v1"):
        store.connect(path)


def test_schema_version_absent_on_empty_db():
    import duckdb
    assert store.schema_version(duckdb.connect()) is None


@pytest.mark.parametrize("stem", ["treedb", "TreeDB"])
def test_file_named_like_schema_is_refused(stem, tmp_path):
    """FILE-1: DuckDB names the catalog after the file stem; `treedb.x` would be ambiguous."""
    with pytest.raises(ValueError, match="collides with schema"):
        store.connect(tmp_path / f"{stem}.duckdb")


def test_catalog_schema_collision_reproducer(tmp_path):
    """FILE-1 reproducer: the DuckDB error the refusal avoids."""
    import duckdb
    con = duckdb.connect(str(tmp_path / "treedb.duckdb"))
    con.execute("CREATE SCHEMA treedb")
    with pytest.raises(duckdb.BinderException, match="Ambiguous reference to catalog or schema"):
        con.execute("CREATE TABLE treedb.x (i INT)")
    con.close()
