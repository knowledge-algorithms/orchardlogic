"""treedb.documents and schema v3 (P6B-2, D-027). Invariants: INV-1 (item ids), INV-20 /
D-022 (missing ids reported, never dropped), FK-1 (references checked by integrity_violations)."""

from __future__ import annotations

import pytest

from ingest_helpers import load_all
from orchardlogic.treedb import store


def _add(con, item_id, text="t", title=None, metadata=None):
    con.execute("INSERT INTO treedb.documents VALUES (?, ?, ?, ?)", [item_id, title, text, metadata])


def test_documents_table_columns_and_key():
    con = store.connect()
    cols = con.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_schema = 'treedb' AND table_name = 'documents' ORDER BY ordinal_position").fetchall()
    assert cols == [("item_id", "VARCHAR", "NO"), ("title", "VARCHAR", "YES"),
                    ("text", "VARCHAR", "NO"), ("metadata", "JSON", "YES")]
    _add(con, "a")
    with pytest.raises(Exception):          # primary key on item_id
        _add(con, "a")


def test_integrity_flags_document_without_registered_item():
    con, _ = load_all("F1_tiny3")
    item = con.execute("SELECT item_id FROM treedb.item_registry LIMIT 1").fetchone()[0]
    _add(con, item)
    assert store.integrity_violations(con) == {}
    _add(con, "not-registered")
    assert store.integrity_violations(con) == {"documents.item_id -> item_registry.item_id": [("not-registered",)]}


def test_get_documents_reports_missing_and_never_drops():
    con, _ = load_all("F1_tiny3")
    items = [r[0] for r in con.execute("SELECT item_id FROM treedb.item_registry ORDER BY item_id").fetchall()]
    _add(con, items[0], text="hello", title="T", metadata='{"k": [1, 2]}')
    got = store.get_documents(con, [items[0], "ghost", items[1], "ghost"])
    assert got.documents == {items[0]: {"title": "T", "text": "hello", "metadata": {"k": [1, 2]}}}
    assert got.missing_items == ("ghost", items[1]) and got.missing_count == 2
    with pytest.raises(KeyError, match="ghost"):
        store.get_documents(con, ["ghost"], on_missing="error")
    assert store.get_documents(con, []).documents == {}
    assert list(store.get_documents(con).documents) == [items[0]]


def test_v2_file_is_migrated_in_place_and_keeps_data(tmp_path):
    """Schema v3 only adds a table, so a v2 file is migrated (not refused); data survives."""
    path = tmp_path / "v2.duckdb"
    con, _ = load_all("F1_tiny3", store.connect(path))
    before = con.execute("SELECT count(*) FROM treedb.orchard").fetchone()[0]
    con.execute("DROP TABLE treedb.documents")
    con.execute("UPDATE treedb.schema_version SET version = 2")
    con.close()
    with pytest.raises(store.SchemaVersionError, match="migrate=True"):
        store.connect(path)
    con = store.connect(path, migrate=True)
    assert (tmp_path / "v2.duckdb.v2.bak").exists()
    assert store.schema_version(con) == store.SCHEMA_VERSION == 4
    assert con.execute("SELECT version FROM treedb.schema_version").fetchall() == [(4,)]
    assert con.execute("SELECT count(*) FROM treedb.orchard").fetchone()[0] == before
    assert con.execute("SELECT count(*) FROM treedb.documents").fetchone() == (0,)
    assert store.integrity_violations(con) == {}
    con.close()


@pytest.mark.parametrize("old", [1, 0, 5])
def test_other_versions_are_refused(old, tmp_path):
    path = tmp_path / "old.duckdb"
    con = store.connect(path)
    con.execute("UPDATE treedb.schema_version SET version = ?", [old])
    con.close()
    with pytest.raises(store.SchemaVersionError, match=f"v{old}"):
        store.connect(path)
