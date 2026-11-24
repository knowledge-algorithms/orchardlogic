"""Schema v4, tx, registry import, vectors, documents, cutstore, ledger, checkpoints (docs/DATA_ORCHARD_V4.md)."""

from __future__ import annotations

import json
import shutil
import uuid

import duckdb
import numpy as np
import pytest

from cultivar_builder import SPACE, build_cultivar, perspective
from orchardlogic.treedb import checkpoints, cutstore, documents, enrichment, ledger, registry, store, tx, vectors
from orchardlogic.treedb.adapters import orchard_artifact
from orchardlogic.treedb.api import TreeDB
from orchardlogic.treedb.registry import CultivarIntegrityError, RegistryConflictError

V4_TABLES = ("perspective", "enricher", "cultivar_artifact", "cultivar_tree", "cultivar_alias", "cultivar_node",
             "vector", "item_vector", "item_enrichment", "node_enrichment", "cut", "cut_node", "ledger",
             "checkpoint")


@pytest.fixture
def cultivar(tmp_path):
    return build_cultivar(tmp_path / "cv")


@pytest.fixture
def con():
    c = store.connect()
    yield c
    c.close()


# --- schema -----------------------------------------------------------------------------------

def test_v4_ddl_parses_and_creates_every_table_and_view(con):
    names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables "
                                       "WHERE table_schema = 'treedb'").fetchall()}
    assert set(V4_TABLES) | {"v_item_time", "v_cut_items"} <= names
    assert store.SCHEMA_VERSION == 4 and store.schema_version(con) == 4
    # the raw DDL also runs on a bare DuckDB 2.x connection after the base schema (A1 probe)
    con.execute(store.schema_sql())


@pytest.mark.parametrize("old", [2, 3])
def test_older_files_migrate_in_place(tmp_path, old):
    path = tmp_path / "old.duckdb"
    c = store.connect(path)
    c.execute("INSERT INTO treedb.item_registry (item_id, item_uuid) VALUES ('keep', ?)", [str(uuid.uuid4())])
    for t in V4_TABLES:
        c.execute(f"DROP TABLE treedb.{t}")
    c.execute("DROP VIEW treedb.v_cut_items")
    c.execute("DROP VIEW treedb.v_item_time")
    if old == 2:
        c.execute("DROP TABLE treedb.documents")
    c.execute("UPDATE treedb.schema_version SET version = ?", [old])
    c.close()
    with pytest.raises(store.SchemaVersionError, match="migrate=True"):
        store.connect(path)
    with pytest.raises(store.SchemaVersionError):
        TreeDB(path)
    assert store.schema_version(duckdb.connect(str(path))) == old        # refused open changed nothing
    c = store.connect(path, migrate=True)
    bak = tmp_path / f"old.duckdb.v{old}.bak"
    assert bak.exists()
    b = duckdb.connect(str(bak), read_only=True)
    assert b.execute("SELECT version FROM treedb.schema_version").fetchone() == (old,)
    b.close()
    assert store.schema_version(c) == 4
    assert c.execute("SELECT count(*) FROM treedb.item_registry").fetchone() == (1,)
    assert c.execute("SELECT count(*) FROM treedb.cut_node").fetchone() == (0,)
    assert store.integrity_violations(c) == {}
    c.close()


def test_integrity_violations_cover_v4_references(con):
    tid = str(uuid.uuid4())
    con.execute("INSERT INTO treedb.cultivar_tree VALUES (?, 'x', 'topic')", [tid])
    con.execute("INSERT INTO treedb.cultivar_node VALUES (?, 'k', ?)", [tid, str(uuid.uuid4())])
    con.execute("INSERT INTO treedb.cut_node VALUES (?, 'c', 'k', NULL, 'root', 0, 0, NULL, 1, [], NULL, NULL, NULL)",
                [tid])
    con.execute("INSERT INTO treedb.item_vector VALUES ('i', 's', 'sha')")
    con.execute("INSERT INTO treedb.item_enrichment (item_id, enricher_id, output) VALUES ('i', 'e', '{}')")
    con.execute("INSERT INTO treedb.node_enrichment (tree_id, node_key, enricher_id, output) VALUES (?, 'k', 'e', '{}')",
                [tid])
    con.execute("INSERT INTO treedb.cultivar_alias (alias, tree_id) VALUES ('a', ?)", [tid])
    bad = store.integrity_violations(con)
    assert {"cultivar_tree.tree_id -> tree_registry.id", "cultivar_node.tree_id -> tree_registry.id",
            "cut_node (tree_id, cut_name) -> cut", "item_vector (space, content_sha256) -> vector",
            "item_enrichment.enricher_id -> enricher", "node_enrichment.enricher_id -> enricher",
            "cultivar_alias.tree_id -> tree_registry.id"} <= set(bad)


# --- tx ---------------------------------------------------------------------------------------

def test_tx_commit_rollback_and_nesting(con):
    con.execute("CREATE TABLE t (x INT)")
    with tx.transaction(con):
        con.execute("INSERT INTO t VALUES (1)")
        with tx.transaction(con):
            assert tx.depth(con) == 2
            con.execute("INSERT INTO t VALUES (2)")
    assert tx.depth(con) == 0 and con.execute("SELECT count(*) FROM t").fetchone() == (2,)
    with pytest.raises(ZeroDivisionError):
        with tx.transaction(con):
            con.execute("INSERT INTO t VALUES (3)")
            with tx.transaction(con):
                con.execute("INSERT INTO t VALUES (4)")
                1 / 0
    assert tx.depth(con) == 0 and con.execute("SELECT count(*) FROM t").fetchone() == (2,)
    con.execute("INSERT INTO t VALUES (5)")                  # connection is usable (no stuck transaction)
    with tx.transaction(con):
        pass


def test_writers_join_the_callers_transaction(con):
    with pytest.raises(RuntimeError):
        with tx.transaction(con):
            checkpoints.put_checkpoint(con, "ns", "k", {"a": 1})
            raise RuntimeError
    assert checkpoints.get_checkpoint(con, "ns", "k") is None


# --- registry: perspectives, enrichers --------------------------------------------------------

def test_perspective_and_enricher_registry(con):
    p = perspective()
    pid = registry.put_perspective(con, p)
    assert registry.put_perspective(con, p) == pid and len(pid) == 64
    assert registry.get_perspective(con, pid) == registry.get_perspective(con, "topic@1") == p
    with pytest.raises(RegistryConflictError):
        registry.put_perspective(con, {**p, "description": "other"})
    from cultivar_builder import enricher
    eid = registry.put_enricher(con, enricher("item.extract", "item"))
    assert enrichment.resolve_enricher(con, "item.extract@1") == eid
    with pytest.raises(RegistryConflictError):
        registry.put_enricher(con, {**enricher("item.extract", "item"), "params": {"x": 1}})
    with pytest.raises(ValueError):
        registry.put_enricher(con, {**enricher("bad", "item"), "scope": "x"})


# --- cultivar import --------------------------------------------------------------------------

def test_import_cultivar_artifact_fills_every_table(con, cultivar):
    r = registry.import_cultivar_artifact(con, cultivar, alias="vb/topic")
    assert not r.already_imported and r.documents_written == 12 and r.vectors_written == 12
    tree_id = r.trees["topic"]
    assert r.tree_names["topic"] == f"vb-topic-topic-{r.artifact_sha256[:8]}"
    assert registry.resolve(con, "vb/topic") == tree_id == registry.resolve(con, r.tree_names["topic"])
    count = lambda t: con.execute(f"SELECT count(*) FROM treedb.{t}").fetchone()[0]
    assert count("documents") == 12 and count("item_registry") == 12
    assert count("cultivar_node") == 23 and count("vector") == 12 and count("item_vector") == 12
    assert count("item_enrichment") == 12 and count("perspective") == 1 and count("enricher") == 2
    assert count("cut") == 1 and count("cut_node") == r.cut_nodes and count("node_enrichment") > 0
    assert store.integrity_violations(con) == {}
    info = registry.cultivar_info(con, "vb/topic")
    assert info["aliases"] == ["vb/topic"] and info["artifact_tree_id"] == "topic"
    assert info["perspective"]["name"] == "topic" and len(info["enrichers"]) == 2
    assert info["artifact"]["manifest"]["name"] == "vb-topic"
    # display filled in cut_node
    rows = cutstore.read_cut(con, tree_id, "topics")
    assert rows[0]["kind"] == "root" and rows[0]["title"] == "All" and rows[0]["tags"] == ["all"]
    assert all(r_["title"] is None for r_ in rows if r_["kind"] == "item")
    # cultivar_node maps to the real orchard nodes
    assert con.execute("SELECT count(*) FROM treedb.cultivar_node n WHERE n.tree_id = ? AND (n.node_id IN "
                       "(SELECT branch_id FROM treedb.tree_branches WHERE tree_id = ?) OR n.node_id IN "
                       "(SELECT item_uuid FROM treedb.item_registry))", [str(tree_id), str(tree_id)]
                       ).fetchone()[0] == 23
    # enrichment + vectors readable
    assert enrichment.get_item_outputs(con, "item.extract@1")["vb:0003"]["word_count"] == 7
    ids, m = vectors.item_matrix(con, SPACE)
    assert ids == sorted(ids) and m.shape == (12, 8) and m.dtype == np.float32
    with np.load(cultivar / "vectors" / f"{SPACE.replace('/', '_')}.npz") as z:
        assert np.array_equal(m, z["vectors"][np.argsort([str(i) for i in z["ids"]])])


def test_import_is_idempotent_on_artifact_sha(con, cultivar):
    a = registry.import_cultivar_artifact(con, cultivar)
    n_trees = con.execute("SELECT count(*) FROM treedb.tree_registry").fetchone()[0]
    b = registry.import_cultivar_artifact(con, cultivar, alias="x/y")
    assert b.already_imported and b.trees == a.trees and registry.resolve(con, "x/y") == a.trees["topic"]
    assert con.execute("SELECT count(*) FROM treedb.tree_registry").fetchone()[0] == n_trees


def test_tampered_file_is_refused_naming_it(con, cultivar):
    p = cultivar / "enrichment/items/item.extract__1.json"
    p.write_text(p.read_text(encoding="utf-8").replace("sum", "SUM"), encoding="utf-8")
    with pytest.raises(CultivarIntegrityError, match="item.extract__1.json"):
        registry.import_cultivar_artifact(con, cultivar)
    assert con.execute("SELECT count(*) FROM treedb.tree_registry").fetchone() == (0,)


def test_extra_file_and_bad_artifact_hash_are_refused(con, cultivar):
    (cultivar / "stray.txt").write_text("x")
    with pytest.raises(CultivarIntegrityError, match="stray.txt"):
        registry.import_cultivar_artifact(con, cultivar)
    (cultivar / "stray.txt").unlink()
    cj = json.loads((cultivar / "cultivar.json").read_text())
    cj["artifact_sha256"] = "0" * 64
    (cultivar / "cultivar.json").write_text(json.dumps(cj))
    with pytest.raises(CultivarIntegrityError, match="artifact_sha256"):
        registry.import_cultivar_artifact(con, cultivar)


def test_failure_after_trees_triggers_compensating_delete(con, cultivar, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(vectors, "put_vectors", boom)
    with pytest.raises(RuntimeError, match="boom"):
        registry.import_cultivar_artifact(con, cultivar)
    for t in ("tree_registry", "tree_z", "tree_branches", "orchard", "cultivar_artifact", "cultivar_tree",
              "cultivar_node", "cut", "cut_node", "documents", "perspective", "enricher", "node_enrichment"):
        assert con.execute(f"SELECT count(*) FROM treedb.{t}").fetchone() == (0,), t
    assert tx.depth(con) == 0
    monkeypatch.undo()
    assert not registry.import_cultivar_artifact(con, cultivar).already_imported     # retry works


def test_perspective_conflict_is_refused_before_any_write(con, tmp_path):
    registry.import_cultivar_artifact(con, build_cultivar(tmp_path / "a", seed=1))
    other = build_cultivar(tmp_path / "b", seed=2, tweak_perspective={"description": "changed"})
    with pytest.raises(RegistryConflictError):
        registry.import_cultivar_artifact(con, other)
    assert con.execute("SELECT count(*) FROM treedb.tree_registry").fetchone() == (1,)


def test_documents_modes(con, cultivar):
    documents.upsert_documents(con, [{"item_id": "vb:0001", "title": "mine", "text": "keep me"}])
    with pytest.raises(orchard_artifact.ArtifactError, match="already have a document"):
        registry.import_cultivar_artifact(con, cultivar, documents="refuse")
    registry.import_cultivar_artifact(con, cultivar, documents="keep")
    assert store.get_documents(con, ["vb:0001"]).documents["vb:0001"]["title"] == "mine"
    c2 = store.connect()
    documents.upsert_documents(c2, [{"item_id": "vb:0001", "title": "mine", "text": "keep me"}])
    registry.import_cultivar_artifact(c2, cultivar, documents="replace")
    assert store.get_documents(c2, ["vb:0001"]).documents["vb:0001"]["title"] == "Title 1"


def test_two_cultivars_with_explicit_tree_name(con, tmp_path):
    a = registry.import_cultivar_artifact(con, build_cultivar(tmp_path / "a", seed=1), tree_name="first", alias="vb/topic")
    b = registry.import_cultivar_artifact(con, build_cultivar(tmp_path / "b", seed=1, name="other"), tree_name="second",
                                          alias="vb/topic")
    assert a.tree_names["topic"] == "first" and registry.resolve(con, "vb/topic") == b.trees["topic"]
    assert store.integrity_violations(con) == {}


def test_facade_methods(tmp_path, cultivar):
    with TreeDB(tmp_path / "f.duckdb") as db:
        r = db.import_cultivar_artifact(cultivar, alias="vb/topic")
        assert db.resolve("vb/topic") == r.trees["topic"]
        assert db.read_cut("vb/topic", "topics")[0]["kind"] == "root"
        ids, m = db.item_matrix(SPACE)
        assert len(db.nearest(SPACE, m[0], k=3)) == 3 and db.cultivar_info("vb/topic")["aliases"] == ["vb/topic"]
        db.ledger_append("test", item_id="vb:0001")
        assert db.ledger_events(action="test")[0]["item_id"] == "vb:0001"


def test_read_artifact_matches_old_import(tmp_path, cultivar):
    p = orchard_artifact.read_artifact(cultivar)
    assert len(p.item_ids) == 12 and p.trees[0]["name"] == "topic"
    c = store.connect()
    assert orchard_artifact.import_orchard_artifact(c, cultivar).documents_written == 12   # cultivar dir stays a valid artifact


# --- vectors ----------------------------------------------------------------------------------

def test_vectors_roundtrip_nearest_and_skip_existing(con):
    rng = np.random.default_rng(0)
    m = rng.normal(size=(50, 16)).astype(np.float32)
    rows = [(f"sha{i}", m[i]) for i in range(50)]
    assert vectors.put_vectors(con, "sp", rows) == 50
    assert vectors.put_vectors(con, "sp", rows[:10]) == 0
    got = vectors.get_vectors(con, "sp", ["sha3", "nope"])
    assert list(got) == ["sha3"] and np.array_equal(got["sha3"], m[3])
    vectors.link_items(con, "sp", [(f"it{i}", f"sha{i}") for i in range(50)])
    vectors.link_items(con, "sp", [("it0", "sha1")])                      # upsert
    ids, mat = vectors.item_matrix(con, "sp", ["it7", "it2", "ghost"])
    assert ids == ["it7", "it2"] and np.array_equal(mat, m[[7, 2]])
    top = vectors.nearest(con, "sp", m[7], k=3)
    assert top[0][0] == "it7" and abs(top[0][1] - 1.0) < 1e-5
    assert vectors.nearest(con, "sp", m[7], k=3, item_ids=["it2", "it3"])[0][0] in ("it2", "it3")
    with pytest.raises(ValueError):
        vectors.put_vectors(con, "sp", [("x", np.ones(3))])               # dimension clash


# --- documents --------------------------------------------------------------------------------

def test_timestamps_normalized_to_utc_z(con):
    cases = {"2026-01-02T03:04:05+02:00": "2026-01-02T01:04:05.000000Z", "2026-01-02 03:04:05+00": "2026-01-02T03:04:05.000000Z",
             "2026-01-02T03:04:05.250Z": "2026-01-02T03:04:05.250000Z", "2026-01-02T03:04:05": "2026-01-02T03:04:05.000000Z",
             "2026-01-02T03:04:05-0530": "2026-01-02T08:34:05.000000Z"}
    for src, want in cases.items():
        assert documents.normalize_timestamp(src) == want, src
    documents.upsert_documents(con, [{"item_id": "a", "text": "t", "metadata": {"captured_at": "2026-01-02T03:00:00+02:00",
                                                                           "updated_at": "2026-01-02 03:00:00+00"}}])
    md = store.get_documents(con, ["a"]).documents["a"]["metadata"]
    assert md == {"captured_at": "2026-01-02T01:00:00.000000Z", "updated_at": "2026-01-02T03:00:00.000000Z"}
    assert con.execute("SELECT captured_at FROM treedb.v_item_time").fetchone()[0].hour == 1


def test_upsert_registers_items_and_url_lookup(con):
    r = documents.upsert_documents(con, [
        {"item_id": "vb:1", "title": "a", "text": "x", "metadata": {"url": "HTTPS://Example.com/a#frag"}},
        {"item_id": "vb:2", "text": "y", "metadata": {"url": "https://example.com", "captured_at": "2026-02-01T00:00:00Z"}}])
    assert r.inserted == ("vb:1", "vb:2")
    assert documents.upsert_documents(con, [{"item_id": "vb:1", "text": "z"}]).updated == ("vb:1",)
    assert con.execute("SELECT count(*) FROM treedb.item_registry").fetchone() == (2,)
    assert store.integrity_violations(con) == {}
    assert documents.find_by_url(con, "https://example.com/") == ["vb:2"]
    assert documents.normalize_url("HTTP://A.com/p/#x") == "http://a.com/p/"
    assert documents.item_id_for_url("https://Example.com/") == documents.item_id_for_url("https://example.com")
    assert documents.item_id_for_url("https://example.com").startswith("vb:")
    assert [d["item_id"] for d in documents.recent_items(con)] == ["vb:2", "vb:1"]
    assert documents.recent_items(con, since="2026-01-01T00:00:00Z")[0]["captured_at"] == "2026-02-01T00:00:00.000000Z"
    with pytest.raises(ValueError):
        documents.upsert_documents(con, [{"item_id": "", "text": "x"}])


# --- cutstore ---------------------------------------------------------------------------------

def test_read_cut_windows_and_add_items(con, cultivar):
    r = registry.import_cultivar_artifact(con, cultivar)
    tid = r.trees["topic"]
    full = cutstore.read_cut(con, tid, "topics")
    assert len(full) == r.cut_nodes and full[0]["depth"] == 0
    # items captured every 10 days from 2026-01-01: days 0..110
    win = cutstore.read_cut(con, tid, "topics", since="2026-02-01T00:00:00Z", until="2026-03-15T00:00:00Z")
    items = [x for x in win if x["kind"] == "item"]
    assert sorted(x["item_id"] for x in items) == ["vb:0004", "vb:0005", "vb:0006", "vb:0007"]   # days 40..70 (Feb 10..Mar 12)
    keys = {x["node_key"] for x in win}
    assert all(p in keys for x in win for p in x["path"])                    # ancestors present
    assert win[0]["kind"] == "root" and win[0]["window_count"] == 4
    assert all(x["captured_at"] for x in items)
    assert cutstore.read_cut(con, tid, "topics", since="2030-01-01T00:00:00Z") == []
    # incremental add
    cluster = next(x for x in full if x["kind"] == "cluster")
    rev = cutstore.cut_revision(con, tid, "topics")
    n = cutstore.add_items(con, tid, "topics", [{"item_id": "vb:new", "parent_key": cluster["node_key"]},
                                               {"item_id": "vb:0001", "parent_key": cluster["node_key"]}])
    assert n == 1 and cutstore.cut_revision(con, tid, "topics") == rev + 1       # re-capture adds no second row
    after = {x["node_key"]: x for x in cutstore.read_cut(con, tid, "topics")}
    assert after[cluster["node_key"]]["descendant_count"] == cluster["descendant_count"] + 1
    assert after[full[0]["node_key"]]["descendant_count"] == 13
    new = next(x for x in after.values() if x["item_id"] == "vb:new")
    assert new["depth"] == cluster["depth"] + 1 and new["path"] == cluster["path"] + [cluster["node_key"]]
    cutstore.update_display(con, tid, "topics", {cluster["node_key"]: {"title": "Renamed"}})
    row = next(x for x in cutstore.read_cut(con, tid, "topics") if x["node_key"] == cluster["node_key"])
    assert row["title"] == "Renamed" and row["summary"] == cluster["summary"]
    with pytest.raises(KeyError):
        cutstore.add_items(con, tid, "topics", [{"item_id": "z", "parent_key": "nope"}])
    cutstore.put_cut(con, tid, json.loads((cultivar / "trees/topic/cuts/topics.json").read_text()))   # replace
    assert cutstore.cut_revision(con, tid, "topics") == rev + 3
    assert store.integrity_violations(con) == {}


# --- ledger, checkpoints ----------------------------------------------------------------------

def test_ledger_and_checkpoints(con):
    t = uuid.uuid4()
    e = ledger.append(con, "attach", actor="me", tree_id=t, payload={"a": 1})
    ledger.append(con, "other", item_id="i")
    ev = ledger.events(con, action="attach")
    assert len(ev) == 1 and ev[0]["event_id"] == e and ev[0]["payload"] == {"a": 1} and ev[0]["actor"] == "me"
    assert len(ledger.events(con)) == 2 and len(ledger.events(con, tree_id=t)) == 1
    with pytest.raises(TypeError):
        ledger.append(con, "x", bogus=1)
    checkpoints.put_checkpoint(con, "ns", "a", {"n": 1})
    checkpoints.put_checkpoint(con, "ns", "a", {"n": 2})
    checkpoints.put_checkpoint(con, "ns", "b", [1])
    assert checkpoints.get_checkpoint(con, "ns", "a") == {"n": 2} and checkpoints.get_checkpoint(con, "ns", "z") is None
    assert checkpoints.list_checkpoints(con, "ns") == {"a": {"n": 2}, "b": [1]}


# --- attach_item ------------------------------------------------------------------------------

def test_attach_item_success_and_refusals(con, cultivar):
    r = registry.import_cultivar_artifact(con, cultivar)
    tid = r.trees["topic"]
    rows = cutstore.read_cut(con, tid, "topics")
    cluster = next(x for x in rows if x["kind"] == "cluster")["node_key"]
    root = rows[0]["node_key"]
    assert registry.node_id_for(con, tid, "nope") is None and registry.node_id_for(con, tid, cluster) is not None
    n_before = con.execute("SELECT count(*) FROM treedb.cultivar_node").fetchone()[0]
    out = registry.attach_item(con, tid, cluster, "item_new", "vb:new1")
    assert out.ok and out.reason is None
    assert registry.node_id_for(con, tid, "item_new") == out.node_id
    assert con.execute("SELECT count(*) FROM treedb.cultivar_node").fetchone()[0] == n_before + 1
    assert con.execute("SELECT count(*) FROM treedb.item_registry WHERE item_id = 'vb:new1'").fetchone()[0] == 1
    assert store.integrity_violations(con) == {}
    # refusals never raise
    assert registry.attach_item(con, tid, "nope", "k2", "vb:new2").reason == "unknown_parent_key"
    assert registry.attach_item(con, tid, root, "k3", "vb:new3").reason == "no_orchard_row"
    assert registry.attach_item(con, tid, cluster, "item_new", "vb:new4").reason == "node_key_exists"
    dup = registry.attach_item(con, tid, cluster, "k5", "vb:new1")          # item already has a parent
    assert not dup.ok and dup.reason
    with tx.transaction(con):
        assert registry.attach_item(con, tid, cluster, "k6", "vb:new6").reason == "inside_transaction"


def test_fresh_and_empty_files_get_v4_without_flag(tmp_path):
    c = store.connect(tmp_path / "new.duckdb")
    assert store.schema_version(c) == 4
    c.close()
    duckdb.connect(str(tmp_path / "empty.duckdb")).close()
    c = store.connect(tmp_path / "empty.duckdb")
    assert store.schema_version(c) == 4
    c.close()
    assert not list(tmp_path.glob("*.bak"))
