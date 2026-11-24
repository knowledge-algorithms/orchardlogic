"""P6A: the ``treedb_*`` SQL macros return what the Python calls return (docs/SQL_API.md)."""

from __future__ import annotations

import pytest

from api_helpers import load_into, parent_branch
from orchardlogic.treedb import TreeDB, read
from orchardlogic.treedb.fixtures import item_uuid


@pytest.fixture(params=["F5_twocult8"])
def loaded(request):
    db = TreeDB()
    fx = load_into(db, request.param)
    yield db, fx
    db.close()


def _sql(db, text, params=None):
    return db.connection.execute(text, params or []).fetchall()


def test_treedb_cultivars_matches_python(loaded):
    db, _ = loaded
    got = _sql(db, "SELECT cultivar, state_version FROM treedb_cultivars()")
    assert got == [(c["name"], c["state_version"]) for c in db.cultivars()]


def test_treedb_node_matches_python(loaded):
    db, fx = loaded
    cv = fx["cultivars"][0]["name"]
    item = sorted(fx["item_ids"])[0]
    leaf = db.node(cv, item_uuid(item))
    branch = db.node(cv, leaf.parent_branch_id)
    root_id = next(c for c in db.cultivars() if c["name"] == cv)["root_branch_id"]
    for rec in (leaf, branch, db.node(cv, root_id)):
        rows = _sql(db, "SELECT cultivar, state_version, kind, item_id, orchard_id, parent_branch_id, edge_ix, "
                        "edge_coph_distance FROM treedb_node(?, ?)", [cv, str(rec.node_id)])
        assert rows == [(rec.cultivar, rec.state_version, rec.kind, rec.item_id, rec.orchard_id,
                         rec.parent_branch_id, rec.edge_ix, rec.edge_coph_distance)]
    assert _sql(db, "SELECT * FROM treedb_node(?, ?)", [cv, "00000000-0000-0000-0000-000000000000"]) == []


def test_treedb_ancestry_matches_python(loaded):
    db, fx = loaded
    cv = fx["cultivars"][0]["name"]
    item = sorted(fx["item_ids"])[3]
    for node in (item_uuid(item), parent_branch(db, cv, item)):
        want = db.node_ancestry(cv, node)
        got = _sql(db, "SELECT * FROM treedb_ancestry(?, ?)", [cv, str(node)])
        assert sorted(got, key=repr) == sorted(want, key=repr) and got


def test_treedb_ancestry_of_a_root_is_empty(loaded):
    db, fx = loaded
    info = next(c for c in db.cultivars() if c["name"] == fx["cultivars"][0]["name"])
    assert db.node_ancestry(info["name"], info["root_branch_id"]) == []
    assert _sql(db, "SELECT * FROM treedb_ancestry(?, ?)", [info["name"], str(info["root_branch_id"])]) == []


def test_treedb_subtree_and_cohort_items_match_python(loaded):
    db, fx = loaded
    ids = sorted(fx["item_ids"])[:9]
    c = db.cohort(ids)
    assert [r[0] for r in _sql(db, "SELECT item_id FROM treedb_cohort_items(?)", [str(c.cohort_id)])] == db.items(c)
    for cv in (x["name"] for x in fx["cultivars"]):
        want = db.subtree(cv, ids).rows
        cur = db.connection.execute("SELECT * FROM treedb_subtree(?, ?)", [cv, str(c.cohort_id)])
        assert tuple(d[0] for d in cur.description) == read.COMPRESSED_COLUMNS
        got = cur.fetchall()
        assert sorted(got, key=repr) == sorted(want, key=repr) and got


def test_treedb_cohort_header_matches_python(loaded):
    db, fx = loaded
    c = db.cohort(sorted(fx["item_ids"])[:4])
    rows = _sql(db, "SELECT cohort_id, membership_hash, size, kind FROM treedb_cohort(?)", [str(c.cohort_id)])
    assert rows == [(c.cohort_id, c.membership_hash, c.size, c.kind)]


def test_macros_survive_reopen_of_a_file(tmp_path):
    path = tmp_path / "m.duckdb"
    with TreeDB(path) as db:
        load_into(db, "F5_twocult8")
        want = _sql(db, "SELECT * FROM treedb_cultivars()")
    with TreeDB(path) as db:
        assert _sql(db, "SELECT * FROM treedb_cultivars()") == want
