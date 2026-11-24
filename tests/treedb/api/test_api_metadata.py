"""P6A, architecture 8.9 (INV-11): labels and cards are metadata. Editing one never changes topology,
lineage or state_version, and labels follow node identity through attach and detach."""

from __future__ import annotations

import json
import warnings

import pytest

from api_helpers import load_into, orchard_row
from orchardlogic.treedb import TreeDB, read
from orchardlogic.treedb.fixtures import item_uuid

TOPOLOGY_TABLES = ("orchard", "tree_branches", "tree_z", "tree_registry", "item_registry", "cohort", "cohort_member")


def _topology(db):
    con = db.connection
    return {t: con.execute(f"SELECT * FROM treedb.{t} ORDER BY ALL").fetchall() for t in TOPOLOGY_TABLES}


@pytest.fixture
def db():
    d = TreeDB()
    load_into(d, "F5_twocult8")
    d.cohort(["s0", "s1", "s2"])
    yield d
    d.close()


def test_label_and_card_edits_leave_topology_unchanged(db):
    leaf = item_uuid("s1")
    branch = db.node("alpha", leaf).parent_branch_id
    c = db.cohort(["s0", "s1", "s2", "s3"])
    before, rows_before = _topology(db), db.project(c, "alpha").raw_rows
    ancestry_before = db.ancestry("alpha", ["s0", "s1", "s2", "s3"]).rows

    db.set_label("alpha", branch, "mine", "the branch")
    db.set_label("alpha", branch, "mine", "the branch, renamed")        # replace
    db.set_label("alpha", leaf, "mine", "a leaf")
    db.set_card("alpha", branch, {"note": "x", "n": [1, 2]})
    db.set_card("alpha", branch, {"note": "y"})                         # replace

    assert _topology(db) == before                                      # every structural table, state_version included
    assert db.project(c, "alpha").raw_rows == rows_before
    assert db.ancestry("alpha", ["s0", "s1", "s2", "s3"]).rows == ancestry_before
    rec = db.node("alpha", branch)
    assert rec.labels["mine"] == "the branch, renamed" and rec.card == {"note": "y"}
    assert rec.state_version == 0
    assert db.node("alpha", leaf).labels == {"mine": "a leaf"}
    p = db.project(c, "alpha")
    assert p.labels[branch]["mine"] == "the branch, renamed" and p.cards[branch] == {"note": "y"}
    stored = db.connection.execute("SELECT card FROM treedb.node_cards WHERE node_id = ?", [branch]).fetchone()[0]
    assert stored == json.dumps({"note": "y"}, separators=(",", ":"))


def test_source_node_key_is_set_once_then_kept(db):
    branch = db.node("alpha", item_uuid("s1")).parent_branch_id
    db.set_label("alpha", branch, "one", "a")
    db.set_label("alpha", branch, "two", "b")
    db.set_card("alpha", branch, {"x": 1})
    db.set_label("alpha", branch, "one", "c")
    keys = {r[0] for r in db.connection.execute(
        "SELECT source_node_key FROM treedb.node_labels WHERE node_id = ? UNION ALL "
        "SELECT source_node_key FROM treedb.node_cards WHERE node_id = ?", [branch, branch]).fetchall()}
    assert keys == {f"set:{branch}"}


def test_labels_follow_node_identity_through_attach_and_detach(db):
    leaf, other = item_uuid("s3"), item_uuid("s4")
    info = next(c for c in db.cultivars() if c["name"] == "alpha")
    root = info["root_branch_id"]
    db.set_label("alpha", root, "mine", "root label")
    db.set_label("alpha", leaf, "mine", "s3 label")
    db.set_card("alpha", leaf, {"k": 1})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db.detach(orchard_row(db, "alpha", "s3"))
    assert db.node("alpha", root).labels["mine"] == "root label"        # a node that stayed
    db.attach(leaf, orchard_row(db, "alpha", "s4"), 1)                  # same node identity, new place
    rec = db.node("alpha", leaf)
    assert rec.labels["mine"] == "s3 label" and rec.card == {"k": 1}
    assert rec.node_id == leaf and db.node("alpha", other).item_id == "s4"
    assert read.tree_info(db.connection, "alpha")["state_version"] == 2  # moves only for the two mutations


def test_set_on_a_node_of_another_cultivar_or_unknown_is_refused(db):
    foreign = db.node("beta", item_uuid("s0")).parent_branch_id
    with pytest.raises(read.NodeNotFoundError):
        db.set_label("alpha", foreign, "mine", "x")
    with pytest.raises(read.NodeNotFoundError):
        db.set_card("alpha", "00000000-0000-0000-0000-000000000000", {})
    with pytest.raises(read.ReadError):
        db.set_label("nope", item_uuid("s0"), "mine", "x")
    with pytest.raises(ValueError):
        db.set_label("alpha", item_uuid("s0"), "", "x")
