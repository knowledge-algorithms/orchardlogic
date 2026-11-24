"""Shared items registry and tree isolation (Arch §2 source of truth item 3, §8.2; INV-1, INV-25).

Two cultivars over the same item ids (F5 alpha/beta, AppWorld domain/function)
share one items registry row per item, and their Orchard trees do not touch.
F6 adds a cultivar that holds a subset of the corpus (INV-20 data).
"""

from __future__ import annotations

import pytest

from ingest_helpers import FIXTURES, cultivar, load_all, tree_rows
from orchardlogic.treedb import canonical, ingest, store
from orchardlogic.treedb.fixtures import item_uuid

PAIRS = [("F5_twocult8", "alpha", "beta"), ("F6_partial", "full", "partial")]


@pytest.mark.parametrize("stem,a,b", PAIRS)
def test_items_registry_is_shared(stem, a, b):
    con, results = load_all(stem)
    _, _, items_a = cultivar(stem, a)
    _, _, items_b = cultivar(stem, b)
    corpus = set(items_a) | set(items_b)
    rows = con.execute("SELECT item_id, item_uuid FROM treedb.item_registry").fetchall()
    assert len(rows) == len(corpus)                      # one row per item, no duplicates
    assert {r[0] for r in rows} == corpus
    assert all(u == item_uuid(i) for i, u in rows)      # INV-5 CONTRACTS UUID
    # the same leaf UUID is used in both trees for every shared item (INV-25)
    leaves = {}
    for name in (a, b):
        leaves[name] = {nd for _b, nd, e, _d in tree_rows(con, results[name].tree_id) if e == ingest.EDGE_LEAF}
    shared = set(items_a) & set(items_b)
    assert {item_uuid(i) for i in shared} <= leaves[a] & leaves[b]
    assert leaves[a] == {item_uuid(i) for i in items_a}
    assert leaves[b] == {item_uuid(i) for i in items_b}


@pytest.mark.parametrize("stem,a,b", PAIRS)
def test_trees_are_isolated(stem, a, b):
    """No branch, edge or staging row is shared; importing B leaves A's snapshot unchanged."""
    con = store.connect()
    fx = FIXTURES[stem]
    ra = ingest.import_fixture_cultivar(con, fx, a)
    snap_a = canonical.canonical_snapshot(con, a)
    rb = ingest.import_fixture_cultivar(con, fx, b)
    assert canonical.canonical_snapshot(con, a) == snap_a
    br = {}
    for name, r in ((a, ra), (b, rb)):
        br[name] = {x[0] for x in con.execute(
            "SELECT branch_id FROM treedb.tree_branches WHERE tree_id = ?", [r.tree_id]).fetchall()}
    assert not br[a] & br[b]
    assert ra.root_branch_id != rb.root_branch_id
    for name, r in ((a, ra), (b, rb)):
        other = br[b] if name == a else br[a]
        for _b, nd, _e, _d in tree_rows(con, r.tree_id):
            assert nd not in other                       # no cross-tree edge
    # every staging row belongs to exactly one tree and v_tree_map root rows are per tree
    roots = con.execute("SELECT tree_id, doc_id FROM treedb.v_tree_map WHERE b_side = 'root' ORDER BY 1").fetchall()
    assert sorted(roots) == sorted([(ra.tree_id, ra.root_branch_id), (rb.tree_id, rb.root_branch_id)])
    assert store.integrity_violations(con) == {}


def test_registry_keeps_existing_rows_and_source_ref():
    """A second import never rewrites an existing item row (ON CONFLICT DO NOTHING)."""
    con = store.connect()
    con.execute("INSERT INTO treedb.item_registry VALUES (?, ?, 'src:s0')", ["s0", item_uuid("s0")])
    fx = FIXTURES["F5_twocult8"]
    ingest.import_fixture_cultivar(con, fx, "alpha")
    assert con.execute("SELECT source_ref FROM treedb.item_registry WHERE item_id = 's0'").fetchone() == ("src:s0",)


def test_registry_rejects_non_contract_uuid():
    con = store.connect()
    con.execute("INSERT INTO treedb.item_registry VALUES ('s0', gen_random_uuid(), NULL)")
    fx = FIXTURES["F5_twocult8"]
    with pytest.raises(ingest.IngestError, match="non-CONTRACTS UUID"):
        ingest.import_fixture_cultivar(con, fx, "alpha")
    assert con.execute("SELECT count(*) FROM treedb.tree_registry").fetchone() == (0,)
