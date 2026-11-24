"""PX-G4 (P5B review M1): the query layer raises on an ancestry cycle, never returns a quiet result.

D-024 Q7 (cycle guard) applied to execute and explain. Invariants: INV-19 (a result is
a subset cohort; none is made when the walk fails), INV-24-style "nothing created on error".
"""

from __future__ import annotations

import pytest

from query_helpers import fixture_store, new_store, node_by_label, shared_store, translate
from orchardlogic.treedb import cohort, read
from orchardlogic.treedb.query import Query, execute, explain


def _cyclic_store():
    """F7 with A1's parent row pointed at A3 (A3 is A1's child): the cycle A1 <-> A3."""
    con = fixture_store("F7_twocult8")
    a1, a3 = node_by_label(con, "A", "A1"), node_by_label(con, "A", "A3")
    con.execute("UPDATE treedb.orchard SET branch_id = ? WHERE node_id = ?", [a3, a1])
    return con


def _include(con, label):
    return Query.from_json(translate(con, {
        "base": {"cohort": "c0"}, "rules": [{"cultivar": "A", "node": label, "op": "include"}]}))


def test_execute_raises_on_a_cycle_and_creates_no_cohort():
    con = _cyclic_store()
    base = cohort.all_items(con)
    before = con.execute("SELECT count(*) FROM treedb.cohort").fetchone()[0]
    with pytest.raises(read.AncestryCycleError):
        execute(con, base, _include(con, "A4"))
    assert con.execute("SELECT count(*) FROM treedb.cohort").fetchone()[0] == before


def test_explain_raises_on_a_cycle():
    con = _cyclic_store()
    base = cohort.all_items(con)
    q = _include(con, "A4")
    with pytest.raises(read.AncestryCycleError):
        explain(con, q, "i1", base=base)


@pytest.mark.parametrize("stem", ["F7_twocult8", "F3_caterpillar6"])
def test_valid_trees_are_unchanged(stem):
    """Valid trees do not raise. (This is not the ``hops + 1 == count(orchard)`` edge: no F3 chain
    reaches every orchard row, because each branch has a sibling row. That edge is
    ``test_chain_through_every_orchard_row_is_not_a_cycle`` below.)"""
    con = shared_store(stem)
    base = cohort.all_items(con)
    items = cohort.items(con, base)
    cult = con.execute("SELECT name FROM treedb.tree_registry ORDER BY name").fetchone()[0]
    node = con.execute(
        "SELECT o.branch_id FROM treedb.v_orchard o JOIN treedb.tree_registry t ON t.id = o.tree_id "
        "WHERE t.name = ? AND o.node_id = (SELECT item_uuid FROM treedb.item_registry WHERE item_id = ?)",
        [cult, items[-1]]).fetchone()[0]
    q = Query.from_json({"rules": [{"cultivar": cult, "node": str(node), "op": "include"}]})
    res = execute(con, base, q)
    assert res.items
    assert explain(con, res, res.items[0]).verdict is True


def _unary_chain_store(extra: int):
    """A cultivar whose orchard rows all lie on ONE chain.

    ``ingest`` a 2-item tree, drop the row of i2, then hang ``extra`` unary branches above the
    root: rows are (root, i1), (b1, root), (b2, b1), ... so the chain from i1 visits every row.
    Returns (con, top branch id, row count).
    """
    import uuid

    from orchardlogic.treedb import ingest

    con = new_store()
    imp = ingest.import_cultivar(con, [[0.0, 1.0, 1.0, 2.0]], ["i1", "i2"], "U")
    i2 = con.execute("SELECT item_uuid FROM treedb.item_registry WHERE item_id = 'i2'").fetchone()[0]
    con.execute("DELETE FROM treedb.orchard WHERE node_id = ?", [i2])
    top = imp.root_branch_id
    for k in range(extra):
        nxt = uuid.uuid4()
        con.execute("INSERT INTO treedb.tree_branches (branch_id, tree_id) VALUES (?, ?)", [nxt, imp.tree_id])
        con.execute("INSERT INTO treedb.orchard (branch_id, node_id, edge_ix, coph_distance) VALUES (?, ?, 2, ?)",
                    [nxt, top, 2.0 + k])
        top = nxt
    return con, top, con.execute("SELECT count(*) FROM treedb.orchard").fetchone()[0]


def test_chain_through_every_orchard_row_is_not_a_cycle():
    """INV-19 / D-024 Q7: the bound ``hops + 1 <= count(orchard)`` is exact. Here the chain from
    i1 has ``hops == count(orchard)`` (4); no false positive, and the result is correct."""
    con, top, rows = _unary_chain_store(3)
    assert rows == 4
    base = cohort.all_items(con)
    q = Query.from_json({"rules": [{"cultivar": "U", "node": str(top), "op": "include"}]})
    res = execute(con, base, q)
    assert res.items == ("i1",)
    assert explain(con, res, "i1").verdict is True
