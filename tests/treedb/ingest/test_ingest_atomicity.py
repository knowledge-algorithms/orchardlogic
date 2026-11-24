"""Import is one transaction; invalid input writes nothing (PORTING_NOTES CO-1; INV-1, 2, 9, 10, 11)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ingest_helpers import FIXTURES, cultivar
from orchardlogic.treedb import ingest, store
from orchardlogic.treedb.fixtures import node_keys

COUNTED = ("tree_registry", "tree_z", "tree_branches", "orchard", "item_registry", "node_labels", "node_cards")


def _counts(con):
    return {t: con.execute(f"SELECT count(*) FROM treedb.{t}").fetchone()[0] for t in COUNTED}


def _f1():
    fx, cv, items = cultivar("F1_tiny3", "main")
    return cv["z"], items


@pytest.mark.parametrize("step", ["_create_orchard", "_insert_labels", "_insert_cards"])
def test_failure_mid_import_rolls_back_everything(step, monkeypatch):
    con = store.connect()
    z, items = _f1()
    before = _counts(con)
    real = getattr(ingest, step)

    def boom(*args, **kwargs):
        real(*args, **kwargs)          # do the step's writes, then fail
        raise RuntimeError("injected failure")

    monkeypatch.setattr(ingest, step, boom)
    with pytest.raises(RuntimeError, match="injected"):
        ingest.import_cultivar(con, z, items, "main", cards={node_keys(z, items)[0]: {"k": 1}})
    assert _counts(con) == before
    monkeypatch.undo()
    # the connection is usable afterwards and the same import now succeeds
    ingest.import_cultivar(con, z, items, "main")
    assert _counts(con)["orchard"] == 4


def test_duplicate_name_is_rejected_atomically():
    con = store.connect()
    z, items = _f1()
    ingest.import_cultivar(con, z, items, "main")
    before = _counts(con)
    with pytest.raises(Exception, match="(?i)duplicate key|unique"):
        ingest.import_cultivar(con, z, items, "main")
    assert _counts(con) == before


def test_cards_are_stored_by_node_identity():
    con = store.connect()
    z, items = _f1()
    keys = node_keys(z, items)
    res = ingest.import_cultivar(con, z, items, "main", cards={keys[3]: {"title": "pair"}, keys[0]: {"x": [1]}})
    rows = dict((k, (n, json.loads(c))) for n, c, k in con.execute(
        "SELECT node_id, card, source_node_key FROM treedb.node_cards").fetchall())
    assert rows[keys[3]] == (res.node_ids[3], {"title": "pair"})
    assert rows[keys[0]] == (res.node_ids[0], {"x": [1]})
    assert store.integrity_violations(con) == {}


BAD_INPUTS = {
    "bad name": dict(name="1bad"),
    "dup items": dict(items=["t0", "t0", "t2"]),
    "space item": dict(items=["t0 ", "t1", "t2"]),
    "one leaf": dict(items=["t0"], z=np.zeros((0, 4))),
    "wrong shape": dict(z=[[0, 1, 0.5, 2]]),
    "negative distance": dict(z=[[0, 1, -0.5, 2], [2, 3, 1.0, 3]]),
    "bad count": dict(z=[[0, 1, 0.5, 2], [2, 3, 1.0, 7]]),
    "non-integral child": dict(z=[[0.5, 1, 0.5, 2], [2, 3, 1.0, 3]]),
    "nan": dict(z=[[0, 1, float("nan"), 2], [2, 3, 1.0, 3]]),
    "unknown label key": dict(labels={"s": {"member_x": "x"}}),
    "non-string label": dict(labels="KEY0_INT"),
    "unknown card key": dict(cards={"member_x": {}}),
}


@pytest.mark.parametrize("case", sorted(BAD_INPUTS))
def test_invalid_input_writes_nothing(case):
    con = store.connect()
    z, items = _f1()
    args = dict(z=z, items=items, name="main", labels=None, cards=None)
    args.update(BAD_INPUTS[case])
    if args["labels"] == "KEY0_INT":
        args["labels"] = {"s": {node_keys(z, items)[0]: 5}}
    before = _counts(con)
    with pytest.raises(ingest.IngestError):
        ingest.import_cultivar(con, args["z"], args["items"], args["name"],
                               labels=args["labels"], cards=args["cards"])
    assert _counts(con) == before


def test_non_integral_child_passes_scipy_validation():
    """Reproducer for the extra INV-8 check: scipy's is_valid_linkage accepts a 0.5 child index."""
    from scipy.cluster.hierarchy import is_valid_linkage
    assert is_valid_linkage(np.array([[0.5, 1, 0.5, 2], [2, 3, 1.0, 3]], dtype=float))


def test_import_inside_open_transaction_is_refused():
    """import_cultivar owns its transaction; DuckDB has no nested BEGIN."""
    import duckdb
    con = store.connect()
    z, items = _f1()
    con.execute("BEGIN TRANSACTION")
    with pytest.raises(duckdb.TransactionException):
        ingest.import_cultivar(con, z, items, "main")
    con.execute("ROLLBACK")
    assert FIXTURES  # fixtures loaded
