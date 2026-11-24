"""P6A: the facade delegates; it adds no behaviour (architecture 7, "Core" operations).

Each test calls the facade and the module directly on the same store and compares. Cohort handles
are fresh per call, so cohorts are compared by kind, members and membership hash.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from api_helpers import load_into, orchard_row, parent_branch
from orchardlogic.treedb import TreeDB, cohort, normalize, project, read
from orchardlogic.treedb.fixtures import item_uuid
from orchardlogic.treedb.query import Query, execute, explain


@pytest.fixture(params=["F5_twocult8"])
def loaded(request):
    db = TreeDB()
    fx = load_into(db, request.param)
    yield db, fx
    db.close()


def _items(fx, n):
    return sorted(fx["item_ids"])[:n]


def test_api_exports_and_closes():
    db = TreeDB()
    assert db.cultivars() == []
    db.close()
    with pytest.raises(ValueError):
        db.connection
    db.close()  # twice is harmless
    with TreeDB() as db2:
        assert db2.connection is not None


def test_import_and_read_match_modules(loaded):
    db, fx = loaded
    con = db.connection
    names = [c["name"] for c in fx["cultivars"]]
    assert [c["name"] for c in db.cultivars()] == sorted(names)
    assert db.cultivars() == [read.tree_info(con, n) for n in sorted(names)]
    assert all(c["state_version"] == 0 for c in db.cultivars())
    cv = names[0]
    items = _items(fx, 6)
    assert db.node(cv, item_uuid(items[0])) == read.get_node(con, cv, item_uuid(items[0]))
    a = db.ancestry(cv, items)
    b = read.ancestry_paths_for_items(con, cv, items)
    assert a == b and a.resolution.state_version == 0
    assert db.node_ancestry(cv, item_uuid(items[0])) == read.node_ancestry(con, cv, item_uuid(items[0]))
    r = db.subtree(cv, items)
    assert r == read.compressed_subtree_for_items(con, cv, items)
    Z, leaves, ctx = db.subtree(cv, items, output="scipy", leaf_order="item_id")
    Z2, leaves2, ctx2 = normalize.scipy_linkage_for_items(con, cv, items, leaf_order="item_id")
    assert np.array_equal(Z, Z2) and leaves == leaves2 and ctx["state_version"] == ctx2["state_version"] == 0
    with pytest.raises(ValueError):
        db.subtree(cv, items, output="bogus")


def test_cohort_algebra_matches_modules(loaded):
    db, fx = loaded
    con = db.connection
    ids = _items(fx, 8)
    a, b = db.cohort(ids[:5]), db.cohort(ids[3:])
    for facade, direct in ((db.union(a, b), cohort.union(con, a, b)),
                           (db.intersect(a, b), cohort.intersect(con, a, b)),
                           (db.difference(a, b), cohort.difference(con, a, b))):
        assert (facade.kind, facade.membership_hash, facade.size, facade.base_cohort_id) == \
               (direct.kind, direct.membership_hash, direct.size, direct.base_cohort_id)
        assert db.items(facade) == cohort.items(con, direct)
    assert db.all_items().size == len(fx["item_ids"])
    assert db.describe(a) == cohort.describe(con, a)
    assert a.cultivar_state_versions and all(v["state_version"] == 0 for v in a.cultivar_state_versions.values())
    with pytest.raises(cohort.UnknownItemsError):
        db.cohort(["nope"])


def test_export_cohort_is_ids_and_provenance_only(loaded):
    """D-033: an ID list plus provenance; no linkage."""
    db, fx = loaded
    c = db.cohort(_items(fx, 4))
    out = db.export_cohort(c)
    assert set(out) == {"cohort_id", "membership_hash", "item_ids", "cultivar_state_versions"}
    assert out["item_ids"] == _items(fx, 4)
    assert out["membership_hash"] == c.membership_hash and out["cohort_id"] == str(c.cohort_id)
    assert out["cultivar_state_versions"] == c.cultivar_state_versions


def test_project_matches_module(loaded):
    db, fx = loaded
    con = db.connection
    c = db.cohort(_items(fx, 12))
    for cv in (x["name"] for x in fx["cultivars"]):
        a = db.project(c, cv, output="both")
        b = project.project(con, c, cv, output="both")
        assert (a.raw_rows, a.state_version, a.moved, a.present_items, a.missing_items) == \
               (b.raw_rows, b.state_version, b.moved, b.present_items, b.missing_items)
        assert np.array_equal(a.z, b.z) and a.leaf_item_ids == b.leaf_item_ids


def test_execute_and_explain_match_modules(loaded):
    db, fx = loaded
    con = db.connection
    cv = fx["cultivars"][0]["name"]
    items = _items(fx, 10)
    base = db.cohort(items)
    node = parent_branch(db, cv, items[0])
    q = Query.from_json({"rules": [{"cultivar": cv, "node": str(node), "op": "include"}]})
    a, b = db.execute(base, q), execute(con, base, q)
    assert a.items == b.items and a.cohort.kind == "execute" == b.cohort.kind
    assert a.cohort.membership_hash == b.cohort.membership_hash and a.sql == b.sql
    assert items[0] in a.items
    assert db.explain(a, items[0]) == explain(con, a, items[0])
    assert db.explain(q, items[0], base=base) == explain(con, q, items[0], base=base)


def test_mutations_match_module_semantics_and_report_state_version(loaded):
    """attach/detach are ``mutate.attach_node`` / ``detach_node``; the result carries the new version."""
    db, fx = loaded
    cv = fx["cultivars"][0]["name"]
    items = _items(fx, 6)
    victim = items[0]
    row = orchard_row(db, cv, victim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = db.detach(row)
    assert d.state_version == 1 and d.tree_name == cv
    assert read.tree_info(db.connection, cv)["state_version"] == 1
    target = orchard_row(db, cv, items[1])
    a = db.attach(item_uuid(victim), target, 1)
    assert a.state_version == 2 and a.changed
    assert db.node(cv, item_uuid(victim)).state_version == 2
    assert db.ancestry(cv, [victim]).resolution.state_version == 2


def test_moved_warning_and_on_moved_error_pass_through(loaded):
    """D-021: the facade passes on_moved straight through; CohortMovedWarning reaches the caller."""
    db, fx = loaded
    cv = fx["cultivars"][0]["name"]
    items = _items(fx, 8)
    c = db.cohort(items)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db.detach(orchard_row(db, cv, items[-1]))
    with pytest.warns(project.CohortMovedWarning):
        p = db.project(c, cv)
    assert p.moved is not None and p.moved.recorded_state_version == 0 and p.moved.current_state_version == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(project.CohortMovedError):
            db.project(c, cv, on_moved="error")
    with pytest.raises(read.MissingItemsError):
        db.project(c, cv, on_missing="error")


def test_api_does_not_import_scikit_learn_at_load():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", "import sys, orchardlogic.treedb.api; print('sklearn' in sys.modules)"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


