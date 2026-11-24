"""P6A, architecture 8.10: a file store gives the same answers as memory and keeps its handles.

One scripted session runs in memory and on a file (with a close and reopen in the middle); the
structural results (item sets, hashes, versions, Z) are compared. Random node and cohort ids differ
between runs, so they are never compared across stores; they are checked to resolve instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from api_helpers import load_into, orchard_row, parent_branch
from orchardlogic.treedb import TreeDB
from orchardlogic.treedb.fixtures import item_uuid
from orchardlogic.treedb.project import CohortMovedWarning
from orchardlogic.treedb.query import Query


def _record(p):
    return {"v": p.state_version, "moved": None if p.moved is None else (p.moved.recorded_state_version,
            p.moved.current_state_version), "present": p.present_items, "missing": p.missing_items,
            "z": None if p.z is None else p.z.tolist(), "leaves": p.leaf_item_ids,
            "components": p.component_count}


def _session(open_db, reopen):
    """Return ``(results, handles)``. ``reopen(db)`` returns the db to continue with."""
    out: dict = {}
    db = open_db()
    fx = load_into(db, "F5_twocult8")
    items = sorted(fx["item_ids"])
    c1 = db.cohort(["s0", "s2", "s5"])
    rest = db.difference(db.all_items(), c1)
    out["cohorts"] = [(c.kind, c.size, c.membership_hash) for c in (c1, rest, db.union(c1, db.cohort(["s7"])))]
    out["before"] = {cv: _record(db.project(c1, cv, output="both")) for cv in ("alpha", "beta")}
    q = Query.from_json({"rules": [{"cultivar": "alpha", "node": str(parent_branch(db, "alpha", "s0")),
                                    "op": "include"}]})
    ex = db.execute(db.all_items(), q)
    out["execute"] = (ex.items, ex.cohort.membership_hash)
    handles = {"c1": c1.cohort_id, "ex": ex.cohort.cohort_id,
               "node": db.node("alpha", item_uuid("s2")).node_id,
               "branch": parent_branch(db, "beta", "s3")}
    out["alpha_state"] = db.cultivars()[0]["state_version"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db.detach(orchard_row(db, "alpha", "s2"))
    db = reopen(db)

    # handles resolve after the reopen
    assert db.node("beta", handles["branch"]).kind == "branch"
    assert db.items(handles["c1"]) == ["s0", "s2", "s5"] and db.items(handles["ex"]) == list(ex.items)
    d = db.describe(handles["c1"])
    out["moved_after_detach"] = sorted(v["name"] for k, v in d["cultivar_state_now"].items()
                                       if k in d["moved_cultivars"])
    assert out["moved_after_detach"] == ["alpha"]

    with pytest.warns(CohortMovedWarning):
        mid = db.project(handles["c1"], "alpha", output="both")
    out["after_detach"] = _record(mid)
    db.attach(item_uuid("s2"), orchard_row(db, "alpha", "s1"), 1)       # mutate after the reopen
    with pytest.warns(CohortMovedWarning):
        out["after_attach"] = _record(db.project(handles["c1"], "alpha", output="both"))
    out["beta_unmoved"] = _record(db.project(handles["c1"], "beta"))
    sub = db.execute(handles["ex"], Query.from_json({"rules": [{"item": "s0", "op": "exclude"}]}))
    out["sub_of_old_cohort"] = (sub.items, sub.cohort.base_cohort_id == handles["ex"])
    out["versions"] = {c["name"]: c["state_version"] for c in db.cultivars()}
    out["cohort_count"] = db.connection.execute("SELECT count(*) FROM treedb.cohort").fetchone()[0]
    db.close()
    return out


def test_memory_and_file_sessions_are_equivalent(tmp_path):
    mem = _session(TreeDB, lambda db: db)                                  # memory: no reopen possible
    path = tmp_path / "session.duckdb"

    def reopen(db):
        db.close()
        return TreeDB(path)

    file = _session(lambda: TreeDB(path), reopen)
    assert file == mem
    assert mem["versions"] == {"alpha": 2, "beta": 0}
    assert mem["after_detach"]["missing"] == ("s2",) and mem["after_attach"]["missing"] == ()
    assert mem["beta_unmoved"]["moved"] is None


def test_reopen_keeps_cultivars_cohorts_and_state_versions(tmp_path):
    path = tmp_path / "keep.duckdb"
    with TreeDB(path) as db:
        load_into(db, "F5_twocult8")
        c = db.cohort(["s1", "s3"])
        node = db.node("beta", item_uuid("s1")).node_id
    with TreeDB(path) as db:
        assert [x["name"] for x in db.cultivars()] == ["alpha", "beta"]
        assert db.items(c) == ["s1", "s3"] and db.describe(c)["membership_hash"] == c.membership_hash
        assert db.node("beta", node).item_id == "s1"
        assert db.export_cohort(c)["cultivar_state_versions"] == c.cultivar_state_versions
    with TreeDB(path) as db:                                               # a third open, same answers
        assert db.items(c) == ["s1", "s3"]


def test_file_named_treedb_is_refused(tmp_path):
    with pytest.raises(ValueError):
        TreeDB(tmp_path / "treedb.duckdb")


def test_failed_open_does_not_leak(tmp_path):
    bad = tmp_path / "bad.duckdb"
    bad.write_bytes(b"not a duckdb file" * 100)
    with pytest.raises(Exception):
        TreeDB(bad)
