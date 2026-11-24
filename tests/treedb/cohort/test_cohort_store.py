"""Hashes, unknown ids, immutability, mutation, SQL macros, persistence, AppWorld (INV-17, INV-18, INV-20)."""

from __future__ import annotations

import inspect

import pytest

from cohort_helpers import import_twocult8, new_store, topology_to_z, twocult8_topology
from conftest import load_named
from orchardlogic.treedb import cohort, ingest, store


@pytest.fixture()
def con():
    c = new_store()
    import_twocult8(c)
    return c


def test_install_is_idempotent(con):
    cohort.install(con)
    cohort.install(con)
    assert cohort.integrity_violations(con) == {}


def test_INV17_equal_membership_equal_hash_distinct_handles(con):
    a = cohort.from_ids(con, ["i3", "i1", "i2"])
    b = cohort.from_ids(con, ["i1", "i2", "i3", "i1"])  # order and duplicates do not matter
    c = cohort.union(con, cohort.from_ids(con, ["i1"]), cohort.from_ids(con, ["i2", "i3"]))
    assert a.membership_hash == b.membership_hash == c.membership_hash
    assert len({a.cohort_id, b.cohort_id, c.cohort_id}) == 3
    assert cohort.from_ids(con, ["i1"]).membership_hash != a.membership_hash
    assert len(a.membership_hash) == 64 and a.size == b.size == 3


def test_INV20_unknown_ids_are_listed_not_dropped(con):
    with pytest.raises(cohort.UnknownItemsError) as ei:
        cohort.from_ids(con, ["i1", "zz9", "yy8", "i2"])
    assert ei.value.unknown == ["yy8", "zz9"]
    assert "'zz9'" in str(ei.value) and "'yy8'" in str(ei.value)
    assert con.execute("SELECT count(*) FROM treedb.cohort").fetchone()[0] == 0  # nothing created
    with pytest.raises(cohort.CohortError):
        cohort.from_ids(con, ["i1"], on_unknown="drop")
    with pytest.raises(TypeError):
        cohort.from_ids(con, ["i1", 7])


def test_INV17_no_update_path_and_prior_cohorts_stay_retrievable(con):
    public = {
        n for n, f in inspect.getmembers(cohort, inspect.isfunction)
        if not n.startswith("_") and f.__module__ == cohort.__name__
    }
    assert not {n for n in public if any(w in n for w in ("update", "delete", "remove", "add", "set_", "mutate"))}
    src = inspect.getsource(cohort)
    assert "UPDATE treedb" not in src and "DELETE FROM" not in src
    a = cohort.from_ids(con, ["i1", "i2"])
    before = (cohort.get(con, a.cohort_id), cohort.items(con, a))
    b = cohort.from_ids(con, ["i3"])
    cohort.union(con, a, b)
    cohort.intersect(con, a, b)
    cohort.difference(con, a, b)
    assert (cohort.get(con, a.cohort_id), cohort.items(con, a)) == before
    with pytest.raises(cohort.UnknownCohortError):
        cohort.get(con, "00000000-0000-0000-0000-000000000000")
    ordered = con.execute("SELECT cohort_id FROM treedb.cohort ORDER BY created_seq").fetchall()
    assert [r[0] for r in ordered][:2] == [a.cohort_id, b.cohort_id]  # created_seq is monotone


def test_INV18_provenance_records_state_versions(con):
    a = cohort.from_ids(con, ["i1"])
    versions = {v["name"]: v["state_version"] for v in a.cultivar_state_versions.values()}
    assert versions == {"A": 0, "B": 0}


def test_INV17_INV21_mutation_does_not_change_cohorts_SIMULATED(con):
    """P3C is not merged in this branch, so a committed mutation is simulated by
    deleting a leaf's orchard row and bumping ``state_version`` (D-007).
    Membership never changes (INV-17) and the cohort records the state it saw (INV-21)."""
    a = cohort.from_ids(con, ["i1", "i2", "i3"])
    whole = cohort.all_items(con)
    snap = {c.cohort_id: (cohort.items(con, c), c.membership_hash) for c in (a, whole)}
    leaf = con.execute("SELECT item_uuid FROM treedb.item_registry WHERE item_id = 'i1'").fetchone()[0]
    tree_a = con.execute("SELECT id FROM treedb.tree_registry WHERE name = 'A'").fetchone()[0]
    con.execute(
        "DELETE FROM treedb.orchard WHERE node_id = ? AND branch_id IN "
        "(SELECT branch_id FROM treedb.tree_branches WHERE tree_id = ?)", [leaf, tree_a])
    con.execute("UPDATE treedb.tree_registry SET state_version = state_version + 1 WHERE name = 'A'")
    for c in (a, whole):
        assert (cohort.items(con, c), cohort.get(con, c.cohort_id).membership_hash) == snap[c.cohort_id]
    d = cohort.describe(con, a)
    assert d["moved_cultivars"] == [str(tree_a)]
    assert d["cultivar_state_versions"][str(tree_a)]["state_version"] == 0
    assert d["cultivar_state_now"][str(tree_a)]["state_version"] == 1
    # a cohort created after the mutation records the new version
    assert cohort.describe(con, cohort.from_ids(con, ["i1"]))["moved_cultivars"] == []


def test_OD4_D021_real_split_keeps_membership_and_reports_the_cultivar_moved(con):
    """P3C follow-up F2 (STATUS P3C record), re-pointed from a simulated split (a raw
    ``INSERT`` of a new registry row) to a real P3C root-child detach (D-016c) of
    cultivar A: detach A1 (items i1..i4), a child of A's root A0.

    D-021: the cohort is frozen membership, so items and hash are unchanged and A is
    reported as ``moved`` (its ``state_version`` rose from 0 to 1); this never breaks.
    D-024 Q1: the split creates NO new ``tree_registry`` row (match PG): the registry has
    the same rows before and after, and B is not moved."""
    from orchardlogic.treedb import canonical, mutate
    from orchardlogic.treedb.errors import TreeSplitWarning

    a = cohort.from_ids(con, ["i1", "i5", "i8"])
    snap = (cohort.items(con, a), cohort.get(con, a.cohort_id).membership_hash)
    registry = lambda: con.execute("SELECT id, name FROM treedb.tree_registry ORDER BY name").fetchall()  # noqa: E731
    reg_before = registry()
    tree_a, tree_b = (con.execute("SELECT id FROM treedb.tree_registry WHERE name = ?", [n]).fetchone()[0]
                      for n in ("A", "B"))
    root_a = con.execute("SELECT root_branch_id FROM treedb.tree_registry WHERE id = ?", [tree_a]).fetchone()[0]
    members = canonical.node_members(con, tree_a)
    a1 = next(n for n, m in members.items() if tuple(m) == ("i1", "i2", "i3", "i4"))
    row = mutate.find_orchard_id(con, a1, tree_a)
    assert con.execute("SELECT branch_id FROM treedb.orchard WHERE id = ?", [row]).fetchone()[0] == root_a

    with pytest.warns(TreeSplitWarning):
        res = mutate.detach_node(con, row)
    assert res.split and res.stored_root_deleted and res.tree_id == tree_a and res.state_version == 1
    assert set(res.roots_after) == {a1, res.sibling_node_id}

    assert registry() == reg_before                                     # D-024 Q1: no new registry row
    assert (cohort.items(con, a), cohort.get(con, a.cohort_id).membership_hash) == snap   # D-021: frozen
    d = cohort.describe(con, a)
    assert d["moved_cultivars"] == [str(tree_a)]                         # D-021: notice, not an error
    assert d["cultivar_state_versions"][str(tree_a)]["state_version"] == 0
    assert d["cultivar_state_now"][str(tree_a)]["state_version"] == 1
    assert d["cultivar_state_now"][str(tree_b)]["state_version"] == 0
    assert cohort.integrity_violations(con) == {}


def test_sql_macros_read_membership(con):
    a = cohort.from_ids(con, ["i2", "i1"])
    got = con.execute("SELECT item_id FROM treedb.cohort_items(?) ORDER BY item_id", [str(a.cohort_id)]).fetchall()
    assert [r[0] for r in got] == ["i1", "i2"]
    b = cohort.from_ids(con, ["i2", "i3"])
    both = con.execute(
        "SELECT x.item_id FROM treedb.cohort_items(?) x JOIN treedb.cohort_items(?) y USING (item_id) "
        "JOIN treedb.item_registry r USING (item_id)", [a.cohort_id, str(b.cohort_id)]).fetchall()
    assert both == [("i2",)]
    assert con.execute("SELECT size, kind FROM treedb.cohort_info(?)", [str(a.cohort_id)]).fetchall() == [(2, "from_ids")]
    assert [r[0] for r in cohort.items(con, a, relation=True).fetchall()] == ["i1", "i2"]


def test_persistence_file_reopen_matches_memory(tmp_path):
    path = tmp_path / "orchard.duckdb"
    results = []
    ids = None
    for target in (None, path):
        c = store.connect(target)
        cohort.install(c)
        import_twocult8(c)
        a = cohort.from_ids(c, ["i1", "i2"])
        b = cohort.union(c, a, cohort.from_ids(c, ["i7"]))
        results.append((cohort.items(c, b), b.membership_hash, b.kind, b.broadens_base, b.size))
        if target is not None:
            ids = (a.cohort_id, b.cohort_id)
            c.close()
    assert results[0] == results[1]
    c = store.connect(path)
    cohort.install(c)  # idempotent on reopen
    b2 = cohort.get(c, ids[1])
    assert (cohort.items(c, b2), b2.membership_hash, b2.kind, b2.broadens_base, b2.size) == results[0]
    assert b2.base_cohort_id == ids[0] and cohort.items(c, ids[0]) == ["i1", "i2"]
    assert c.execute("SELECT count(*) FROM treedb.cohort_items(?)", [str(ids[0])]).fetchone()[0] == 2
    new = cohort.from_ids(c, ["i3"])  # sequence continues after reopen
    assert new.created_seq > b2.created_seq
    assert cohort.integrity_violations(c) == {}
    c.close()


def test_transaction_owned_by_caller_is_respected(con):
    con.execute("BEGIN TRANSACTION")
    a = cohort.from_ids(con, ["i1"])
    con.execute("ROLLBACK")
    with pytest.raises(cohort.UnknownCohortError):
        cohort.get(con, a.cohort_id)


def _item_structure(z, item_ids):
    """Resolve Z leaf indices through the cultivar's own item_ids (fixtures/SCHEMA.md:
    a cultivar may carry its own order). Returns (leaf pairs, member sets, depth, sizes)."""
    n = len(item_ids)
    members = {i: frozenset([item_ids[i]]) for i in range(n)}
    depth = {i: 0 for i in range(n)}
    pairs = set()
    for k, row in enumerate(z):
        l, r = int(row[0]), int(row[1])
        if l < n and r < n:
            pairs.add(frozenset((item_ids[l], item_ids[r])))
        members[n + k] = members[l] | members[r]
        depth[n + k] = 1 + max(depth[l], depth[r])
    root = n + len(z) - 1
    return pairs, {members[n + k] for k in range(len(z))}, depth[root], [len(members[n + k]) for k in range(len(z))]


def test_D011_twocult8_vs_F5_reconciliation():
    """D-011: P5A's twocult8 (i1..i8, cultivars A/B) is not P1's F5 (s0..s7, alpha/beta).
    Compared at item level: F5's beta has its own item_ids order, so its leaf pairing
    differs from alpha's even though the raw Z leaf indices coincide. Both fixtures have
    two cultivars with different topologies; they differ in item ids, names and shape
    (F5 both balanced 4-2-1; twocult8 A and B unbalanced). Fixtures are not edited."""
    f5 = load_named("F5_twocult8")
    st = {}
    for cv in f5["cultivars"]:
        st[cv["name"]] = _item_structure(cv["z"], cv.get("item_ids", f5["item_ids"]))
    assert set(st) == {"alpha", "beta"}
    a_pairs, a_sets, a_depth, a_sizes = st["alpha"]
    b_pairs, b_sets, b_depth, b_sizes = st["beta"]
    S = lambda *x: frozenset(x)
    assert a_pairs == {S("s0", "s1"), S("s2", "s3"), S("s4", "s5"), S("s6", "s7")}
    assert b_pairs == {S("s0", "s4"), S("s1", "s5"), S("s2", "s6"), S("s3", "s7")}
    assert a_pairs != b_pairs and a_sets != b_sets
    assert a_depth == b_depth == 3
    assert a_sizes == b_sizes == [2, 2, 2, 2, 4, 4, 8]

    topo = twocult8_topology()
    tw = {}
    for name, cv in topo["cultivars"].items():
        z = topology_to_z(topo["items"], cv["nodes"], cv["root"])
        tw[name] = _item_structure(z, topo["items"])
    assert set(tw) == {"A", "B"}
    assert tw["A"][0] != tw["B"][0] and tw["A"][1] != tw["B"][1]
    assert tw["A"][2] == tw["B"][2] == 4  # both depth 4 (not balanced depth 3); shapes differ via pairs/sets
    assert tw["A"][3] != [2, 2, 2, 2, 4, 4, 8]  # unbalanced, not 4-2-1
    assert tw["B"][3] != [2, 2, 2, 2, 4, 4, 8]
    assert set(topo["items"]).isdisjoint(f5["item_ids"])
    assert len(f5["item_ids"]) == len(topo["items"]) == 8


