"""Seeded, deterministic property tests on F5 twocult8 and AppWorld (phase P5B).

Invariants: INV-17 (cohort handles stay put), INV-19 (result is a subset of its base; no
resurrection; broadening only by an explicit union) and the D-025 order-independence rules.
"""

from __future__ import annotations

import random

import pytest

from query_helpers import random_query, shared_store
from orchardlogic.treedb import cohort
from orchardlogic.treedb.query import ItemRule, NodeRule, Query, execute, explain, normalize

STORES = {"F5": ("F5_twocult8", 40, 8)}


def _setup(name):
    stem, n, n_explain = STORES[name]
    con = shared_store(stem)
    cultivars = [r[0] for r in con.execute("SELECT name FROM treedb.tree_registry ORDER BY name").fetchall()]
    items = [r[0] for r in con.execute("SELECT item_id FROM treedb.item_registry ORDER BY item_id").fetchall()]
    return con, cultivars, items, n, n_explain


def _random_base(con, rng, items):
    if rng.random() < 0.4:
        return cohort.all_items(con)
    return cohort.from_ids(con, rng.sample(items, rng.randint(1, len(items))))


def _shuffled(q, rng):
    rules = list(q["rules"])
    rng.shuffle(rules)
    return {**q, "rules": rules}


@pytest.mark.parametrize("name", STORES)
def test_subset_order_independence_idempotence_explain(name):
    con, cultivars, items, n, n_explain = _setup(name)
    rng = random.Random(20261109)
    for _ in range(n):
        base = _random_base(con, rng, items)
        q = random_query(con, rng, cultivars, items)
        res = execute(con, base, Query.from_json(q))
        # INV-19: result is a subset of the base, kind execute, base recorded.
        assert set(res.items) <= set(cohort.items(con, base))
        assert res.cohort.kind == "execute" and res.cohort.base_cohort_id == base.cohort_id
        assert res.cohort.expression == res.normalized.to_json()
        # order independence of normal form and of the result
        again = execute(con, base, Query.from_json(_shuffled(q, rng)))
        assert again.normalized == res.normalized and again.items == res.items
        # idempotent normalization
        nq = normalize(con, Query.from_json(q))
        assert normalize(con, nq) == nq == normalize(con, normalize(con, nq).as_query())
        # explain reproduces execute's decision (all F5 items; a seeded sample of AppWorld's)
        for item in (items if name == "F5" else rng.sample(items, n_explain)):
            ex = explain(con, res, item)
            assert ex.verdict == (item in res.items), (item, ex.reason)


@pytest.mark.parametrize("name", STORES)
def test_narrowing_and_handles_stay_unchanged(name):
    """Execute on C1 returns a subset of C1; C0 and C1 are unchanged after C2 (INV-17, INV-19)."""
    con, cultivars, items, n, _ = _setup(name)
    rng = random.Random(7)
    for _ in range(max(3, n // 4)):
        c0 = cohort.all_items(con)
        c1 = execute(con, c0, Query.from_json(random_query(con, rng, cultivars, items))).cohort
        before = (cohort.get(con, c0), cohort.get(con, c1), cohort.items(con, c0), cohort.items(con, c1))
        c2 = execute(con, c1, Query.from_json(random_query(con, rng, cultivars, items))).cohort
        assert cohort.is_subset(con, c2, c1) and cohort.is_subset(con, c1, c0)
        assert before == (cohort.get(con, c0), cohort.get(con, c1), cohort.items(con, c0), cohort.items(con, c1))


@pytest.mark.parametrize("name", STORES)
def test_item_outside_base_never_resurrected(name):
    """INV-19, D-025: an item include for an item outside C1 is ignored and reported; it stays out."""
    con, cultivars, items, _, _ = _setup(name)
    rng = random.Random(11)
    for _ in range(4):
        c1 = execute(con, cohort.all_items(con), Query.from_json(random_query(con, rng, cultivars, items))).cohort
        members = set(cohort.items(con, c1))
        outside = [i for i in items if i not in members]
        if not outside:
            continue
        out = rng.choice(outside)
        q = random_query(con, rng, cultivars, items)
        q["rules"] = [r for r in q["rules"] if r.get("item") != out] + [{"item": out, "op": "include"}]
        res = execute(con, c1, Query.from_json(q))
        assert out not in res.items
        assert {"item": out, "op": "include", "reason": "out_of_base"} in res.ignored_rules
        # ignoring means ignoring: the same query without that rule gives the same result
        q2 = {**q, "rules": [r for r in q["rules"] if r.get("item") != out]}
        assert execute(con, c1, Query.from_json(q2)).items == res.items


def test_explicit_union_is_distinguishable_by_kind():
    con, cultivars, items, _, _ = _setup("F5")
    c0 = cohort.all_items(con)
    narrow = execute(con, c0, Query.from_json({"rules": [{"item": items[0], "op": "include"}]})).cohort
    wide = execute(con, c0, Query.from_json({"rules": [{"item": items[1], "op": "include"}]})).cohort
    u = cohort.union(con, narrow, wide)
    assert narrow.kind == wide.kind == "execute" and u.kind == "union" and u.broadens_base
    assert set(cohort.items(con, u)) == {items[0], items[1]}


def test_no_depth_contest_across_cultivars():
    """A deep include in one cultivar and a shallow exclude in the other: the exclusion wins."""
    con = shared_store("F7_twocult8")
    from query_helpers import node_by_label
    a5 = node_by_label(con, "A", "A5")   # depth 2 in A, holds i6 i7 i8
    b1 = node_by_label(con, "B", "B1")   # depth 1 in B, holds i1 i2 i5 i6
    res = execute(con, cohort.all_items(con), Query.from_json({"rules": [
        {"cultivar": "A", "node": a5, "op": "include"}, {"cultivar": "B", "node": b1, "op": "exclude"}]}))
    assert res.items == ("i7", "i8")
    ex = explain(con, res, "i6")
    by = {c.cultivar: c for c in ex.cultivars}
    assert by["A"].winner["depth"] == 2 and by["B"].winner["depth"] == 1   # reported side by side, never compared
    assert by["A"].keep and not by["B"].keep and not ex.verdict


def test_rule_dataclasses_are_hashable_values():
    assert NodeRule("A", "x", "include") == NodeRule("A", "x", "include")
    assert ItemRule("i1", "exclude") in {ItemRule("i1", "exclude")}
