"""Cohort algebra laws (INV-17, INV-19) over the 8 items of P5A's twocult8."""

from __future__ import annotations

import itertools

import pytest

from cohort_helpers import import_twocult8, new_store
from orchardlogic.treedb import cohort

ITEMS = [f"i{k}" for k in range(1, 9)]
SUBSETS = [
    [], ["i1"], ["i1", "i2"], ["i1", "i2", "i3", "i4"], ["i3", "i4", "i5"],
    ["i5", "i6", "i7", "i8"], ITEMS,
]


@pytest.fixture(scope="module")
def con():
    c = new_store()
    import_twocult8(c)
    return c


def mk(con, ids):
    return cohort.from_ids(con, ids)


def same(con, a, b) -> bool:
    return cohort.items(con, a) == cohort.items(con, b) and a.membership_hash == b.membership_hash


@pytest.mark.parametrize("xs,ys", list(itertools.product(SUBSETS, SUBSETS)))
def test_INV17_commutativity_and_subset_laws(con, xs, ys):
    a, b = mk(con, xs), mk(con, ys)
    assert same(con, cohort.union(con, a, b), cohort.union(con, b, a))
    assert same(con, cohort.intersect(con, a, b), cohort.intersect(con, b, a))
    u = cohort.union(con, a, b)
    assert cohort.items(con, u) == sorted(set(xs) | set(ys))
    assert cohort.items(con, cohort.intersect(con, a, b)) == sorted(set(xs) & set(ys))
    d = cohort.difference(con, a, b)
    assert cohort.items(con, d) == sorted(set(xs) - set(ys))
    # INV-19: difference and intersection are subsets of the left operand (the base)
    assert cohort.is_subset(con, d, a)
    assert cohort.is_subset(con, cohort.intersect(con, a, b), a)


@pytest.mark.parametrize("xs,ys,zs", [(SUBSETS[2], SUBSETS[3], SUBSETS[4]), (SUBSETS[1], SUBSETS[5], SUBSETS[6])])
def test_INV17_associativity(con, xs, ys, zs):
    a, b, c = mk(con, xs), mk(con, ys), mk(con, zs)
    assert same(con, cohort.union(con, cohort.union(con, a, b), c), cohort.union(con, a, cohort.union(con, b, c)))
    assert same(con, cohort.intersect(con, cohort.intersect(con, a, b), c),
                cohort.intersect(con, a, cohort.intersect(con, b, c)))
    assert same(con, cohort.union(con, a, b, c), cohort.union(con, cohort.union(con, a, b), c))


@pytest.mark.parametrize("xs", SUBSETS)
def test_INV17_idempotence(con, xs):
    a = mk(con, xs)
    assert same(con, cohort.union(con, a, a), a)
    assert same(con, cohort.intersect(con, a, a), a)
    assert cohort.items(con, cohort.difference(con, a, a)) == []


def complement(con, universe, a):
    return cohort.difference(con, universe, a)


@pytest.mark.parametrize("xs,ys", list(itertools.product(SUBSETS[:5], SUBSETS[2:])))
def test_INV17_de_morgan_relative_to_all_items(con, xs, ys):
    u = cohort.all_items(con)
    a, b = mk(con, xs), mk(con, ys)
    # not(A or B) == not(A) and not(B);  not(A and B) == not(A) or not(B)
    assert same(con, complement(con, u, cohort.union(con, a, b)),
                cohort.intersect(con, complement(con, u, a), complement(con, u, b)))
    assert same(con, complement(con, u, cohort.intersect(con, a, b)),
                cohort.union(con, complement(con, u, a), complement(con, u, b)))


def test_INV19_union_broadening_is_explicit_and_not_an_execute(con):
    base = mk(con, ["i1", "i2", "i3", "i4"])
    inside = mk(con, ["i1", "i2"])
    outside = mk(con, ["i5", "i6"])
    quiet = cohort.union(con, base, inside)
    broad = cohort.union(con, base, outside)
    assert quiet.kind == broad.kind == "union"
    assert not quiet.broadens_base and broad.broadens_base
    assert broad.operands["added_outside_base"] == 2
    assert cohort.describe(con, broad)["broadens_base"] is True
    # An execute result must be a subset of its base: reaching outside is refused.
    ok = cohort.create(con, ["i1", "i2"], "execute", base=base, expression={"q": 1})
    assert ok.kind == "execute" and not ok.broadens_base and ok.expression == {"q": 1}
    with pytest.raises(cohort.CohortError, match="not a subset"):
        cohort.create(con, ["i1", "i5"], "execute", base=base)
    with pytest.raises(cohort.CohortError, match="needs a base"):
        cohort.create(con, ["i1"], "execute")


def test_INV19_q16c_successive_execute_then_explicit_union(con):
    """P5A Q16c shape: two Execute results then an explicit union (kind union, not execute)."""
    c0 = cohort.all_items(con)
    c1 = cohort.create(con, ["i1", "i2", "i3", "i4"], "execute", base=c0)
    c2 = cohort.create(con, ["i5", "i6", "i7", "i8"], "execute", base=c0)
    c3 = cohort.union(con, c1, c2)
    assert cohort.items(con, c3) == ITEMS
    assert c3.kind == "union" and c3.broadens_base and c3.base_cohort_id == c1.cohort_id
    assert cohort.is_subset(con, c1, c0) and cohort.is_subset(con, c2, c0)


def test_INV18_provenance_records_kind_and_operands(con):
    a, b = mk(con, ["i1"]), mk(con, ["i2"])
    for fn, kind in ((cohort.union, "union"), (cohort.intersect, "intersect"), (cohort.difference, "difference")):
        r = fn(con, a, b)
        d = cohort.describe(con, r)
        assert d["kind"] == kind
        assert d["operands"]["cohorts"] == [str(a.cohort_id), str(b.cohort_id)]
        assert d["base_cohort_id"] == str(a.cohort_id)
    assert cohort.describe(con, a)["base_cohort_id"] is None
