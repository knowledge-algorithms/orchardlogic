"""Grammar, rejection, D-025 semantics (nested include, item rules) and reporting (phase P5B).

Invariants: INV-19, INV-20 (nothing dropped silently), D-022, D-025.
"""

from __future__ import annotations

import json

import pytest

from query_helpers import node_by_label, shared_store
from orchardlogic.treedb import cohort, read
from orchardlogic.treedb.query import (CODES, ItemRule, NodeRule, Query, QueryError, compile_query, execute, explain,
                          explain_sql, normalize)

QV = "treedb.query/1"


def con7():
    return shared_store("F7_twocult8")


def run(rules, base=None, con=None):
    con = con or con7()
    return execute(con, base or cohort.all_items(con), Query.from_json({"version": QV, "rules": rules}))


def n(cultivar, label, op):
    return {"cultivar": cultivar, "node": node_by_label(con7(), cultivar, label), "op": op}


# --- grammar and rejection ------------------------------------------------------------------------------------

@pytest.mark.parametrize("obj,code", [
    ({"any_of": []}, "GROUPING_UNSUPPORTED"),
    ({"rules": [], "not": {}}, "GROUPING_UNSUPPORTED"),
    ({"version": "0.1-candidate", "rules": []}, "UNSUPPORTED_VERSION"),
    ({"rules": [{"item": "i1", "op": "neutral"}]}, "NEUTRAL_UNSUPPORTED"),
    ({"rules": [{"item": "i1", "op": "include", "replace": True}]}, "REPLACE_UNSUPPORTED"),
    ({"rules": [{"item": "i1", "op": "maybe"}]}, "INVALID_RULE"),
    ({"rules": [{"item": "i1", "cultivar": "A", "node": "x", "op": "include"}]}, "INVALID_RULE"),
    ({"rules": "x"}, "INVALID_QUERY"),
    ({"base": {"cohort": "c0"}, "rules": []}, "INVALID_QUERY"),
])
def test_shape_errors(obj, code):
    with pytest.raises(QueryError) as e:
        Query.from_json(obj)
    assert e.value.code == code and code in CODES


def test_store_errors():
    con = con7()
    for rules, code in [
        ([{"cultivar": "Z", "node": node_by_label(con, "A", "A1"), "op": "include"}], "UNKNOWN_CULTIVAR"),
        ([{"cultivar": "A", "node": node_by_label(con, "B", "B1"), "op": "include"}], "UNKNOWN_NODE"),  # B's node, named under A
        ([{"cultivar": "A", "node": "not-a-uuid", "op": "include"}], "UNKNOWN_NODE"),
        ([{"item": "nope", "op": "include"}], "UNKNOWN_ITEM"),
        ([n("A", "A1", "include"), n("A", "A1", "exclude")], "CONFLICT_NODE"),
        ([{"item": "i1", "op": "include"}, {"item": "i1", "op": "exclude"}], "CONFLICT_ITEM"),
    ]:
        with pytest.raises(QueryError) as e:
            normalize(con, Query.from_json({"rules": rules}))
        assert e.value.code == code, rules


def test_leaf_is_not_a_node_target():
    con = con7()
    leaf = str(con.execute("SELECT item_uuid FROM treedb.item_registry WHERE item_id = 'i1'").fetchone()[0])
    with pytest.raises(QueryError) as e:
        normalize(con, Query.from_json({"rules": [{"cultivar": "A", "node": leaf, "op": "include"}]}))
    assert e.value.code == "UNKNOWN_NODE"


def test_same_node_in_two_cultivars_is_not_a_conflict():
    """Rules on different cultivars never conflict (no cross-cultivar contest)."""
    r = run([n("A", "A3", "include"), n("B", "B3", "exclude")])
    assert r.items == ("i2",)


def test_json_round_trip_and_normal_form_is_sorted_and_deduplicated():
    con = con7()
    q = Query.from_json({"rules": [n("B", "B1", "include"), {"item": "i2", "op": "exclude"}, n("A", "A1", "include"),
                                   n("A", "A1", "include"), {"item": "i2", "op": "exclude"}]})
    nq = normalize(con, q)
    assert len(nq.rules) == 3 and [r.cultivar for r in nq.node_rules] == ["A", "B"]
    assert Query.from_json(q.dumps()) == q and Query.from_json(json.dumps(nq.to_json())) == nq.as_query()


# --- D-025 semantics, worked examples ----------------------------------------------------------------------------

def test_nested_include_only_restores_q01():
    assert run([n("A", "A1", "exclude"), n("A", "A3", "include")]).items == ("i1", "i2", "i5", "i6", "i7", "i8")


def test_top_level_include_selects_and_nested_exclude_removes():
    assert run([n("A", "A1", "include"), n("A", "A3", "exclude")]).items == ("i3", "i4")


def test_item_include_alone_narrows_to_that_item():
    assert run([{"item": "i3", "op": "include"}]).items == ("i3",)
    assert run([{"item": "i3", "op": "include"}, {"item": "i5", "op": "include"}]).items == ("i3", "i5")


def test_item_exclude_alone_removes_only_that_item():
    assert run([{"item": "i3", "op": "exclude"}]).items == ("i1", "i2", "i4", "i5", "i6", "i7", "i8")


def test_item_include_nested_under_an_exclude_restores():
    assert run([n("A", "A1", "exclude"), {"item": "i3", "op": "include"}]).items == ("i3", "i5", "i6", "i7", "i8")


def test_item_include_beside_node_include_is_a_union_in_that_cultivar():
    assert run([n("A", "A3", "include"), {"item": "i5", "op": "include"}]).items == ("i1", "i2", "i5")


def test_item_rules_count_only_in_node_constrained_cultivars():
    """Q08b: the exclusion is in B only, so the item include restores in B and does not make A select."""
    assert run([n("B", "B1", "exclude"), {"item": "i1", "op": "include"}]).items == ("i1", "i3", "i4", "i7", "i8")


def test_item_include_outside_every_excluded_chain_selects_where_the_cultivar_is_constrained():
    """B excludes B4={i2,i6}; i1 is not under B4, so i1's include is top-level in B and B narrows to {i1}."""
    assert run([n("B", "B4", "exclude"), {"item": "i1", "op": "include"}]).items == ("i1",)


def test_item_include_restores_in_one_cultivar_and_selects_in_another():
    """A selects A3={i1,i2}, B excludes B1 and i1 sits under B1: restored in B, selecting-union in A."""
    assert run([n("A", "A3", "include"), n("B", "B1", "exclude"), {"item": "i1", "op": "include"}]).items == ("i1",)


def test_empty_result_is_valid_and_empty_rules_keep_the_base():
    assert run([n("A", "A3", "include"), n("B", "B1", "exclude")]).items == ()
    assert len(run([]).items) == 8


def test_rule_on_the_root_node():
    con = con7()
    root = read.tree_info(con, "A")["root_branch_id"]
    assert len(run([{"cultivar": "A", "node": str(root), "op": "include"}]).items) == 8
    assert run([{"cultivar": "A", "node": str(root), "op": "exclude"}]).items == ()


# --- reporting ---------------------------------------------------------------------------------------------

def test_out_of_base_item_rule_is_reported_and_has_no_effect():
    con = con7()
    base = cohort.from_ids(con, ["i1", "i2", "i3", "i4"])
    res = run([n("A", "A3", "include"), {"item": "i7", "op": "include"}], base)
    assert res.items == ("i1", "i2")
    assert res.ignored_rules == ({"item": "i7", "op": "include", "reason": "out_of_base"},)
    assert cohort.get(con, res.cohort.cohort_id).operands["ignored_rules"] == list(res.ignored_rules)
    # alone, the ignored rule is as if absent: the base stays whole
    assert run([{"item": "i7", "op": "include"}], base).items == ("i1", "i2", "i3", "i4")
    ex = explain(con, res, "i7")
    assert ex.in_base is False and ex.ignored_item_rule["reason"] == "out_of_base" and not ex.verdict


def test_missing_items_are_reported_for_constrained_cultivars_d022():
    con = shared_store("F6_partial")
    root = str(read.tree_info(con, "partial")["root_branch_id"])
    sel = execute(con, cohort.all_items(con), Query.from_json({"rules": [{"cultivar": "partial", "node": root, "op": "include"}]}))
    assert sel.missing_items == {"partial": ("p6", "p7")} and sel.items == tuple(f"p{i}" for i in range(6))
    node = next(str(b) for (b,) in con.execute(
        "SELECT o.branch_id FROM treedb.v_orchard o JOIN treedb.tree_registry t ON t.id = o.tree_id "
        "WHERE t.name = 'partial' AND o.branch_id <> ? ORDER BY o.branch_id", [root]).fetchall())
    keep = execute(con, cohort.all_items(con), Query.from_json({"rules": [{"cultivar": "partial", "node": node, "op": "exclude"}]}))
    assert {"p6", "p7"} <= set(keep.items) and keep.missing_items == {"partial": ("p6", "p7")}
    assert explain(con, keep, "p6").cultivars[0].present is False


def test_execute_records_provenance_and_has_no_timestamp_in_the_expression():
    con = con7()
    base = cohort.all_items(con)
    res = execute(con, base, Query.from_json({"rules": [n("A", "A3", "include")]}))
    c = cohort.get(con, res.cohort.cohort_id)
    assert c.kind == "execute" and c.base_cohort_id == base.cohort_id
    assert set(c.expression) == {"version", "rules"} and c.expression == res.normalized.to_json()
    assert {v["name"] for v in c.cultivar_state_versions.values()} == {"A", "B"}


# --- compile / explain_sql -----------------------------------------------------------------------------------

def test_explain_sql_is_readable_and_equals_what_execute_runs():
    con = con7()
    base = cohort.all_items(con)
    res = execute(con, base, Query.from_json({"rules": [n("A", "A1", "exclude"), {"item": "i3", "op": "include"}]}))
    text = explain_sql(res.normalized, base.cohort_id)
    assert text == res.sql == compile_query(res.normalized, base.cohort_id)
    for cte in ("base AS", "node_rules(", "item_rules(", "constrained(", "item_chain(", "applicable(", "nearest(",
                "selecting(", "verdict("):
        assert cte in text
    assert "<base cohort id>" in explain_sql(res.normalized)
    assert [r[0] for r in con.execute(text).fetchall()] == list(res.items)


def test_explain_structure_for_q01():
    con = con7()
    res = run([n("A", "A1", "exclude"), n("A", "A3", "include")])
    ex = explain(con, res, "i1")   # under A1 (depth 1, exclude) and A3 (depth 2, include): A3 is nearer
    a = ex.cultivars[0]
    assert [c.cultivar for c in ex.cultivars] == ["A"] and ex.verdict and a.keep
    assert [(r["kind"], r["op"], r["depth"]) for r in a.applicable] == [("node", "include", 2), ("node", "exclude", 1)]
    assert a.winner["op"] == "include" and not a.selecting and a.chain_depth == 3
    ex3 = explain(con, res, "i3")
    assert not ex3.verdict and ex3.cultivars[0].winner["depth"] == 1
    assert explain(con, res, "i5").cultivars[0].winner is None
