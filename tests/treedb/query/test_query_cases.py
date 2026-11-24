"""Every fixtures/query_cases case, as accepted at G3 (D-025), runs on F7 twocult8_shared.

Invariants: INV-17/19 (result is a new subset cohort), and the D-025 rules named in each case.
Also: explain agrees with execute on every item of every step, and permuting rules never changes the outcome.
"""

from __future__ import annotations

import itertools

import pytest

from query_helpers import case_files, load_case, shared_store, translate
from orchardlogic.treedb import cohort
from orchardlogic.treedb.query import Query, QueryError, execute, explain

CASES = case_files()
STEPS = [(p.stem, i) for p in CASES for i, s in enumerate(load_case(p)["steps"]) if "query" in s]


def run_case(con, case, *, permute_rules=None):
    """Run all steps. Yields (step, outcome) where outcome is ('items', ids, result) or ('error', code)."""
    handles = {}
    for name, spec in case["cohorts"].items():
        handles[name] = cohort.all_items(con) if spec == "all" else cohort.from_ids(con, spec)
    for step in case["steps"]:
        if step.get("op") == "cohort_union":
            c = cohort.union(con, *(handles[o] for o in step["operands"]))
            handles[step["result_as"]] = c
            yield step, ("cohort", c, None)
            continue
        q = translate(con, step["query"])
        if permute_rules is not None and "rules" in q:
            q["rules"] = permute_rules(q["rules"])
        try:
            res = execute(con, handles[step["query"]["base"]["cohort"]], Query.from_json(q))
        except QueryError as e:
            yield step, ("error", e.code, None)
            return
        if "result_as" in step:
            handles[step["result_as"]] = res.cohort
        yield step, ("items", list(res.items), res)


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_case_expectation(path):
    case = load_case(path)
    con = shared_store("F7_twocult8")
    for step, outcome in run_case(con, case):
        exp = step["expected"]
        if "error" in exp:
            assert outcome[:2] == ("error", exp["error"]), step
        elif outcome[0] == "cohort":
            assert cohort.items(con, outcome[1]) == exp["items"]
            assert outcome[1].kind == "union" and outcome[1].broadens_base, "Q16c: explicit union is distinguishable"
        else:
            assert outcome[1] == exp["items"], step
            res = outcome[2]
            assert res.cohort.kind == "execute"
            assert list(res.ignored_rules) == exp.get("ignored_rules", [])


@pytest.mark.parametrize("path", [p for p in CASES if load_case(p).get("permute")], ids=lambda p: p.stem)
def test_case_is_order_independent(path):
    """INV-19 / D-025: the outcome depends on the set of rules, never on their order."""
    case = load_case(path)
    rules = case["steps"][0]["query"].get("rules", [])
    expected = [o[:2] if o[0] == "error" else o[1] for _, o in run_case(shared_store("F7_twocult8"), case)]
    for perm in itertools.permutations(range(len(rules))):
        got = [o[:2] if o[0] == "error" else o[1]
               for _, o in run_case(shared_store("F7_twocult8"), case, permute_rules=lambda rs, p=perm: [rs[i] for i in p])]
        assert got == expected, perm


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_explain_agrees_with_execute_on_every_item(path):
    """Acceptance 3: explain reproduces execute's decision for every registered item."""
    con = shared_store("F7_twocult8")
    all_ids = [r[0] for r in con.execute("SELECT item_id FROM treedb.item_registry ORDER BY item_id").fetchall()]
    for step, outcome in run_case(con, load_case(path)):
        if outcome[0] != "items":
            continue
        res = outcome[2]
        for item in all_ids:
            ex = explain(con, res, item)
            assert ex.verdict == (item in res.items), (path.stem, item, ex.reason)


def test_all_case_files_are_covered():
    assert len(CASES) == 35
    assert len(STEPS) >= 35
    for p in CASES:
        for s in load_case(p)["steps"]:
            if "query" in s:
                assert "expected" in s and "option_axis" not in s and "expected_by_option" not in s
