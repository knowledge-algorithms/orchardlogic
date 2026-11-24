"""Session semantics: cohorts, ledger algebra, commits, exposure, hopper."""

from __future__ import annotations

import pytest

from orchardlogic.cultivate.catalog import build_fixture_catalog
from orchardlogic.cultivate.search import parse_term
from orchardlogic.cultivate.navigation import COMMIT_WARNING, Catalog, SessionManager


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return build_fixture_catalog()


@pytest.fixture
def manager(catalog: Catalog) -> SessionManager:
    manager = SessionManager(catalog)
    manager.dispatch("session", {"action": "open", "task_id": "t1"})
    return manager


def groups(response: dict, cultivar: str) -> list[dict]:
    block = next(b for b in response["cultivars"] if b["cultivar"] == cultivar)
    return block["groups"]


def members(manager: SessionManager, cohort_id: str, ref: str) -> set[str]:
    session = manager.get(None)
    cohort = session.cohorts[cohort_id]
    view, node_id = session._resolve_ref(cohort, ref)
    return set(view.members_of(node_id))


def first_tool(manager: SessionManager, cohort_id: str, ref: str) -> str:
    """Drill down until an individual tool card is shown; return its ID."""
    while True:
        result = manager.dispatch("explore", {"cohort_id": cohort_id, "refs": [ref]})["results"][0]
        if result.get("tool_cards"):
            return result["tool_cards"][0]["tool_id"]
        ref = result["groups"][0]["ref"]


def test_overview_creates_fresh_cohorts_and_resumes(manager: SessionManager, catalog: Catalog) -> None:
    first = manager.dispatch("overview", {})
    assert first["cohort_id"] == "C1" and first["tool_count"] == len(catalog.items)
    assert first["catalog_id"] == catalog.catalog_id and first["session_id"]
    assert {b["cultivar"] for b in first["cultivars"]} == {"domain", "function"}
    for entry in groups(first, "domain"):
        assert entry["retained_pct"] == 100.0 and entry["restricted"] is False
        assert entry["label"] and "summary" in entry

    again = manager.dispatch("overview", {"cohort_id": "new"})
    assert again["cohort_id"] == "C2"
    resumed = manager.dispatch("overview", {"cohort_id": "C1"})
    assert resumed["cohort_id"] == "C1"
    assert [g["ref"] for g in groups(resumed, "domain")] == [g["ref"] for g in groups(first, "domain")]

    status = manager.dispatch("session", {"action": "status"})
    assert status["continuation_cohort_id"] == "C2"
    assert status["counts"]["cohorts"] == 2 and "cohorts" not in status
    assert "unknown_cohort" == manager.dispatch("overview", {"cohort_id": "C9"})["error"]["code"]
    assert "cohort_required" == manager.dispatch("explore", {"refs": ["D1"]})["error"]["code"]


def test_explore_is_read_only_multi_node_and_alias_stable(manager: SessionManager) -> None:
    overview = manager.dispatch("overview", {})
    session = manager.get(None)
    d_ref, f_ref = groups(overview, "domain")[0]["ref"], groups(overview, "function")[0]["ref"]
    before = (list(session.cohorts["C1"].steps), session.cohorts["C1"].revision, dict(session.hopper))

    one = manager.dispatch("explore", {"cohort_id": "C1", "refs": [d_ref, f_ref]})
    two = manager.dispatch("explore", {"cohort_id": "C1", "refs": [f_ref, d_ref]})
    assert [r["cultivar"] for r in one["results"]] == ["domain", "function"]
    by_ref = {r["ref"]: r for r in two["results"]}
    for result in one["results"]:
        assert [g["ref"] for g in result.get("groups", [])] == [
            g["ref"] for g in by_ref[result["ref"]].get("groups", [])
        ]
    assert before == (list(session.cohorts["C1"].steps), session.cohorts["C1"].revision, dict(session.hopper))
    assert len(session.cohorts) == 1

    bad = manager.dispatch("explore", {"cohort_id": "C1", "refs": [d_ref, "D999"]})
    assert bad["error"]["code"] == "unknown_ref"
    # No executable schema ever appears on a card.
    assert "input_schema" not in str(one)


def test_selection_algebra_and_prospective_counts(manager: SessionManager) -> None:
    overview = manager.dispatch("overview", {})
    d = [g["ref"] for g in groups(overview, "domain")]
    f = [g["ref"] for g in groups(overview, "function")]
    m = lambda ref: members(manager, "C1", ref)  # noqa: E731
    everything = set(manager.get(None).cohorts["C1"].members)
    stage = lambda decisions: manager.dispatch(  # noqa: E731
        "filter", {"cohort_id": "C1", "decisions": decisions, "preview": True}
    )

    # Steps apply in order from all members: keep_only then add = union.
    out = stage([{"ref": d[0], "action": "keep_only"}, {"ref": f"domain.{d[1]}", "action": "add"}])
    assert out["committed"] is False
    assert out["prospective_count"] == len(m(d[0]) | m(d[1]))
    assert out["ledger"]["steps"] == [f"keep_only domain.{d[0]}", f"add domain.{d[1]}"]
    # keep_only on another cultivar intersects.
    out = stage([{"ref": f"function.{f[0]}", "action": "keep_only"}])
    assert out["prospective_count"] == len((m(d[0]) | m(d[1])) & m(f[0]))
    # drop subtracts.
    out = stage([{"ref": d[1], "action": "drop"}])
    assert out["prospective_count"] == len(((m(d[0]) | m(d[1])) & m(f[0])) - m(d[1]))
    # clear removes every staged step on that ref.
    out = stage([{"ref": d[1], "action": "clear"}, {"ref": d[0], "action": "clear"}])
    assert out["ledger"]["steps"] == [f"keep_only function.{f[0]}"]
    assert out["prospective_count"] == len(m(f[0]))
    # add alone, starting from every member, keeps everything.
    out = stage([{"ref": f[0], "action": "clear"}, {"ref": d[2], "action": "add"}])
    assert out["prospective_count"] == len(everything)

    # Invalid batches fail atomically; old verbs and unknown sources are refused.
    revision = out["ledger"]["revision"]
    bad = stage([{"ref": d[0], "action": "keep_only"}, {"ref": "F999", "action": "keep_only"}])
    assert bad["error"]["code"] == "unknown_ref"
    bad = stage([{"ref": d[0], "action": "include"}])
    assert bad["error"]["code"] == "invalid_decision" and "keep_only" in bad["error"]["message"]
    bad = stage([{"ref": f"color.{d[0]}", "action": "keep_only"}])
    assert bad["error"]["code"] == "unknown_source" and "search" in bad["error"]["message"]
    bad = stage([{"ref": f"function.{d[0]}", "action": "keep_only"}])
    assert bad["error"]["code"] == "unknown_ref"
    bad = stage([{"ref": "search.~", "action": "keep_only"}])
    assert bad["error"]["code"] == "invalid_search"
    after = stage([])
    assert after["ledger"]["revision"] == revision
    assert after["ledger"]["steps"] == [f"add domain.{d[2]}"]


def test_search_steps(manager: SessionManager, catalog: Catalog) -> None:
    overview = manager.dispatch("overview", {})
    session = manager.get(None)
    tool_id = next(iter(catalog.items))
    app = tool_id.split(".")[0].split("_")[0]
    stage = lambda decisions, preview=True: manager.dispatch(  # noqa: E731
        "filter", {"cohort_id": "C1", "decisions": decisions, "preview": preview}
    )
    exact = set(session._search_index().match(parse_term(f'"{app}"')))
    fuzzy = set(session._search_index().match(parse_term(f"~{app}")))
    assert tool_id in exact and exact <= fuzzy

    out = stage([{"ref": f'search."{app}"', "action": "keep_only"}])
    match = out["search_matches"][0]
    assert match["ref"] == f'search."{app}"' and match["count"] == len(exact) == out["prospective_count"]
    assert match["in_cohort"] == len(exact) and "tool_ids" not in match  # counts only
    # Top-level groups holding the in-cohort matches, per cultivar, largest first.
    top = {(c["cultivar"], g["ref"]) for c in overview["cultivars"] for g in c.get("groups", [])}
    for cultivar in {g["cultivar"] for g in match["groups"]}:
        mine = [g for g in match["groups"] if g["cultivar"] == cultivar]
        assert all((cultivar, g["ref"]) in top and g["count"] > 0 for g in mine)
        assert [g["count"] for g in mine] == sorted((g["count"] for g in mine), reverse=True)
        view = session.cohorts["C1"].views[cultivar]
        root = view.view.cut["root"]
        loose = {view.view.tree.nodes[n]["item_id"] for n in root["direct_leaf_node_ids"]}
        assert sum(g["count"] for g in mine) == len(exact - loose)
    none = stage([{"ref": "search.~xyzzyq", "action": "keep_only"}])
    assert none["search_matches"][0]["groups"] == []
    stage([{"ref": "search.~xyzzyq", "action": "clear"}])
    assert tool_id not in session.exposed_tools  # search exposes nothing
    child = stage([], preview=False)
    assert child["committed"] and session.cohorts[child["cohort_id"]].members == frozenset(exact)
    assert child["applied_steps"] == [f'keep_only search."{app}"']

    # add takes matches from the whole catalog, not only the cohort.
    d = [g["ref"] for g in groups(overview, "domain")]
    widened = stage([{"ref": d[0], "action": "keep_only"},
                     {"ref": f"search.~{app}", "action": "add"}], preview=False)
    assert widened["committed"]
    assert session.cohorts[widened["cohort_id"]].members == frozenset(members(manager, "C1", d[0]) | fuzzy)
    refused = manager.dispatch("explore", {"cohort_id": "C1", "refs": [f"search.{app}"]})
    assert refused["error"]["code"] == "invalid_ref"


def test_commit_creates_child_clears_source_and_keeps_source_usable(manager: SessionManager) -> None:
    overview = manager.dispatch("overview", {})
    d = [g["ref"] for g in groups(overview, "domain")]
    wanted = members(manager, "C1", d[0]) | members(manager, "C1", d[1])
    manager.dispatch(
        "filter",
        {"cohort_id": "C1", "decisions": [{"ref": d[0], "action": "keep_only"}], "preview": True},
    )
    # Without preview, one more decision filters on the accumulated ledger.
    child = manager.dispatch(
        "filter", {"cohort_id": "C1", "decisions": [{"ref": d[1], "action": "add"}]}
    )
    assert child["committed"] is True and child["warning"] == COMMIT_WARNING
    assert child["cohort_id"] == "C2" and child["source_cohort_id"] == "C1"
    assert child["tool_count"] == len(wanted) == child["prospective_count"]

    session = manager.get(None)
    assert session.cohorts["C2"].members == frozenset(wanted)
    assert session.cohorts["C1"].steps == [] and session.continuation_cohort_id == "C2"
    assert len(session.cohorts["C1"].members) == overview["tool_count"]

    # Child groups are restricted views with honest retained percentages.
    for block in child["cultivars"]:
        for entry in block.get("groups", []):
            assert 0 < entry["retained_pct"] <= 100
            assert entry["restricted"] == (entry["retained_pct"] < 100)

    # Source aliases still resolve with the source cohort; child aliases are its own.
    source_again = manager.dispatch("explore", {"cohort_id": "C1", "refs": [d[0]]})
    assert source_again["results"][0]["tools"] == len(members(manager, "C1", d[0]))
    resumed = manager.dispatch("overview", {"cohort_id": "C1"})
    assert [g["ref"] for g in groups(resumed, "domain")] == d

    # Committing an empty decision list with nothing staged creates nothing.
    noop = manager.dispatch("filter", {"cohort_id": "C2", "decisions": []})
    assert noop["committed"] is False and len(session.cohorts) == 2


def test_empty_commit_warns_and_clears_ledger(manager: SessionManager) -> None:
    overview = manager.dispatch("overview", {})
    d = [g["ref"] for g in groups(overview, "domain")]
    out = manager.dispatch(
        "filter",
        {
            "cohort_id": "C1",
            "decisions": [
                {"ref": d[0], "action": "keep_only"},
                {"ref": d[1], "action": "add"},
            ]
            + [{"ref": ref, "action": "drop"} for ref in d],
        },
    )
    assert out["committed"] is False and out["prospective_count"] == 0
    assert "zero tools" in out["warning"] and "cleared" in out["warning"]
    assert out["cleared_ledger"]["steps"] == [f"keep_only domain.{d[0]}", f"add domain.{d[1]}"] + [
        f"drop domain.{r}" for r in d]
    assert out["cleared_ledger"]["prospective_count"] == 0
    assert len(manager.get(None).cohorts) == 1
    cohort = manager.get(None).cohorts["C1"]
    assert cohort.steps == []
    # The next filter starts from an empty ledger, not on top of the zero combination.
    again = manager.dispatch("filter", {"cohort_id": "C1", "decisions": [{"ref": d[0], "action": "keep_only"}]})
    assert again["committed"] is True and again["tool_count"] == len(members(manager, "C1", d[0]))


def test_failed_recut_preserves_source_and_ledger(manager: SessionManager, monkeypatch) -> None:
    overview = manager.dispatch("overview", {})
    ref = groups(overview, "domain")[0]["ref"]
    session = manager.get(None)

    def boom(members):
        raise RuntimeError("recut exploded")

    monkeypatch.setattr(session, "_build_views", boom)
    out = manager.dispatch(
        "filter", {"cohort_id": "C1", "decisions": [{"ref": ref, "action": "keep_only"}]}
    )
    assert out["committed"] is False and "Recut failed" in out["warning"]
    assert out["ledger"]["steps"] == [f"keep_only domain.{ref}"]
    assert list(session.cohorts) == ["C1"] and session.continuation_cohort_id == "C1"


def test_collection_before_commit_guessing_and_cross_cohort_hopper(manager: SessionManager, catalog: Catalog) -> None:
    overview = manager.dispatch("overview", {})
    ref = groups(overview, "domain")[0]["ref"]
    session = manager.get(None)
    tool = first_tool(manager, "C1", ref)
    unseen = next(t for t in catalog.items if t not in session.exposed_tools)

    out = manager.dispatch("hopper", {"action": "add", "tool_ids": [tool, unseen, "nope.x"]})
    assert out["added"] == [tool] and out["hopper"] == [tool]
    assert {r["reason"] for r in out["rejected"]} == {"not_exposed", "unknown_tool"}
    assert "input_schema" not in str(out)
    assert len(session.cohorts) == 1  # collected with no commit
    assert len(session.rejected_ids) == 2

    # Schemas only for collected tools, and exposure is recorded.
    got = manager.dispatch("schemas", {"tool_ids": [tool, unseen]})
    assert [s["tool_id"] for s in got["schemas"]] == [tool]
    assert got["schemas"][0]["input_schema"]["type"] == "object"
    assert got["rejected"][0]["reason"] == "not_collected"
    assert session.schema_exposures == {tool: 1}

    # Hopper survives fresh cohorts and commits.
    fresh = manager.dispatch("overview", {})
    child = manager.dispatch(
        "filter",
        {"cohort_id": fresh["cohort_id"], "decisions": [{"ref": ref, "action": "keep_only"}]},
    )
    assert child["committed"] is True
    assert manager.dispatch("hopper", {"action": "list"})["hopper"] == [tool]

    # Idempotent retry by request_id.
    second = first_tool(manager, child["cohort_id"], groups(child, "function")[0]["ref"]) if any(
        b.get("groups") for b in child["cultivars"]
    ) else child["tools"][-1]["tool_id"]
    a = manager.dispatch("hopper", {"action": "add", "tool_ids": [second], "request_id": "r1"})
    b = manager.dispatch("hopper", {"action": "add", "tool_ids": [second], "request_id": "r1"})
    assert b.get("replayed") and b["hopper_revision"] == a["hopper_revision"]

    removed = manager.dispatch("hopper", {"action": "remove", "tool_ids": [tool]})
    assert tool in removed["removed"] and tool not in removed["hopper"]
    assert manager.dispatch("schemas", {"tool_ids": [tool]})["schemas"] == []
    assert manager.dispatch("hopper", {"action": "clear"})["hopper"] == []
    # Exposure records are cumulative: removal erases nothing.
    assert session.schema_exposures == {tool: 1} and tool in session.exposed_tools


def test_small_cohort_shows_individuals(manager: SessionManager) -> None:
    overview = manager.dispatch("overview", {})
    session = manager.get(None)
    cohort = session.cohorts["C1"]
    # Find any alias holding at most eight tools.
    target = None
    for view in cohort.views.values():
        for alias, node_id in view.node_by_alias.items():
            if 2 <= len(view.members_of(node_id)) <= 8:
                target = alias
                break
        if target:
            break
    assert target is not None, overview
    shown = manager.dispatch("explore", {"cohort_id": "C1", "refs": [target]})["results"][0]
    assert "groups" not in shown and len(shown["tool_cards"]) == shown["tools"]

    child = manager.dispatch(
        "filter", {"cohort_id": "C1", "decisions": [{"ref": target, "action": "keep_only"}]}
    )
    assert child["committed"] and len(child["tools"]) == child["tool_count"] <= 8
    assert all(b["cut_status"] == "individuals" for b in child["cultivars"])


def test_sessions_are_isolated_and_close_tears_down(catalog: Catalog) -> None:
    manager = SessionManager(catalog)
    a = manager.dispatch("session", {"action": "open", "task_id": "a"})["session_id"]
    b = manager.dispatch("session", {"action": "open", "task_id": "b"})["session_id"]
    assert manager.dispatch("overview", {})["error"]["code"] == "session_required"

    over_a = manager.dispatch("overview", {"session_id": a})
    tool = first_tool_in(manager, a, over_a)
    manager.dispatch("hopper", {"session_id": a, "action": "add", "tool_ids": [tool]})
    # Session b never saw that tool and has no cohort C1.
    assert manager.dispatch("hopper", {"session_id": b, "action": "add", "tool_ids": [tool]})["added"] == []
    assert manager.dispatch("overview", {"session_id": b, "cohort_id": "C1"})["error"]["code"] == "unknown_cohort"

    closed = manager.dispatch("session", {"action": "close", "session_id": a})
    assert closed["open"] is False
    assert manager.dispatch("overview", {"session_id": a})["error"]["code"] == "session_closed"
    assert manager.sessions[a].hopper == {}
    assert manager.dispatch("session", {"action": "status", "session_id": b})["open"] is True
    assert all(event["session_id"] == b for event in manager.sessions[b].events)


def first_tool_in(manager: SessionManager, session_id: str, overview: dict) -> str:
    ref = overview["cultivars"][0]["groups"][0]["ref"]
    while True:
        result = manager.dispatch(
            "explore", {"session_id": session_id, "cohort_id": overview["cohort_id"], "refs": [ref]}
        )["results"][0]
        if result.get("tool_cards"):
            return result["tool_cards"][0]["tool_id"]
        ref = result["groups"][0]["ref"]



def test_trace_events_carry_cohort_sizes(manager: SessionManager) -> None:
    """Overview and committed filters record tool_count; filters record prospective_count."""
    overview = manager.dispatch("overview", {})
    ref = groups(overview, "domain")[0]["ref"]
    manager.dispatch("filter", {"cohort_id": "C1", "preview": True,
                                "decisions": [{"action": "keep_only", "ref": ref}]})
    committed = manager.dispatch("filter", {"cohort_id": "C1"})
    events = manager.get(None).events
    assert events[1]["command"] == "overview" and events[1]["tool_count"] == overview["tool_count"]
    preview, commit = events[2], events[3]
    assert preview["tool_count"] is None and preview["prospective_count"] == committed["tool_count"]
    assert commit["tool_count"] == committed["tool_count"] and commit["cohort_id"] == committed["cohort_id"]
