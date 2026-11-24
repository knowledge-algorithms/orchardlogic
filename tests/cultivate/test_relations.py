"""Tool relations (orchard_relations_v1): needs./feeds. refs, neighborhood, hopper check."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from orchardlogic.cultivate.catalog import build_fixture_catalog, load_relations
from orchardlogic.cultivate.navigation import SessionManager
from orchardlogic.cultivate.render import render_text

LOGIN, SEARCH, BUY, NOTE, PAY = "mail.search_00", "shop.search_05", "pay.update_02", "notes.show_04", "music.create_01"


def relations_file(tmp_path: Path) -> Path:
    tools = {
        BUY: {
            "needs": [
                {"param": "token", "source": "tools", "producer_count": 1,
                 "producers": [{"tool_id": LOGIN, "via": "token"}]},
                {"param": "item_id", "source": "tools", "producer_count": 2,
                 "producers": [{"tool_id": SEARCH, "via": "item_id"}, {"tool_id": "gone.tool", "via": "x"}]},
                {"param": "city", "source": "tools", "producer_count": 1,
                 "producers": [{"tool_id": SEARCH, "via": "city"}]},
                {"param": "note", "source": "task", "producer_count": 40},
            ],
            "feeds": [{"tool_id": NOTE, "via": ["order_id->order_id"]}],
            "feeds_count": 1,
        },
        LOGIN: {"needs": [], "feeds": [{"tool_id": t, "via": ["token->token"]}
                                       for t in (BUY, NOTE, PAY, SEARCH, "files.delete_03", "music.create_01")]},
        "gone.tool": {"needs": [], "feeds": []},
    }
    path = tmp_path / "relations.json"
    path.write_text(json.dumps({"schema_version": "orchard_relations_v1", "tools": tools}), encoding="utf-8")
    return path


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    base = build_fixture_catalog()
    relations = load_relations(relations_file(tmp_path), base.items)
    catalog = dataclasses.replace(base, relations=relations, catalog_id="")
    m = SessionManager(catalog, available_tools=[PAY])
    m.dispatch("session", {"action": "open"})
    return m


def test_load_relations_drops_unknown_tools_and_validates(tmp_path: Path) -> None:
    items = build_fixture_catalog().items
    rel = load_relations(relations_file(tmp_path), items)
    assert "gone.tool" not in rel
    assert [p["tool_id"] for p in rel[BUY]["needs"][1]["producers"]] == [SEARCH]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "x", "tools": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_relations(bad, items)


def test_catalog_id_changes_with_relations(manager: SessionManager) -> None:
    assert manager.catalog.catalog_id != build_fixture_catalog().catalog_id


def test_neighborhood_groups_inputs_and_exposes_listed_tools(manager: SessionManager) -> None:
    r = manager.dispatch("neighborhood", {"tool_id": BUY})
    needs = {tuple(n["params"]): n for n in r["needs"]}
    assert ("item_id", "city") in needs  # same producers -> one line
    assert needs[("note",)]["source"] == "task"
    assert [p["tool_id"] for p in needs[("token",)]["producers"]] == [LOGIN]
    session = manager.get(None)
    assert LOGIN in session.exposed_tools and NOTE in session.exposed_tools
    assert BUY in session.exposed_tools  # the inspected tool itself can be collected
    added = manager.dispatch("hopper", {"action": "add", "tool_ids": [LOGIN]})
    assert added["added"] == [LOGIN]
    text = render_text("neighborhood", r)
    assert "item_id, city from any of:" in text and "note: from the task or you" in text


def test_shared_feeds_are_summarized(manager: SessionManager) -> None:
    r = manager.dispatch("neighborhood", {"tool_id": LOGIN})
    assert r["feeds"] == [] and r["feeds_shared"][0]["count"] == 6
    assert "token->token fills 6 tools" in render_text("neighborhood", r)


def test_needs_and_feeds_filter_refs(manager: SessionManager) -> None:
    cohort = manager.dispatch("overview", {})["cohort_id"]
    r = manager.dispatch("filter", {"cohort_id": cohort, "decisions": [{"action": "keep_only", "ref": f"needs.{BUY}"}]})
    assert r["committed"] and r["tool_count"] == 2
    assert r["applied_steps"] == [f"keep_only needs.{BUY}"]
    r = manager.dispatch("filter", {"cohort_id": cohort, "preview": True,
                                    "decisions": [{"action": "keep_only", "ref": f"feeds.{BUY}"}]})
    assert r["prospective_count"] == 1
    err = manager.dispatch("filter", {"cohort_id": cohort, "decisions": [{"action": "add", "ref": "needs.no.such"}]})
    assert err["error"]["code"] == "unknown_tool"


def test_hopper_check_counts_available_and_groups_gaps(manager: SessionManager) -> None:
    manager.dispatch("neighborhood", {"tool_id": BUY})  # shows BUY and its producers
    assert manager.dispatch("hopper", {"action": "add", "tool_ids": [BUY]})["added"] == [BUY]
    check = manager.dispatch("hopper", {"action": "check"})["check"]
    assert {g["param"] for g in check["gaps"]} == {"token", "item_id", "city"}
    assert check["from_task"] == {BUY: ["note"]}
    with_have = manager.dispatch("hopper", {"action": "check", "tool_ids": [SEARCH]})["check"]
    assert {g["param"] for g in with_have["gaps"]} == {"token"} and with_have["satisfied_inputs"] == 2
    manager.dispatch("hopper", {"action": "add", "tool_ids": [LOGIN, SEARCH]})
    done = manager.dispatch("hopper", {"action": "check"})["check"]
    assert done["gaps"] == [] and "0 missing" in render_text("hopper", {"hopper": [], "check": done})








def test_task_supplied_input_with_ranked_producers(tmp_path: Path) -> None:
    base = build_fixture_catalog()
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"schema_version": "orchard_relations_v1", "tools": {BUY: {"needs": [
        {"param": "email", "source": "task", "producer_count": 40, "producers_ranked": True,
         "producers": [{"tool_id": SEARCH, "via": "email"}, {"tool_id": LOGIN, "via": "email"}]}],
        "feeds": []}}}), encoding="utf-8")
    catalog = dataclasses.replace(base, relations=load_relations(path, base.items), catalog_id="")
    m = SessionManager(catalog)
    m.dispatch("session", {"action": "open"})
    r = m.dispatch("neighborhood", {"tool_id": BUY})
    need = r["needs"][0]
    assert need["source"] == "task" and [p["tool_id"] for p in need["producers"]] == [SEARCH, LOGIN]
    assert need["more"] == 38
    assert "email: from the task or you; otherwise any of:" in render_text("neighborhood", r)
    m.dispatch("hopper", {"action": "add", "tool_ids": [BUY]})
    assert m.dispatch("hopper", {"action": "check"})["check"]["gaps"] == []  # task inputs are not gaps
