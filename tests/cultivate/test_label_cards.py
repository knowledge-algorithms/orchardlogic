"""Card schema and versioned labeling prompts. No model client."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import orchardlogic.cultivate as orchard
import orchardlogic.cultivate.labeling.cards as cards_mod
import orchardlogic.cultivate.labeling.cost as cost_mod
import orchardlogic.cultivate.labeling.prompts as prompts_mod
from orchardlogic.cultivate.exceptions import LabelingError, OrchardError
from orchardlogic.cultivate.labeling.cards import (
    CARD_SCHEMA_VERSION,
    NodeCard,
    parse_card,
    parse_model_card,
    serialize_card,
)
from orchardlogic.cultivate.labeling.prompts import (
    BOTTOM_UP_LEAF_PROMPT_VERSION,
    BOTTOM_UP_LEAF_SYSTEM,
    BOTTOM_UP_PARENT_PROMPT_VERSION,
    BOTTOM_UP_PARENT_SYSTEM,
    CONTRAST_PROMPT_VERSION,
    CONTRAST_SYSTEM,
    render_contrast_prompt,
    render_leaf_prompt,
    render_parent_prompt,
)

FIXTURE = Path(__file__).parent / "fixtures" / "labeling_16.json"
_BANNED = (
    "unified",
    "consolidated",
    "combined",
    "integrated",
    "comprehensive",
    "various",
)


def _payload(**overrides: object) -> dict:
    payload = {
        "schema_version": CARD_SCHEMA_VERSION,
        "title": "Mail routing",
        "summary": "Routes inbound letters to the right desk.",
        "scope": ["inbound letters"],
        "capabilities": ["route to a desk"],
        "concepts": ["mail"],
        "uncertainty": [],
    }
    payload.update(overrides)
    return payload


def _card() -> NodeCard:
    return parse_card(_payload())


def test_valid_card_parses_and_round_trips() -> None:
    card = _card()
    assert card.schema_version == "orchard_node_card_v1"
    assert card.title == "Mail routing"
    assert card.style_warnings == ()
    assert "retry" not in card.to_dict()
    assert parse_card(card.to_dict()) == card
    assert serialize_card(card) == serialize_card(parse_card(card.to_dict()))


def test_title_and_summary_are_stripped() -> None:
    card = parse_card(_payload(title="  Mail routing  ", summary="  Routes letters.  "))
    assert card.title == "Mail routing"
    assert card.summary == "Routes letters."


def test_empty_title_and_summary_are_hard_failures() -> None:
    with pytest.raises(LabelingError, match="title is empty"):
        parse_card(_payload(title="   "))
    with pytest.raises(LabelingError, match="summary is empty"):
        parse_card(_payload(summary=""))
    with pytest.raises(LabelingError, match="title is empty"):
        NodeCard(
            title="",
            summary="Routes letters.",
            scope=("letters",),
            capabilities=("route",),
            concepts=("mail",),
            uncertainty=(),
        )


def test_schema_mismatch_is_a_hard_failure() -> None:
    with pytest.raises(LabelingError, match="schema mismatch"):
        parse_card(_payload(schema_version="orchard_node_card_v2"))
    with pytest.raises(LabelingError, match="schema mismatch"):
        parse_card(["not", "an", "object"])
    missing = _payload()
    del missing["uncertainty"]
    with pytest.raises(LabelingError, match="schema mismatch"):
        parse_card(missing)
    with pytest.raises(LabelingError, match="schema mismatch"):
        parse_card(_payload(title=12))
    with pytest.raises(LabelingError, match="schema mismatch"):
        parse_card(_payload(scope="inbound letters"))
    with pytest.raises(LabelingError, match="schema mismatch"):
        parse_card(_payload(concepts=["mail", ""]))
    with pytest.raises(LabelingError, match="schema mismatch"):
        parse_card(_payload(capabilities=["route", 3]))


def test_model_json_is_title_and_summary_and_parse_card_stays_strict() -> None:
    card = parse_model_card({"title": "Mail routing", "summary": "Routes letters."})
    assert card.schema_version == CARD_SCHEMA_VERSION
    assert card.scope == ()
    assert card.capabilities == ()
    assert card.concepts == ()
    assert card.uncertainty == ()
    kept = parse_model_card(
        {
            "title": "Mail routing",
            "summary": "Routes letters.",
            "scope": ["letters"],
            "capabilities": ["route"],
            "schema_version": "ignored",
        }
    )
    assert kept.scope == ("letters",)
    assert kept.capabilities == ("route",)
    assert kept.concepts == ()
    assert kept.uncertainty == ()
    assert kept.schema_version == CARD_SCHEMA_VERSION
    with pytest.raises(LabelingError, match="schema mismatch"):
        parse_card({"title": "Mail routing", "summary": "Routes letters."})


def test_style_warnings_are_recorded_and_do_not_fail() -> None:
    card = parse_card(
        _payload(
            title="The unified loaf",
            summary="A comprehensive note about bread.",
        )
    )
    assert card.title == "The unified loaf"
    assert card.style_warnings == (
        "title contains article 'the'",
        "title contains banned word 'unified'",
        "summary contains banned word 'comprehensive'",
    )
    assert "retry" not in card.to_dict()
    tampered = card.to_dict()
    tampered["style_warnings"] = []
    assert parse_card(tampered).style_warnings == card.style_warnings


def test_articles_in_the_summary_are_not_warnings() -> None:
    card = parse_card(
        _payload(
            title="Mail routing",
            summary="The desk sorts a letter.",
        )
    )
    assert card.style_warnings == ()


def test_banned_word_stems_and_summary_articles_stay_quiet() -> None:
    card = parse_card(_payload(title="Combination lock", summary="The lock opens."))
    assert card.style_warnings == ()


@pytest.mark.parametrize("word", _BANNED)
def test_each_banned_word_warns_without_raising(word: str) -> None:
    card = parse_card(_payload(title=f"Desk {word}"))
    assert any(word in warning for warning in card.style_warnings)


def test_labeling_error_is_an_orchard_error() -> None:
    assert issubclass(LabelingError, OrchardError)
    assert issubclass(LabelingError, ValueError)


def test_prompt_versions_and_contract_phrases() -> None:
    versions = {
        BOTTOM_UP_LEAF_PROMPT_VERSION,
        BOTTOM_UP_PARENT_PROMPT_VERSION,
        CONTRAST_PROMPT_VERSION,
    }
    assert versions == {
        "bottom_up_leaf_v2",
        "bottom_up_parent_v2",
        "contrast_v2",
    }
    assert BOTTOM_UP_LEAF_PROMPT_VERSION in BOTTOM_UP_LEAF_SYSTEM
    assert BOTTOM_UP_PARENT_PROMPT_VERSION in BOTTOM_UP_PARENT_SYSTEM
    assert CONTRAST_PROMPT_VERSION in CONTRAST_SYSTEM

    leaf = render_leaf_prompt("Feed the starter until it doubles.")
    assert "Write one card from the document in the user message." in leaf.system
    assert "No articles." in leaf.system
    assert "A word stays only if removing it would change the meaning." in leaf.system
    assert "one or two plain sentences" in leaf.system
    assert "Feed the starter until it doubles." in leaf.user
    assert "title and summary" in leaf.system
    assert "descendant" not in leaf.system.lower()
    for word in _BANNED:
        assert word in leaf.system
    for field in ("scope", "capabilities", "concepts", "uncertainty"):
        assert field not in leaf.system

    parent = render_parent_prompt(_card(), _card())
    assert "two child cards in the user message only" in parent.system
    assert "Cover both children." in parent.system
    assert "more abstract than the children" in parent.system
    assert serialize_card(_card()) in parent.user
    assert parent.user.count(serialize_card(_card())) == 2

    neighbor = parse_card(
        _payload(title="Brake pads", summary="Align pads to the rim.")
    )
    contrast = render_contrast_prompt(_card(), [neighbor])
    assert (
        "Rewrite the title and the summary so the card is less like the neighbor"
        in contrast.system
    )
    assert "A word stays when it increases that distance." in contrast.system
    assert serialize_card(_card()) in contrast.user
    assert serialize_card(neighbor) in contrast.user
    for field in ("scope", "capabilities", "concepts", "uncertainty"):
        assert field not in parent.system
        assert field not in contrast.system


def test_labeling_modules_do_not_reference_a_client() -> None:
    text = "\n".join(
        inspect.getsource(module)
        for module in (cards_mod, prompts_mod, cost_mod)
    )
    lowered = text.lower()
    for token in ("openai", "urllib", "requests", "httpx", "socket"):
        assert token not in lowered
    assert "def label_bottom_up" not in text
    assert "def label_contrast" not in text
    assert hasattr(orchard, "label_bottom_up")
    assert "label_bottom_up" in orchard.__all__
    assert hasattr(orchard, "label_contrast")
    assert "label_contrast" in orchard.__all__
    assert "estimate_labeling" in orchard.__all__


def test_labeling_fixture_has_sixteen_invented_documents() -> None:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(rows) == 16
    groups: dict[str, int] = {}
    item_ids = [row["item_id"] for row in rows]
    assert len(set(item_ids)) == 16
    blob = FIXTURE.read_text(encoding="utf-8").lower()
    assert "appworld" not in blob
    for row in rows:
        assert row["source"] == "invented"
        assert row["text"].strip()
        assert row["title"].strip()
        groups[row["group"]] = groups.get(row["group"], 0) + 1
    assert groups == {
        "bread": 4,
        "bicycle": 4,
        "library": 4,
        "weather": 4,
    }
