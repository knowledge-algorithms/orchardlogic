"""Lexical search terms: parsing, exact words and phrases, fuzzy text."""

from __future__ import annotations

import pytest

from orchardlogic.cultivate.render import render_text
from orchardlogic.cultivate.search import SearchSyntaxError, build_index, parse_term, stem

INDEX = build_index([
    ("amazon.show_cart", "Show the items in your Amazon cart."),
    ("amazon.add_to_cart", "Add a product to the cart."),
    ("venmo.send_money", "Send money to a friend."),
    ("venmo.show_transactions", "List payments sent and received."),
    ("spotify.like_song", "Like a song."),
])


def m(term: str) -> list[str]:
    return INDEX.match(parse_term(term))


def test_parse_forms_and_canonical_text() -> None:
    assert parse_term('"amazon"').canonical == 'search."amazon"'
    assert parse_term("amazon") == parse_term('"amazon"')
    assert parse_term("~amazon").canonical == "search.~amazon"
    assert parse_term('~"Send Money"').canonical == 'search.~"send money"'
    for bad in ("", "~", '""', "~ '' "):
        with pytest.raises(SearchSyntaxError):
            parse_term(bad)


def test_exact_matches_whole_words_and_adjacent_phrases() -> None:
    assert m('"amazon"') == ["amazon.show_cart", "amazon.add_to_cart"]  # ID or description
    assert m('"cart"') == ["amazon.show_cart", "amazon.add_to_cart"]
    assert m('"send money"') == ["venmo.send_money"]
    assert m('"money send"') == []  # order matters
    assert m('"amaz"') == []  # no partial words
    assert m('"payment"') == []  # no stemming


def test_fuzzy_substring_stem_and_typo() -> None:
    assert m("~amaz") == ["amazon.show_cart", "amazon.add_to_cart"]  # substring
    assert m("~payment") == ["venmo.show_transactions"]  # stem: payments
    assert m("~transaction") == ["venmo.show_transactions"]
    assert m("~spotfy") == ["spotify.like_song"]  # one edit
    assert m('~"money friend"') == ["venmo.send_money"]  # every word, any order
    assert m("~xyz") == []
    assert stem("payments") == "payment" and stem("liked") == "lik" and stem("class") == "class"


def test_render_lists_search_matches() -> None:
    text = render_text("filter", {
        "cohort_id": "C1", "prospective_count": 2, "committed": False,
        "search_matches": [{"ref": 'search."amazon"', "count": 32, "in_cohort": 2}],
        "ledger": {"revision": 1, "steps": ['keep_only search."amazon"'], "prospective_count": 2},
    })
    assert 'search."amazon": 32 matches (2 in cohort C1)' in text
    assert "amazon.show_cart" not in text
    assert 'Staged: keep_only search."amazon" -> 2 tools' in text


def test_render_search_match_groups() -> None:
    groups = [
        {"cultivar": "a", "ref": "D2", "label": "Spotify", "count": 3},
        {"cultivar": "b", "ref": "F1", "label": "Read", "count": 2},
        {"cultivar": "b", "ref": "F3", "label": "Update", "count": 1},
    ]
    text = render_text("filter", {
        "cohort_id": "C2", "prospective_count": 3, "committed": False,
        "search_matches": [
            {"ref": "search.~library", "count": 12, "in_cohort": 3, "groups": groups},
        ],
    })
    assert ("search.~library: 12 matches (3 in cohort C2) · D2 Spotify (3) · "
            "F1 Read (2), F3 Update (1)") in text
