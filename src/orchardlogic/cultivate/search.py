"""Lexical search over catalog items, used as a filter source (``search.<term>``).

Term syntax (the part after ``search.``):

* ``"amazon"`` or ``amazon`` - exact: every word appears as a whole word; a quoted
  phrase of several words must appear in that order, adjacent.
* ``~amazon`` or ``~"send money"`` - fuzzy: every word matches some word of the tool
  by substring (words of 3+ letters), by a simple shared stem, or within a fixed edit
  distance (1 for words of 5+ letters, 2 for 8+).

Each tool is matched against its full tool ID (split on ``.`` and ``_``) and its
description. Matching is deterministic and needs no model or network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

_WORD = re.compile(r"[a-z0-9]+")


class SearchSyntaxError(ValueError):
    """A search term that cannot be parsed; the message states the syntax."""


@dataclass(frozen=True)
class SearchTerm:
    words: tuple[str, ...]
    fuzzy: bool

    @property
    def canonical(self) -> str:
        text = " ".join(self.words)
        quoted = f'"{text}"'
        if self.fuzzy:
            return f"search.~{text if len(self.words) == 1 else quoted}"
        return f"search.{quoted}"


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def parse_term(value: str) -> SearchTerm:
    """Parse the text after ``search.``."""
    raw = value.strip()
    fuzzy = raw.startswith("~")
    if fuzzy:
        raw = raw[1:].strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1]
    found = tuple(words(raw))
    if not found:
        raise SearchSyntaxError(
            f"search term {value!r} has no words. Use search.\"word\" (exact) or "
            "search.~word (fuzzy)."
        )
    return SearchTerm(found, fuzzy)


def stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    if len(word) > 3 and word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _within(a: str, b: str, limit: int) -> bool:
    """Levenshtein distance between a and b is at most ``limit``."""
    if abs(len(a) - len(b)) > limit:
        return False
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char_a != char_b)))
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


def _fuzzy_word(query: str, word: str) -> bool:
    if query == word or (len(query) >= 3 and query in word):
        return True
    if stem(query) == stem(word):
        return True
    limit = 2 if len(query) >= 8 else 1 if len(query) >= 5 else 0
    return limit > 0 and _within(query, word, limit)


class SearchIndex:
    """Word lists per item: tool ID words and description words kept apart, so an
    exact phrase never spans the ID and the description."""

    def __init__(self, texts: Mapping[str, tuple[str, str]]) -> None:
        self.order = list(texts)
        self.fields = {item: (words(tid), words(desc)) for item, (tid, desc) in texts.items()}

    def match(self, term: SearchTerm) -> list[str]:
        return [item for item in self.order if self._matches(term, self.fields[item])]

    @staticmethod
    def _matches(term: SearchTerm, fields: tuple[list[str], list[str]]) -> bool:
        if term.fuzzy:
            vocabulary = set(fields[0]) | set(fields[1])
            return all(any(_fuzzy_word(q, w) for w in vocabulary) for q in term.words)
        n = len(term.words)
        return any(
            tuple(field[i:i + n]) == term.words
            for field in fields for i in range(len(field) - n + 1)
        )


def build_index(items: Iterable[tuple[str, str]]) -> SearchIndex:
    """``items``: (tool_id, description) pairs in catalog order."""
    return SearchIndex({tool_id: (tool_id, description) for tool_id, description in items})
