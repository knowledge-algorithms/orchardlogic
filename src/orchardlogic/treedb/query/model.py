"""Structured query: dataclasses and JSON (de)serialization (phase P5B, D-025).

A query is a list of rules and a schema version. The base cohort is **not** part
of the query: it is the first argument of ``execute``. A rule is one of::

    {"cultivar": "A", "node": "<branch uuid>", "op": "include" | "exclude"}   # node rule
    {"item": "i1", "op": "include" | "exclude"}                               # item rule

Accepted at G3 (D-025): conflicts are rejected, there is no ``replace`` flag, no
``neutral`` operation and no Boolean grouping. Each of those has its own error
code (``REPLACE_UNSUPPORTED``, ``NEUTRAL_UNSUPPORTED``, ``GROUPING_UNSUPPORTED``).

Errors are collected over **all** rules and the one with the lowest ``CODES``
position is raised, so the code never depends on the order of the rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "treedb.query/1"
OPS = ("include", "exclude")

#: Every error code, in the order in which a query with several faults reports them.
CODES = (
    "GROUPING_UNSUPPORTED", "UNSUPPORTED_VERSION", "INVALID_QUERY", "INVALID_RULE",
    "REPLACE_UNSUPPORTED", "NEUTRAL_UNSUPPORTED",
    "UNKNOWN_CULTIVAR", "UNKNOWN_NODE", "UNKNOWN_ITEM",
    "CONFLICT_NODE", "CONFLICT_ITEM",
)

_GROUPING_KEYS = ("any_of", "all_of", "not", "or", "and", "group", "groups")


class QueryError(ValueError):
    """A query is rejected. ``code`` is one of ``CODES``; ``details`` names the offenders."""

    def __init__(self, code: str, message: str, **details: Any):
        assert code in CODES, code
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details


def raise_first(errors: Iterable[QueryError]) -> None:
    """Raise the error with the lowest code position (ties: the smaller message)."""
    errs = list(errors)
    if errs:
        raise min(errs, key=lambda e: (CODES.index(e.code), str(e)))


@dataclass(frozen=True)
class NodeRule:
    cultivar: str
    node: str          # branch id (UUID text) in that cultivar
    op: str

    def to_json(self) -> dict[str, str]:
        return {"cultivar": self.cultivar, "node": self.node, "op": self.op}


@dataclass(frozen=True)
class ItemRule:
    item: str
    op: str

    def to_json(self) -> dict[str, str]:
        return {"item": self.item, "op": self.op}


Rule = NodeRule | ItemRule


@dataclass(frozen=True)
class Query:
    """Rules in any order, duplicates and conflicts allowed (``normalize`` decides)."""

    rules: tuple[Rule, ...] = ()
    version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {"version": self.version, "rules": [r.to_json() for r in self.rules]}

    def dumps(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, obj: Mapping[str, Any] | str) -> "Query":
        """Parse and check the shape. Needs no database. Raises ``QueryError``."""
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except ValueError as exc:
                raise QueryError("INVALID_QUERY", f"not valid JSON: {exc}") from None
        if not isinstance(obj, Mapping):
            raise QueryError("INVALID_QUERY", "a query is a JSON object")
        errors: list[QueryError] = []
        for key in obj:
            if key in _GROUPING_KEYS:
                errors.append(QueryError("GROUPING_UNSUPPORTED",
                                         f"explicit Boolean grouping ({key!r}) is not supported (D-025)"))
            elif key not in ("version", "rules"):
                errors.append(QueryError("INVALID_QUERY", f"unknown field {key!r}"
                                         + ("; the base cohort is an argument of execute" if key == "base" else "")))
        version = obj.get("version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            errors.append(QueryError("UNSUPPORTED_VERSION", f"version {version!r}; expected {SCHEMA_VERSION!r}"))
        raw = obj.get("rules", [])
        rules: list[Rule] = []
        if not isinstance(raw, list):
            errors.append(QueryError("INVALID_QUERY", "'rules' must be a list"))
            raw = []
        for r in raw:
            try:
                rules.append(_parse_rule(r))
            except QueryError as exc:
                errors.append(exc)
        raise_first(errors)
        return cls(tuple(rules), SCHEMA_VERSION)


def _parse_rule(r: Any) -> Rule:
    if not isinstance(r, Mapping):
        raise QueryError("INVALID_RULE", f"a rule is a JSON object, got {r!r}")
    if "replace" in r:
        raise QueryError("REPLACE_UNSUPPORTED", "there is no replace flag; an adapter resolves its own ledger (D-025)")
    op = r.get("op")
    if op == "neutral":
        raise QueryError("NEUTRAL_UNSUPPORTED", "neutral is not a query operation (D-025)")
    if op not in OPS:
        raise QueryError("INVALID_RULE", f"op must be one of {OPS}, got {op!r}")
    keys = set(r) - {"op"}
    if keys == {"item"}:
        if not isinstance(r["item"], str) or not r["item"]:
            raise QueryError("INVALID_RULE", f"item must be a non-empty string, got {r['item']!r}")
        return ItemRule(r["item"], op)
    if keys == {"cultivar", "node"}:
        if not (isinstance(r["cultivar"], str) and isinstance(r["node"], str) and r["cultivar"] and r["node"]):
            raise QueryError("INVALID_RULE", f"cultivar and node must be non-empty strings in {dict(r)!r}")
        return NodeRule(r["cultivar"], r["node"], op)
    raise QueryError("INVALID_RULE", f"a rule has either 'item' or both 'cultivar' and 'node', got keys {sorted(r)}")
