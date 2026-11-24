"""Order-independent normal form of a query (phase P5B, D-025).

``normalize(con, query)``:

* checks every rule against the store: the cultivar exists, the node is a branch
  of that cultivar, the item is in the item registry (``UNKNOWN_CULTIVAR``,
  ``UNKNOWN_NODE``, ``UNKNOWN_ITEM``);
* rejects ``include X`` together with ``exclude X`` on a node or on an item
  (``CONFLICT_NODE``, ``CONFLICT_ITEM``): no policy picks a winner (D-025);
* removes identical duplicates and sorts: node rules by (cultivar, node, op),
  then item rules by (item, op). Node ids are written as lower-case UUID text.

The result depends only on the *set* of rules, so permuting the input gives the
same ``NormalQuery`` and normalizing a normal form returns it unchanged. It
does not depend on the base cohort: out-of-base item rules are dealt with by
``execute`` (reported as ``ignored_rules``). Nothing here reads a clock.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import duckdb

from .model import (SCHEMA_VERSION, ItemRule, NodeRule, Query, QueryError, raise_first)


@dataclass(frozen=True)
class NormalQuery:
    """Sorted, duplicate-free, conflict-free rules."""

    node_rules: tuple[NodeRule, ...] = ()
    item_rules: tuple[ItemRule, ...] = ()
    version: str = SCHEMA_VERSION

    @property
    def rules(self) -> tuple[NodeRule | ItemRule, ...]:
        return (*self.node_rules, *self.item_rules)

    @property
    def cultivars(self) -> tuple[str, ...]:
        return tuple(sorted({r.cultivar for r in self.node_rules}))

    def to_json(self) -> dict[str, Any]:
        return {"version": self.version, "rules": [r.to_json() for r in self.rules]}

    def as_query(self) -> Query:
        return Query(self.rules, self.version)


def _check_store(con: duckdb.DuckDBPyConnection, node_rules: set[NodeRule], item_rules: set[ItemRule]) -> list[QueryError]:
    errors: list[QueryError] = []
    trees = dict(con.execute("SELECT name, id FROM treedb.tree_registry").fetchall())
    for name in sorted({r.cultivar for r in node_rules} - set(trees)):
        errors.append(QueryError("UNKNOWN_CULTIVAR", f"no cultivar named {name!r}", cultivar=name))
    for r in sorted(node_rules, key=lambda r: (r.cultivar, r.node, r.op)):
        if r.cultivar not in trees:
            continue
        found = False
        try:
            found = con.execute("SELECT 1 FROM treedb.tree_branches WHERE tree_id = ? AND branch_id = ?",
                                [trees[r.cultivar], uuid.UUID(r.node)]).fetchone() is not None
        except ValueError:
            pass
        if not found:
            errors.append(QueryError("UNKNOWN_NODE", f"{r.node!r} is not a branch of cultivar {r.cultivar!r}",
                                     cultivar=r.cultivar, node=r.node))
    ids = sorted({r.item for r in item_rules})
    if ids:
        known = {x[0] for x in con.execute(
            "SELECT item_id FROM treedb.item_registry WHERE list_contains(CAST(? AS VARCHAR[]), item_id)", [ids]).fetchall()}
        unknown = [i for i in ids if i not in known]
        if unknown:
            errors.append(QueryError("UNKNOWN_ITEM", f"unknown item id(s): {unknown[:20]}", items=unknown))
    return errors


def normalize(con: duckdb.DuckDBPyConnection, query: Query | NormalQuery) -> NormalQuery:
    """The normal form of ``query``, or ``QueryError`` (see the module docstring)."""
    if query.version != SCHEMA_VERSION:
        raise QueryError("UNSUPPORTED_VERSION", f"version {query.version!r}; expected {SCHEMA_VERSION!r}")
    node_rules: set[NodeRule] = set()
    item_rules: set[ItemRule] = set()
    errors: list[QueryError] = []
    for r in query.rules:
        if isinstance(r, NodeRule):
            try:
                node = str(uuid.UUID(r.node))
            except ValueError:
                node = r.node  # reported as UNKNOWN_NODE by the store check
            node_rules.add(NodeRule(r.cultivar, node, r.op))
        else:
            item_rules.add(ItemRule(r.item, r.op))
    errors += _check_store(con, node_rules, item_rules)
    for key in sorted({(r.cultivar, r.node) for r in node_rules if r.op == "include"}
                      & {(r.cultivar, r.node) for r in node_rules if r.op == "exclude"}):
        errors.append(QueryError("CONFLICT_NODE", f"node {key[1]} of cultivar {key[0]!r} is both included and excluded",
                                 cultivar=key[0], node=key[1]))
    for item in sorted({r.item for r in item_rules if r.op == "include"} & {r.item for r in item_rules if r.op == "exclude"}):
        errors.append(QueryError("CONFLICT_ITEM", f"item {item!r} is both included and excluded", item=item))
    raise_first(errors)
    return NormalQuery(
        tuple(sorted(node_rules, key=lambda r: (r.cultivar, r.node, r.op))),
        tuple(sorted(item_rules, key=lambda r: (r.item, r.op))),
    )
