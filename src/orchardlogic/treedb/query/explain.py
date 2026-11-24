"""Explain one item's verdict (phase P5B, D-025).

``explain(con, result_or_query, item_id)`` walks the item's ancestor chain in
each constrained cultivar with plain Python lookups (it does not run the
compiled SQL), so agreement with ``execute`` is a real cross-check. For each
cultivar it reports the applicable rules nearest first, the winner and its
depth, the item override, whether the cultivar is selecting, and the verdict.
``depth`` is the distance from the top of the chain (the cultivar root is 0);
an item sits at the depth of its leaf. Depths belong to one cultivar and are
never compared with another's.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import duckdb

from .. import cohort, read
from .execute import ExecuteResult
from .model import ItemRule, NodeRule, Query
from .normalize import NormalQuery, normalize

ITEMS_LAYER = "(items)"


@dataclass(frozen=True)
class CultivarExplanation:
    cultivar: str
    present: bool                        # the item has a leaf row in this cultivar
    chain_depth: int                     # depth of the item's leaf (0 for the pseudo cultivar)
    selecting: bool
    selecting_rules: tuple[dict[str, Any], ...]
    applicable: tuple[dict[str, Any], ...]   # nearest first; each has kind, op, depth, hops (+ node or item)
    winner: dict[str, Any] | None
    item_override: dict[str, Any] | None
    keep: bool
    reason: str


@dataclass(frozen=True)
class Explanation:
    item_id: str
    in_base: bool | None                 # None when no base is known (a bare query)
    cultivars: tuple[CultivarExplanation, ...]
    ignored_item_rule: dict[str, Any] | None
    verdict: bool
    reason: str


def _parent(con, tree_id, node) -> uuid.UUID | None:
    row = con.execute("SELECT branch_id FROM treedb.v_orchard WHERE tree_id = ? AND node_id = ?", [tree_id, node]).fetchone()
    return None if row is None else row[0]


def _ancestors(con, tree_id, node, limit: int) -> list[uuid.UUID]:
    """Branches above ``node``, nearest first."""
    out: list[uuid.UUID] = []
    cur = _parent(con, tree_id, node)
    while cur is not None:
        out.append(cur)
        if len(out) > limit:   # a valid chain uses distinct rows, so it is at most ``limit`` long
            raise read.AncestryCycleError(f"treedb.query: ancestry cycle: the chain above node {node} repeats an orchard row")
        cur = _parent(con, tree_id, cur)
    return out


def explain(con: duckdb.DuckDBPyConnection, subject: ExecuteResult | Query | NormalQuery, item_id: str,
            *, base: cohort.CohortRef | None = None) -> Explanation:
    """Why ``item_id`` is in or out. With an ``ExecuteResult`` the base is its cohort's base."""
    if isinstance(subject, ExecuteResult):
        nq = subject.normalized
        base = subject.cohort.base_cohort_id
    else:
        nq = normalize(con, subject)
    base_members = None if base is None else set(cohort.items(con, base))
    in_base = None if base_members is None else item_id in base_members

    rules_by_item = {r.item: r for r in nq.item_rules}
    effective_items = [r for r in nq.item_rules if base_members is None or r.item in base_members]
    ignored = None
    if item_id in rules_by_item and base_members is not None and item_id not in base_members:
        ignored = {**rules_by_item[item_id].to_json(), "reason": "out_of_base"}
    item_rule = rules_by_item.get(item_id) if ignored is None else None

    trees = dict(con.execute("SELECT name, id FROM treedb.tree_registry").fetchall())
    uuids = dict(con.execute("SELECT item_id, item_uuid FROM treedb.item_registry").fetchall())
    limit = int(con.execute("SELECT count(*) FROM treedb.orchard").fetchone()[0])
    names = list(nq.cultivars) or ([ITEMS_LAYER] if effective_items else [])

    out: list[CultivarExplanation] = []
    for name in names:
        tree_id = trees.get(name)
        node_rules = {r.node: r for r in nq.node_rules if r.cultivar == name}
        excluded = {n for n, r in node_rules.items() if r.op == "exclude"}

        def above(start) -> list[uuid.UUID]:
            return [] if tree_id is None else _ancestors(con, tree_id, start, limit)

        # This item's chain in this cultivar.
        chain = above(uuids[item_id]) if item_id in uuids else []
        depth_of = {str(n): len(chain) - 1 - i for i, n in enumerate(chain)}
        leaf_depth = len(chain)
        applicable: list[dict[str, Any]] = []
        if item_rule is not None:
            applicable.append({"kind": "item", "item": item_id, "op": item_rule.op, "depth": leaf_depth, "hops": 0})
        for hops, n in enumerate(chain, start=1):
            r = node_rules.get(str(n))
            if r is not None:
                applicable.append({"kind": "node", "node": str(n), "op": r.op, "depth": depth_of[str(n)], "hops": hops})
        winner = applicable[0] if applicable else None

        # Selecting includes: no exclude above them on their own chain.
        selecting: list[dict[str, Any]] = []
        for n, r in sorted(node_rules.items()):
            if r.op == "include" and not any(str(a) in excluded for a in above(uuid.UUID(n))):
                selecting.append({"kind": "node", "node": n, "op": "include"})
        for ir in effective_items:
            if ir.op == "include" and not any(str(a) in excluded for a in above(uuids[ir.item])):
                selecting.append({"kind": "item", "item": ir.item, "op": "include"})

        if winner is not None:
            keep = winner["op"] == "include"
            reason = f"nearest rule decides: {winner['kind']} {winner['op']} at depth {winner['depth']}"
        elif selecting:
            keep, reason = False, "no rule on the chain, and the cultivar has a selecting include"
        else:
            keep, reason = True, "no rule on the chain, and the cultivar has no selecting include"
        out.append(CultivarExplanation(
            cultivar=name, present=bool(chain), chain_depth=leaf_depth, selecting=bool(selecting),
            selecting_rules=tuple(selecting), applicable=tuple(applicable), winner=winner,
            item_override=None if item_rule is None else item_rule.to_json(), keep=keep, reason=reason))

    if in_base is False:
        return Explanation(item_id, in_base, tuple(out), ignored, False, "not in the base cohort: never in the result")
    out_in = [c.cultivar for c in out if not c.keep]
    if out_in:
        return Explanation(item_id, in_base, tuple(out), ignored, False, f"removed by cultivar(s) {out_in}")
    return Explanation(item_id, in_base, tuple(out), ignored, True,
                       "kept by every constrained cultivar" if out else "no constrained cultivar: base is unchanged")
