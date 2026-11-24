"""Normal form -> SQL over the ancestry closure (phase P5B, D-025).

``compile_query(nq, base_cohort_id)`` returns one SQL statement (readable CTEs)
that lists the result ``item_id`` values. ``explain_sql`` returns the same text
with comments, for documentation and debugging. Semantics (docs/QUERY_LANGUAGE.md):

* Each cultivar named by a node rule is *constrained*. If no cultivar is, but
  there are item rules, one pseudo cultivar ``(items)`` with no tree is
  constrained, so a lone item rule works on the flat item list.
* Per item and constrained cultivar, the **nearest rule** on the item's own
  ancestor chain decides (item rules sit on the leaf, nearest of all). The
  chain is walked inside that cultivar only; depths are never compared across
  cultivars.
* A *selecting* include has no exclude above it on its chain. With a selecting
  include, an item without a rule is out of that cultivar; without one, in.
* A cycle in a walked chain raises (``CYCLE_MARKER``), it never ends the walk quietly.
* The result is the base items that are in every constrained cultivar (AND).
* Item rules for items outside the base are dropped (``item_rules`` CTE); the
  Python side reports them as ``ignored_rules``.
"""

from __future__ import annotations

import uuid

from .normalize import NormalQuery

_BASE_PLACEHOLDER = "<base cohort id>"

#: Text of the cycle error() below; ``execute`` maps it to ``read.AncestryCycleError``.
CYCLE_MARKER = "treedb.query: ancestry cycle"

# The next hop count along a chain. A valid chain uses distinct orchard rows, one per hop, so
# hops never exceed the number of rows; one more means a row repeats (a cycle), and the
# recursion raises (same idea as the CYC-2 depth guard of get_ancestry_paths).
_HOP = ("CASE WHEN {h} + 1 <= (SELECT count(*) FROM treedb.orchard) THEN {h} + 1 "
        "ELSE error('" + CYCLE_MARKER + ": a chain repeats an orchard row') END")


def _q(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _values(rows: list[str], empty: str) -> str:
    return "VALUES\n        " + ",\n        ".join(rows) if rows else empty


def compile_query(nq: NormalQuery, base_cohort_id: uuid.UUID | str | None = None) -> str:
    """The SQL of ``nq`` against the base cohort ``base_cohort_id``."""
    base = _BASE_PLACEHOLDER if base_cohort_id is None else str(base_cohort_id)
    node_rows = [f"({_q(r.cultivar)}, {_q(r.node)}::UUID, {_q(r.op)})" for r in nq.node_rules]
    item_rows = [f"({_q(r.item)}, {_q(r.op)})" for r in nq.item_rules]
    node_values = _values(node_rows, "SELECT NULL::VARCHAR, NULL::UUID, NULL::VARCHAR WHERE false")
    item_values = _values(item_rows, "SELECT NULL::VARCHAR, NULL::VARCHAR WHERE false")
    return f"""\
WITH RECURSIVE
-- The cohort being narrowed. The result is always a subset of it.
base AS (
    SELECT i.item_id, r.item_uuid
    FROM treedb.cohort_items({_q(base)}) i
    JOIN treedb.item_registry r USING (item_id)
),
-- Node rules: a branch of one cultivar is included or excluded.
node_rules(cultivar, node_id, op) AS (
    {node_values}
),
-- Item rules, for items in the base only (an out-of-base rule is ignored, never a resurrection).
item_rules(item_id, op) AS (
    SELECT v.item_id, v.op
    FROM (
    {item_values}
    ) v(item_id, op)
    JOIN base USING (item_id)
),
-- Constrained cultivars: those with node rules; else one pseudo cultivar '(items)' for item rules alone.
constrained(cultivar, tree_id) AS (
    SELECT DISTINCT n.cultivar, t.id
    FROM node_rules n JOIN treedb.tree_registry t ON t.name = n.cultivar
    UNION ALL
    SELECT '(items)', NULL::UUID
    WHERE NOT EXISTS (SELECT 1 FROM node_rules) AND EXISTS (SELECT 1 FROM item_rules)
),
-- Ancestry closure of every base item inside each constrained cultivar: hops = 1 is the parent branch.
item_chain(cultivar, item_id, node_id, hops) AS (
    SELECT c.cultivar, b.item_id, o.branch_id, 1
    FROM constrained c
    CROSS JOIN base b
    JOIN treedb.v_orchard o ON o.tree_id = c.tree_id AND o.node_id = b.item_uuid
    UNION ALL
    SELECT ch.cultivar, ch.item_id, o.branch_id, {_HOP.format(h='ch.hops')}
    FROM item_chain ch
    JOIN constrained c ON c.cultivar = ch.cultivar
    JOIN treedb.v_orchard o ON o.tree_id = c.tree_id AND o.node_id = ch.node_id
),
-- Ancestry closure of every rule node (to tell a selecting include from a restoring one).
node_chain(cultivar, rule_node, node_id, hops) AS (
    SELECT c.cultivar, r.node_id, o.branch_id, 1
    FROM node_rules r
    JOIN constrained c USING (cultivar)
    JOIN treedb.v_orchard o ON o.tree_id = c.tree_id AND o.node_id = r.node_id
    UNION ALL
    SELECT nc.cultivar, nc.rule_node, o.branch_id, {_HOP.format(h='nc.hops')}
    FROM node_chain nc
    JOIN constrained c ON c.cultivar = nc.cultivar
    JOIN treedb.v_orchard o ON o.tree_id = c.tree_id AND o.node_id = nc.node_id
),
-- Rules on each item's own chain. An item rule is a rule on the leaf (hops 0) in every constrained cultivar.
applicable(cultivar, item_id, hops, op) AS (
    SELECT ch.cultivar, ch.item_id, ch.hops, r.op
    FROM item_chain ch
    JOIN node_rules r ON r.cultivar = ch.cultivar AND r.node_id = ch.node_id
    UNION ALL
    SELECT c.cultivar, i.item_id, 0, i.op
    FROM constrained c CROSS JOIN item_rules i
),
-- The nearest rule decides (hops are distinct on one chain, so there is no tie).
nearest(cultivar, item_id, op) AS (
    SELECT cultivar, item_id, arg_min(op, hops) FROM applicable GROUP BY cultivar, item_id
),
-- A cultivar is selecting when it has an include with no exclude above it.
selecting(cultivar) AS (
    SELECT r.cultivar
    FROM node_rules r
    WHERE r.op = 'include' AND NOT EXISTS (
        SELECT 1 FROM node_chain nc
        JOIN node_rules x ON x.cultivar = nc.cultivar AND x.node_id = nc.node_id AND x.op = 'exclude'
        WHERE nc.cultivar = r.cultivar AND nc.rule_node = r.node_id)
    UNION
    SELECT c.cultivar
    FROM constrained c JOIN item_rules i ON i.op = 'include'
    WHERE NOT EXISTS (
        SELECT 1 FROM item_chain ch
        JOIN node_rules x ON x.cultivar = ch.cultivar AND x.node_id = ch.node_id AND x.op = 'exclude'
        WHERE ch.cultivar = c.cultivar AND ch.item_id = i.item_id)
),
-- One verdict per item and constrained cultivar.
verdict(cultivar, item_id, keep) AS (
    SELECT c.cultivar, b.item_id,
           CASE n.op
               WHEN 'include' THEN true
               WHEN 'exclude' THEN false
               ELSE NOT EXISTS (SELECT 1 FROM selecting s WHERE s.cultivar = c.cultivar)
           END
    FROM constrained c
    CROSS JOIN base b
    LEFT JOIN nearest n ON n.cultivar = c.cultivar AND n.item_id = b.item_id
)
-- Cultivars combine with AND: one out-verdict removes the item.
SELECT b.item_id
FROM base b
WHERE NOT EXISTS (SELECT 1 FROM verdict v WHERE v.item_id = b.item_id AND NOT v.keep)
ORDER BY b.item_id"""


def explain_sql(nq: NormalQuery, base_cohort_id: uuid.UUID | str | None = None) -> str:
    """Readable SQL for a normal form (the exact text ``execute`` runs, see ``compile_query``)."""
    return compile_query(nq, base_cohort_id)
