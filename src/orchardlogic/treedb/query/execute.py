"""Execute: a base cohort and a query give a new bounded cohort (phase P5B, D-025).

``execute(con, base_cohort, query)`` normalizes the query, runs the compiled SQL
over the base, and creates a cohort of kind ``execute`` (through
``cohort.create``, which re-checks result <= base, INV-19). The new cohort
records ``base_cohort_id``, the normalized expression and the cultivar
``state_version`` values (D-007). It has no notion of UI history, undo, neutral
or time.

Reported, never silent (INV-20, D-022, D-025):

* ``ignored_rules``: item rules whose item is outside the base, reason
  ``out_of_base``. They change nothing, not even which cultivars narrow.
* ``missing_items``: per constrained cultivar, base items with no leaf row in it.
  They get the verdict of "no rule" (or an item rule) and stay in the result
  when it says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import duckdb

from .. import cohort, read
from .compile import CYCLE_MARKER, compile_query
from .model import Query
from .normalize import NormalQuery, normalize


@dataclass(frozen=True)
class ExecuteResult:
    cohort: cohort.Cohort
    normalized: NormalQuery
    ignored_rules: tuple[dict[str, str], ...]
    missing_items: Mapping[str, tuple[str, ...]]
    sql: str
    items: tuple[str, ...]   # sorted members of ``cohort``

def execute(con: duckdb.DuckDBPyConnection, base_cohort: cohort.CohortRef, query: Query | NormalQuery) -> ExecuteResult:
    """Run ``query`` over ``base_cohort`` and return the new cohort with its provenance."""
    base = cohort.get(con, base_cohort)  # raises UnknownCohortError
    nq = normalize(con, query)
    base_items = cohort.items(con, base)
    in_base = set(base_items)
    ignored = tuple({**r.to_json(), "reason": "out_of_base"} for r in nq.item_rules if r.item not in in_base)
    sql = compile_query(nq, base.cohort_id)
    try:
        result_ids = [r[0] for r in con.execute(sql).fetchall()]
    except duckdb.Error as exc:
        if CYCLE_MARKER in str(exc):
            raise read.AncestryCycleError(str(exc)) from exc
        raise
    missing: dict[str, tuple[str, ...]] = {}
    for name in nq.cultivars:
        gone = read.resolve_items(con, name, base_items).missing_items
        if gone:
            missing[name] = tuple(gone)
    operands: dict[str, Any] = {"ignored_rules": list(ignored), "missing_items": {k: list(v) for k, v in missing.items()}}
    new = cohort.create(con, result_ids, "execute", base=base.cohort_id, operands=operands, expression=nq.to_json())
    return ExecuteResult(new, nq, ignored, missing, sql, tuple(sorted(result_ids)))
