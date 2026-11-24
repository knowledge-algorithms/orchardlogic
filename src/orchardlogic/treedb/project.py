"""Cohort projection: express a cohort as a compressed Steiner subtree of any cultivar (P4B).

``project(con, cohort, cultivar)`` reads the cohort's frozen membership (a set of leaf
``item_id`` values, INV-17), maps it to the cultivar's leaf entry rows (INV-6), calls
``read.compressed_subtree_for_items`` and attaches labels and cards. It returns a
``Projection``. Nothing is written: the projection is not a cohort, and the cultivar is
never mutated (it only reads; the read macros are installed on first use as in P3B).

Rulings applied:

* **D-021 (OD-4).** A cohort is membership only. The cultivar is always read as it is
  now. When its ``state_version`` differs from the one recorded in the cohort
  (``cohort.cultivar_state_versions``, D-007) the result carries ``moved`` (a ``Moved``
  notice with both versions). This never breaks by default; ``on_moved="error"`` raises
  ``CohortMovedError`` instead. With the default it also emits a ``CohortMovedWarning`` (Python
  warning, every time, text = ``Moved.message()``), per the user's D-021 ruling. Labels and cards come from the cultivar as it is now.
* **D-022 (OD-5).** ``on_missing="report"`` (default): present items are projected and
  the rest are listed in ``missing_items`` (registered, but not in this cultivar) and
  ``unknown_items``. ``on_missing="error"`` raises ``read.MissingItemsError``. Nothing
  is dropped silently (INV-20); membership is never narrowed (the hash stays the
  cohort's).
* **D-024 Q8.** ``components`` and ``is_forest`` pass P3B's notice through. A forest
  (a cohort spread over several components after a split) never raises.
* **D-018.** ``output="scipy"`` (or ``"both"``) also gives the normalized 0-based Z,
  leaves in ascending ``item_id`` order, plus the normalizer's context. The normalizer
  refuses anything that is not one tree, so for a forest ``z`` is None and
  ``normalization_error`` says why; the raw rows are still returned. ``raw_rows`` is
  filled for every ``output`` (the normalizer needs them anyway).

Labels and cards (``labels`` / ``cards``) are keyed by node id for every node that
appears in the raw rows (branches and leaves); nodes with none are absent.

In every case the three reads run in one transaction, so ``state_version``, rows and
labels describe one state.
"""

from __future__ import annotations

import json
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np

from orchardlogic.treedb import cohort as cohort_mod
from orchardlogic.treedb import normalize, read

OUTPUTS = ("raw", "scipy", "both")
ON_MOVED = ("report", "error")


class ProjectionError(ValueError):
    """Invalid projection request."""


class CohortMovedWarning(UserWarning):
    """D-021: a cohort was projected onto a cultivar that moved since the cohort was created.

    Emitted on every such projection with ``on_moved="report"`` (the default), next to the
    ``Projection.moved`` field. The text is ``Moved.message()``.
    """


class CohortMovedError(ProjectionError):
    """``on_moved="error"``: the cultivar moved since the cohort was created (D-021)."""

    def __init__(self, moved: "Moved"):
        self.moved = moved
        super().__init__(moved.message())


@dataclass(frozen=True)
class Moved:
    """Notice that the cultivar's ``state_version`` differs from the cohort's record.

    ``recorded_state_version`` is None when the cultivar did not exist when the cohort
    was created.
    """

    cultivar: str
    tree_id: uuid.UUID
    recorded_state_version: int | None
    current_state_version: int

    def message(self) -> str:
        rec = "absent" if self.recorded_state_version is None else self.recorded_state_version
        return (f"cultivar {self.cultivar!r} moved since the cohort was created "
                f"(state_version {rec} -> {self.current_state_version}); "
                "the cohort's membership is unchanged and is projected onto the cultivar as it is now")


@dataclass(frozen=True)
class Projection:
    """A cohort expressed in one cultivar, at one ``state_version``. Not persisted."""

    cohort_id: uuid.UUID
    membership_hash: str
    cohort_size: int
    cultivar: str
    tree_id: uuid.UUID
    state_version: int                         # the cultivar's, at evaluation time
    moved: Moved | None                        # D-021 notice; None when not moved
    output: str
    columns: tuple[str, ...]
    raw_rows: list[tuple]                      # PostgreSQL contract (INV-14), unchanged
    present_items: tuple[str, ...]             # cohort items found in the cultivar
    missing_items: tuple[str, ...]             # D-022
    unknown_items: tuple[str, ...]
    components: tuple[read.Component, ...]     # D-024 Q8
    z: np.ndarray | None = None                # normalized, 0-based (D-018)
    leaf_item_ids: list[str | None] | None = None
    context: dict[str, Any] | None = None
    normalization_error: str | None = None     # set when output asked for Z but it was refused
    labels: dict[uuid.UUID, dict[str, str]] = field(default_factory=dict)
    cards: dict[uuid.UUID, Any] = field(default_factory=dict)

    @property
    def is_forest(self) -> bool:
        return len(self.components) > 1

    @property
    def component_count(self) -> int:
        return len(self.components)


def _labels_and_cards(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID, raw_rows: list[tuple]):
    cols = read.COMPRESSED_COLUMNS
    ix = [cols.index(c) for c in ("branch_id", "l_node_id", "r_node_id")]
    nodes = sorted({r[i] for r in raw_rows for i in ix if r[i] is not None}, key=str)
    if not nodes:
        return {}, {}
    labels: dict[uuid.UUID, dict[str, str]] = {}
    for node, label_set, label in con.execute(
            "SELECT node_id, label_set, label FROM treedb.node_labels "
            "WHERE tree_id = ? AND list_contains(CAST(? AS UUID[]), node_id) ORDER BY node_id, label_set",
            [tree_id, nodes]).fetchall():
        labels.setdefault(node, {})[label_set] = label
    cards = {node: json.loads(card) for node, card in con.execute(
        "SELECT node_id, card FROM treedb.node_cards "
        "WHERE tree_id = ? AND list_contains(CAST(? AS UUID[]), node_id)", [tree_id, nodes]).fetchall()}
    return labels, cards


def project(
    con: duckdb.DuckDBPyConnection,
    cohort: cohort_mod.CohortRef,
    cultivar: str | uuid.UUID,
    *,
    output: str = "raw",
    on_missing: str = "report",
    on_moved: str = "report",
) -> Projection:
    """Project ``cohort`` onto ``cultivar`` (see the module docstring)."""
    if output not in OUTPUTS:
        raise ProjectionError(f"output must be one of {OUTPUTS}, got {output!r}")
    if on_moved not in ON_MOVED:
        raise ProjectionError(f"on_moved must be one of {ON_MOVED}, got {on_moved!r}")
    own = not read.in_transaction(con)
    if own:
        read.ensure_installed(con)  # installs the macros (DDL) first: a failed probe would abort our transaction
        con.execute("BEGIN TRANSACTION")
    try:
        result = _project(con, cohort, cultivar, output, on_missing, on_moved)
        if own:
            con.execute("COMMIT")
    except BaseException:
        if own:
            con.execute("ROLLBACK")
        raise
    if result.moved is not None:                           # on_moved="error" has already raised
        _warn_every_time(result.moved.message(), CohortMovedWarning)
    return result


def _warn_every_time(message: str, category: type[Warning]) -> None:
    """``warnings.warn`` without the per-call-site dedup registry (same pattern as mutate's split
    warning), attributed to the caller of ``project``. Active filters still apply."""
    try:
        frame = sys._getframe(2)                           # helper <- project <- caller
    except ValueError:
        frame = sys._getframe(1)
    warnings.warn_explicit(message, category, frame.f_code.co_filename, frame.f_lineno,
                           module=frame.f_globals.get("__name__", "<unknown>"), registry=None,
                           module_globals=frame.f_globals)


def _project(con, cohort, cultivar, output: str, on_missing: str, on_moved: str) -> Projection:
    header = cohort_mod.get(con, cohort)
    members = cohort_mod.items(con, header)
    tree_id, name, version = read.cultivar_version(con, cultivar)

    recorded = header.cultivar_state_versions.get(str(tree_id))
    rec_version = None if recorded is None else int(recorded["state_version"])
    moved = None
    if rec_version != version:
        moved = Moved(name, tree_id, rec_version, version)
        if on_moved == "error":
            raise CohortMovedError(moved)

    res = read.compressed_subtree_for_items(con, tree_id, members, on_missing=on_missing)
    labels, cards = _labels_and_cards(con, tree_id, res.rows)

    z = leaves = ctx = norm_err = None
    if output in ("scipy", "both"):
        by_uuid = {u: i for i, u in con.execute(
            "SELECT item_id, item_uuid FROM treedb.item_registry WHERE list_contains(CAST(? AS VARCHAR[]), item_id)",
            [list(res.resolution.item_rows)]).fetchall()}
        try:
            z, leaves, ctx = normalize.to_scipy_linkage(res.rows, item_ids_by_uuid=by_uuid, leaf_order="item_id")
        except normalize.NormalizationError as e:
            if not res.is_forest:
                raise
            norm_err = (f"cannot normalize a forest of {res.component_count} components (D-018: the normalizer "
                        f"refuses anything that is not one tree): {e}")

    return Projection(
        cohort_id=header.cohort_id, membership_hash=header.membership_hash, cohort_size=header.size,
        cultivar=name, tree_id=tree_id, state_version=res.resolution.state_version, moved=moved, output=output,
        columns=res.columns, raw_rows=res.rows, present_items=tuple(res.resolution.item_rows),
        missing_items=res.missing_items, unknown_items=res.unknown_items, components=res.components,
        z=z, leaf_item_ids=leaves, context=ctx, normalization_error=norm_err, labels=labels, cards=cards,
    )


__all__ = ["CohortMovedError", "CohortMovedWarning", "Moved", "OUTPUTS", "Projection", "ProjectionError", "project"]
