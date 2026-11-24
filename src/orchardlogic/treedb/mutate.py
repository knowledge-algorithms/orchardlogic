"""Structural mutation of a cultivar: attach (graft) and detach (prune). Phase P3C.

Ports of the installed PostgreSQL functions (D-002, D-009):

* ``attach_node`` = ``graph.orchard_attach_node(p_new_node_id uuid, p_target_orchard_id uuid,
  p_new_edge_ix integer, p_coph_distance numeric) RETURNS integer``
  (``reference/pg_installed/graph/function/orchard_attach_node_uuid_uuid_integer_numeric.sql``,
  cited ``at:<line>``; same body as migration ``014:251-310``).
* ``detach_node`` = ``graph.orchard_detach_node(p_target_orchard_id uuid) RETURNS integer``
  with the 019 ``edge_ix`` fix
  (``reference/pg_installed/graph/function/orchard_detach_node_uuid.sql``, cited
  ``dt:<line>``; same body as migration ``019:37-83``).

Call contract: ``OrchardRepo.attach_node`` / ``detach_node``
(``reference/graph-engine/src/graph_engine/orchard/repo.py:229-325``). Same
parameters, in the PG order, with the same meaning; ``coph_distance=None`` asks
for the PG default (target row distance minus 0.001, at:28-29). ``repo.py``
opens a connection, calls the function and commits; here each call runs in ONE
DuckDB transaction that commits, or rolls back everything on any failure. The
PG return value is ``result.rowcount``. The ``invalidate_cache`` callback of
repo.py has no TreeDB counterpart: ``state_version`` (D-007) is the signal.

Structure: every data access is a SQL statement in ``sql/mutate/*.sql`` (one
statement per plpgsql statement, cited line by line there); the plpgsql
control flow (detach's two ``IF ... RAISE`` blocks) is Python here, each block
quoting the original. TreeDB additions, all outside the ported statements:

* D-007: ``tree_registry.state_version`` +1 inside the transaction of every
  committed structural change (not on rollback, not on a no-op), for EVERY
  tree whose rows changed: the target's tree, and any other tree whose graft
  row the detach rewire touched (GRAFT-1; ``DetachResult.collateral_tree_ids``
  plus a WARNING log).
* P3A note n1: ``store.integrity_violations`` runs before COMMIT; a NEW
  violation of a foreign-key equivalent raises ``IntegrityViolationError``
  (SQLSTATE 23503); a new violation of only the TreeDB rule "a leaf row names a
  registered item" raises ``UnregisteredItemError`` (no SQLSTATE: PG accepts
  it). Both roll back. Newly orphaned
  labels / cards are kept and reported (``orphaned_annotations``), not raised.
* D-016b (final, G6 D-037): ``attach_node`` raises for an unknown target and
  for a subject that already has a parent in the target's tree, where PG
  returns ok (PG's "ok" is treated as incidental). One switch,
  ``D016B_STRICT_ATTACH`` (or ``strict=`` per call), and one code path,
  ``_d016b_strict_checks``; ``strict=False`` stays only as the PG-faithful
  reproducer for the golden replay.
* G3 D-024 Q7, CYC-1 (a divergence): ``attach_node`` refuses an attach that
  would make the subject its own ancestor in the target's tree
  (``CyclicAttachError``), on BOTH paths; ``_cycle_guard`` is not part of the
  D-016b switch.
* G6 D-038 (amends G3 D-024 Q5), CYC-3 (a divergence): ``detach_node`` refuses
  a detach that would leave any tree whose rows it changed (its own tree and
  every collateral tree of the GRAFT-1 rewire) cyclic (``CyclicDetachError``),
  checked after the rewire and before COMMIT; ``_detach_cycle_guard``, always
  run, not part of the D-016b switch. The rewire itself stays PG-faithful.
* G3 D-024 Q4 (INT-1): a leaf attach (``edge_ix`` 1) may pass the item's
  external ``item_id`` (and ``source_ref``); an item not yet registered is
  registered in the same transaction and listed in ``AttachResult.new_items``.
  A bare unknown UUID without ``item_id`` is still refused
  (``UnregisteredItemError``). ``_register_item`` is the one code path.
* D-016c: a detach that deletes a branch with no parent row in its own tree
  (a child of a root) splits the tree, exactly as in PG; TreeDB is loud on
  every split: ``TreeSplitWarning``, a WARNING on logger ``treedb.mutate`` and
  ``DetachResult.split``.

Porting notes: ``docs/porting/P3C.md``.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Callable

import duckdb

from orchardlogic.treedb import store
from orchardlogic.treedb.errors import (
    CyclicAttachError,
    CyclicDetachError,
    DuplicateAttachError,
    IntegrityViolationError,
    InvalidParameterError,
    ItemConflictError,
    ItemRegistrationError,
    MutationError,
    NoSiblingError,
    NotNullViolationError,
    OrchardRowNotFoundError,
    TransactionStateError,
    TreeSplitWarning,
    UniqueViolationError,
    UnknownAttachTargetError,
    UnregisteredItemError,
)
from orchardlogic.treedb.fixtures import item_uuid

logger = logging.getLogger("treedb.mutate")

#: D-016b switch (G2 2026-11-09; final at G6 2026-11-10, D-037). ``True`` (the
#: decided behaviour): TreeDB keeps the fixtures' ``expect: error`` for an
#: unknown attach target and for a subject that already has a parent in the
#: tree. ``False``: PostgreSQL-faithful (ok, 0 rows / a second parent), kept
#: only as the PG reproducer for the golden replay (D-037).
D016B_STRICT_ATTACH: bool = True

#: ``store.integrity_violations`` checks about TreeDB annotations. A detach can
#: orphan them (labels / cards of a removed node); they are reported, not raised.
ANNOTATION_CHECKS = frozenset({
    "node_labels (tree_id, node_id) -> node of that tree",
    "node_cards (tree_id, node_id) -> node of that tree",
})

#: ``store.integrity_violations`` checks that are TreeDB-only rules with no
#: PostgreSQL foreign key behind them (INT-1). When ONLY these are newly
#: violated, the error is ``UnregisteredItemError`` with ``sqlstate None``.
TREEDB_ONLY_CHECKS = frozenset({
    "orchard leaf node_id -> item_registry.item_uuid",
})


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttachResult:
    """Outcome of a committed ``attach_node`` call."""

    rowcount: int                      # PG return value: rows of the last INSERT (at:47-52); 2, or 0 for a no-op
    changed: bool                      # False only for the PG-faithful no-op (unknown target)
    target_orchard_id: uuid.UUID | None
    new_node_id: uuid.UUID | None
    new_branch_id: uuid.UUID | None    # the branch this call created (at:25, :34-35)
    tree_id: uuid.UUID | None
    tree_name: str | None
    state_version: int | None          # after commit (D-007); None when nothing changed
    strict: bool                       # the D-016b path this call used
    orphaned_annotations: dict[str, list[tuple]] = field(default_factory=dict)
    item_id: str | None = None         # the external item id passed (D-024 Q4), else None
    new_items: tuple[str, ...] = ()    # item ids this call registered in item_registry (D-024 Q4)


@dataclass(frozen=True)
class DetachResult:
    """Outcome of a committed ``detach_node`` call.

    D-016c fields: ``split`` is True when the deleted branch had no parent row
    in its own tree (``removed_branch_id in roots_before``: the target was a
    child of a root); then nothing in the tree is rewired and the sibling, if
    it is a branch, becomes a root. ``new_root_ids`` are the tree's roots that
    did not exist before this call; ``detached_root_id`` is the target node
    when it is a branch (it stays in the tree, parentless, as in PG);
    ``dropped_item_ids`` are items that no longer hang anywhere in the
    cultivar; ``stored_root_deleted`` says the import root
    (``tree_registry.root_branch_id``, ROOT-1) was the deleted branch.

    GRAFT-1 fields: the dt:38-41 rewire has no tree filter, so it also rewires
    rows of OTHER trees that name the deleted branch (``edge_ix`` 3 grafts).
    ``rewired_rows`` / ``rewired_row_ids`` count and list every rewired row,
    in any tree; ``collateral_tree_ids`` are the other trees among them, each
    of whose ``state_version`` was also bumped in the same transaction
    (``collateral_state_versions``, D-007). A split can therefore have
    ``rewired_rows > 0`` (a grafted root).
    """

    rowcount: int                      # PG return value: tree_branches rows deleted (dt:33-36)
    target_orchard_id: uuid.UUID
    tree_id: uuid.UUID
    tree_name: str
    state_version: int                 # after commit (D-007)
    removed_branch_id: uuid.UUID       # v_branch_id (dt:12-15): the target's parent branch
    detached_node_id: uuid.UUID        # v_target_node_id
    sibling_node_id: uuid.UUID         # v_sibling_node_id (dt:21-26)
    sibling_edge_ix: int               # v_sibling_edge_ix (019)
    rewired_rows: int                  # rows of the dt:38-41 UPDATE, in any tree
    split: bool                        # D-016c: removed_branch_id in roots_before
    stored_root_deleted: bool
    roots_before: tuple[uuid.UUID, ...]
    roots_after: tuple[uuid.UUID, ...]
    new_root_ids: tuple[uuid.UUID, ...]
    detached_root_id: uuid.UUID | None
    dropped_item_ids: tuple[str, ...]
    orphaned_annotations: dict[str, list[tuple]] = field(default_factory=dict)
    rewired_row_ids: tuple[uuid.UUID, ...] = ()            # orchard ids the dt:38-41 UPDATE touched
    collateral_tree_ids: tuple[uuid.UUID, ...] = ()        # other trees among them (GRAFT-1)
    collateral_state_versions: dict[uuid.UUID, int] = field(default_factory=dict)   # after commit (D-007)


# ---------------------------------------------------------------------------
# SQL statements (src/treedb/sql/mutate/*.sql)
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^-- name: ([A-Za-z_]\w*)[ \t]*$", re.M)
_PARAM_RE = re.compile(r"\$([A-Za-z_]\w*)")


@lru_cache(maxsize=None)
def _load_sql(file: str) -> dict[str, str]:
    """``{step name: statement}`` of ``sql/mutate/<file>.sql``, in file order."""
    text = resources.files("orchardlogic.treedb").joinpath(f"sql/mutate/{file}.sql").read_text(encoding="utf-8")
    parts = _NAME_RE.split(text)
    out: dict[str, str] = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        if name in out:
            raise RuntimeError(f"sql/mutate/{file}.sql: step {name!r} defined twice")
        out[name] = body.strip()
    return out


def _sql_param(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_sql_param(v) for v in value]
    return str(value) if isinstance(value, uuid.UUID) else value


def _step(con: duckdb.DuckDBPyConnection, file: str, name: str, **params: Any) -> duckdb.DuckDBPyConnection:
    """Run one named statement with exactly the ``$parameters`` it uses."""
    sql = _load_sql(file)[name]
    used = sorted(set(_PARAM_RE.findall(sql)))
    missing = [p for p in used if p not in params]
    if missing:
        raise RuntimeError(f"{file}.{name}: missing parameters {missing}")
    if used:
        return con.execute(sql, {p: _sql_param(params[p]) for p in used})
    return con.execute(sql)


# ---------------------------------------------------------------------------
# Parameter handling (PG binds uuid / integer / numeric at the call)
# ---------------------------------------------------------------------------

def _uuid_param(value: Any, name: str) -> uuid.UUID | None:
    """uuid.UUID, its text, or None (SQL NULL). Bad text: 22P02, as PG's uuid input."""
    if value is None or isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        try:
            return uuid.UUID(value)
        except ValueError:
            raise InvalidParameterError(f'invalid input syntax for type uuid: "{value}"', parameter=name) from None
    raise InvalidParameterError(f"{name} must be a UUID or its text, not {type(value).__name__}", parameter=name)


def _int_param(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidParameterError(f"{name} must be an integer, not {value!r}", parameter=name)
    return value


def _distance_param(value: Any) -> float | None:
    """PG ``numeric`` -> ``::double precision`` (at:30) gives the caller's double.

    The PG harness binds Python's ``repr`` decimal (tools/pg_reference/capture.py:179-183),
    whose float8 is the float itself; DuckDB receives the double directly [AT-2]. Not
    rounded (D-012 A).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidParameterError(f"coph_distance must be a number, not {value!r}", parameter="coph_distance")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise InvalidParameterError(f"coph_distance must be a number, not {value!r}",
                                    parameter="coph_distance") from None


def _pg_text(value: uuid.UUID | None) -> str:
    """How plpgsql RAISE formats a uuid argument (``<NULL>`` for NULL)."""
    return "<NULL>" if value is None else str(value)


# ---------------------------------------------------------------------------
# Transaction coordinator
# ---------------------------------------------------------------------------

def _map_duckdb_error(exc: duckdb.Error) -> MutationError:
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, duckdb.ConstraintException):
        # DuckDB 2.0 words this two ways: 'Duplicate key "..." violates unique constraint'
        # (single-row insert) and 'PRIMARY KEY or UNIQUE constraint violation: duplicate
        # key ...' (INSERT ... SELECT); both are tested.
        if "unique constraint" in low or "primary key" in low or "duplicate key" in low:
            return UniqueViolationError(msg)
        if "not null constraint" in low:
            return NotNullViolationError(msg)
    if isinstance(exc, duckdb.ConversionException):
        return InvalidParameterError(msg)
    return MutationError(msg)


def _new_violations(before: dict[str, list[tuple]], after: dict[str, list[tuple]]) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for name, rows in after.items():
        old = set(before.get(name, []))
        new = [r for r in rows if r not in old]
        if new:
            out[name] = new
    return out


def _run_in_transaction(con: duckdb.DuckDBPyConnection, body: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run ``body`` in ONE transaction, check integrity, bump ``state_version``, commit.

    ``body`` returns a dict; when ``body()["tree_id"]`` is set and
    ``body()["changed"]`` is true, that tree's ``state_version`` is bumped
    (D-007), and so is the version of every tree in
    ``body()["collateral_tree_ids"]`` (GRAFT-1), in tree-id order. Any
    exception rolls the whole transaction back, including every bump.
    """
    try:
        con.execute("BEGIN TRANSACTION")
    except duckdb.TransactionException as exc:
        raise TransactionStateError(
            "the connection is already inside a transaction; a TreeDB mutation opens and commits its own"
        ) from exc
    try:
        before = store.integrity_violations(con)
        out = body()
        if out.get("changed") and out.get("tree_id") is not None:
            row = _step(con, "common", "bump_state_version", tree_id=out["tree_id"]).fetchone()
            out["state_version"], out["tree_name"], out["stored_root_id"] = int(row[0]), row[1], row[2]
            out["collateral_state_versions"] = {}
            for other in sorted(out.get("collateral_tree_ids", ()), key=str):
                crow = _step(con, "common", "bump_state_version", tree_id=other).fetchone()
                out["collateral_state_versions"][other] = int(crow[0])
        new = _new_violations(before, store.integrity_violations(con))
        hard = {k: v for k, v in new.items() if k not in ANNOTATION_CHECKS}
        if hard:
            summary = "; ".join(f"{k}: {len(v)} row(s)" for k, v in sorted(hard.items()))
            if hard.keys() <= TREEDB_ONLY_CHECKS:
                raise UnregisteredItemError(
                    "mutation would break a TreeDB-only reference rule (PostgreSQL would accept it): " + summary
                    + ". To attach a new item, pass its external item_id to attach_node; it is then "
                      "registered in the same transaction (G3 D-024 Q4)",
                    violations=hard,
                )
            raise IntegrityViolationError(
                "mutation would break references that PostgreSQL foreign keys enforce: " + summary,
                violations=hard,
            )
        out["orphaned_annotations"] = {k: v for k, v in new.items() if k in ANNOTATION_CHECKS}
        con.execute("COMMIT")
    except BaseException as exc:
        try:
            con.execute("ROLLBACK")
        except duckdb.Error:
            pass                                 # e.g. COMMIT itself failed and ended the transaction
        if isinstance(exc, duckdb.Error):
            raise _map_duckdb_error(exc) from exc
        raise
    return out


# ---------------------------------------------------------------------------
# attach_node
# ---------------------------------------------------------------------------

def _d016b_strict_checks(con: duckdb.DuckDBPyConnection, target: uuid.UUID | None, node: uuid.UUID | None) -> None:
    """THE D-016b code path (TreeDB, final: G6 D-037). Runs only when strict.

    PostgreSQL has no such checks: an unknown target returns 0 with no change,
    and a subject that already hangs in the tree gets a second parent
    (PG_BASELINE §8). Reads only; raises before anything is written.
    """
    found = _step(con, "attach_node", "strict_target_rows").fetchone()[0]
    if found == 0:
        raise UnknownAttachTargetError(
            f"No orchard row found for id {_pg_text(target)} "
            "(attach target; PostgreSQL returns 0 and changes nothing; TreeDB raises under D-016b, final: G6 D-037)",
            target_orchard_id=target,
        )
    dup = _step(con, "attach_node", "strict_subject_rows").fetchone()[0]
    if dup:
        raise DuplicateAttachError(
            f"node {_pg_text(node)} already has {dup} parent row(s) in the target's tree "
            "(PostgreSQL gives it another parent; TreeDB raises under D-016b, final: G6 D-037)",
            new_node_id=node, existing_rows=int(dup),
        )


def _cycle_guard(con: duckdb.DuckDBPyConnection, target: uuid.UUID | None, node: uuid.UUID | None) -> None:
    """CYC-1, THE cycle guard (TreeDB, G3 D-024 Q7, a divergence). Runs on both paths.

    PostgreSQL has no ancestry check (at:8-52) and commits the cycle. Reads only;
    raises before anything is written, so the rollback leaves rows and
    ``state_version`` unchanged.
    """
    hits = _step(con, "attach_node", "cycle_subject_is_ancestor").fetchone()[0]
    if hits:
        raise CyclicAttachError(
            f"attaching node {_pg_text(node)} above orchard row {_pg_text(target)} would make it its own "
            "ancestor: it is that row's parent branch or an ancestor of it in the same tree "
            "(PostgreSQL commits the cycle; TreeDB refuses it, G3 D-024 Q7, CYC-1)",
            new_node_id=node, target_orchard_id=target,
        )


def _detach_cycle_guard(con: duckdb.DuckDBPyConnection, tree_ids: tuple[uuid.UUID, ...],
                        target: uuid.UUID | None, removed_branch: uuid.UUID,
                        rewired_ids: frozenset[uuid.UUID] = frozenset()) -> None:
    """CYC-3, THE detach cycle guard (TreeDB, G6 D-038, a divergence). Runs on every detach.

    Called after the dt:38-41 rewire (``rewire_parent``) and before COMMIT, with
    every tree whose rows the call changed. PostgreSQL has no such check and
    commits the cycle. Reads only; raising here rolls the whole call back
    (``_run_in_transaction``), before any ``state_version`` bump, and nothing is
    logged or warned, because ``detach_node`` reports only after COMMIT.

    Message only: when no reported row is one the rewire touched (``rewired_ids``), the
    rewire did not close the loop, so the rows were already cyclic (written outside
    ``treedb.mutate``) and the message says so. The check itself is unchanged (D-038).
    """
    rows = _step(con, "common", "detach_cycle_rows_in_changed_trees", tree_ids=list(tree_ids)).fetchall()
    if not rows:
        return
    names: dict[uuid.UUID, str] = {}
    for r in rows:
        names.setdefault(r[0], r[1])
    shown = "; ".join(f"{names[t]!r} ({t}): " + ", ".join(
        f"row {r[2]} branch {r[3]} -> node {r[4]} (edge_ix {r[5]})" for r in rows if r[0] == t)
        for t in names)
    if any(r[2] in rewired_ids for r in rows):
        why = ("PostgreSQL commits the cycle (the dt:38-41 rewire has no tree filter, GRAFT-1); "
               "TreeDB refuses it and changes nothing (G6 D-038, CYC-3)")
        head = "would leave"
    else:
        why = ("already cyclic before this call (rows written outside treedb.mutate), not made by this "
               "detach; TreeDB still refuses it and changes nothing (G6 D-038 checks the whole changed tree, CYC-3)")
        head = "leaves"
    raise CyclicDetachError(
        f"detaching orchard row {_pg_text(target)} (deleting branch {removed_branch}) {head} "
        f"{len(names)} tree(s) cyclic, {len(rows)} row(s) on a cycle: {shown}. {why}",
        target_orchard_id=target, removed_branch_id=removed_branch, tree_ids=tuple(names),
        tree_names=tuple(names.values()), rows=tuple((r[0], r[2], r[3], r[4], int(r[5])) for r in rows),
    )


def _check_item_args(item_id: Any, source_ref: Any, edge: int | None,
                     node: uuid.UUID | None) -> uuid.UUID | None:
    """D-024 Q4 argument rules, checked before the transaction. Returns the subject UUID
    (the item's CONTRACTS UUID, INV-5, when ``new_node_id`` is None and ``item_id`` is given)."""
    if item_id is None:
        if source_ref is not None:
            raise ItemRegistrationError("source_ref is only used to register an item; pass item_id with it",
                                        source_ref=source_ref)
        return node
    # INV-1: a non-empty string with no leading or trailing whitespace.
    if not isinstance(item_id, str) or not item_id or item_id != item_id.strip():
        raise ItemRegistrationError(f"item_id must be a non-empty string without surrounding whitespace "
                                    f"(INV-1), not {item_id!r}", item_id=item_id)
    if source_ref is not None and not isinstance(source_ref, str):
        raise ItemRegistrationError(f"source_ref must be a string, not {type(source_ref).__name__}",
                                    source_ref=source_ref)
    if edge != 1:
        raise ItemRegistrationError(
            f"item_id is only accepted for a leaf attach (edge_ix 1), not edge_ix {edge!r}; "
            "edge_ix 2/3 behaviour is unchanged (D-024 Q4)", item_id=item_id, new_edge_ix=edge)
    return item_uuid(item_id) if node is None else node


def _register_item(con: duckdb.DuckDBPyConnection, item_id: str, source_ref: str | None,
                   node: uuid.UUID) -> tuple[str, ...]:
    """THE item-registration code path (TreeDB, G3 D-024 Q4). Runs only when ``item_id`` is given.

    Compares the registry with the request first and raises ``ItemConflictError`` on any
    difference (nothing has been written at that point); otherwise registers the item when it
    is new and the target row exists. Returns the newly registered item ids.
    """
    rows = _step(con, "attach_node", "item_lookup", item_id=item_id).fetchall()
    by_id = next((r for r in rows if r[0] == item_id), None)
    by_uuid = next((r for r in rows if r[1] == node and r[0] != item_id), None)
    if by_id is not None and by_id[1] != node:
        raise ItemConflictError(
            f"item {item_id!r} is already registered under UUID {by_id[1]}, not {node}",
            item_id=item_id, new_node_id=node, registered_uuid=by_id[1])
    if by_uuid is not None:
        raise ItemConflictError(
            f"UUID {node} is already registered as item {by_uuid[0]!r}, not {item_id!r}",
            item_id=item_id, new_node_id=node, registered_item_id=by_uuid[0])
    if by_id is None and node != item_uuid(item_id):
        # The import refuses such a row (ingest._register_items) and canonical.item_for_uuid
        # rejects the whole registry when one exists; registering it would poison the store.
        raise ItemConflictError(
            f"UUID {node} is not the CONTRACTS UUID of item {item_id!r} (INV-5: {item_uuid(item_id)}); "
            "TreeDB registers an item only under that UUID, as the import does",
            item_id=item_id, new_node_id=node, contracts_uuid=item_uuid(item_id))
    if by_id is not None and source_ref is not None and by_id[2] != source_ref:
        raise ItemConflictError(
            f"item {item_id!r} is already registered with source_ref {by_id[2]!r}, not {source_ref!r}",
            item_id=item_id, source_ref=source_ref, registered_source_ref=by_id[2])
    return tuple(r[0] for r in _step(con, "attach_node", "item_register",
                                     item_id=item_id, source_ref=source_ref).fetchall())


def attach_node(
    con: duckdb.DuckDBPyConnection,
    new_node_id: uuid.UUID | str | None,
    target_orchard_id: uuid.UUID | str | None,
    new_edge_ix: int | None,
    coph_distance: float | None = None,
    *,
    strict: bool | None = None,
    item_id: str | None = None,
    source_ref: str | None = None,
) -> AttachResult:
    """Insert a new branch on the edge above orchard row ``target_orchard_id`` and
    hang ``new_node_id`` from it with edge type ``new_edge_ix``.

    Same parameters and meaning as ``graph.orchard_attach_node`` (at:1) and
    ``OrchardRepo.attach_node`` (repo.py:229-285): ``new_node_id`` is the
    subject (an item UUID for ``edge_ix`` 1, a branch id for 2, a node of
    another tree for 3 ``external_graft``); ``coph_distance=None`` uses the
    target row's distance minus 0.001 (at:28-29). The target row keeps its id
    and distance and now points at the new branch with ``edge_ix`` 2; the new
    branch has two rows, the old node and the subject, both at the new
    distance. Distances are not rounded (D-012 A).

    ``strict`` selects the D-016b path (default ``D016B_STRICT_ATTACH``). D-016b is
    final (G6 D-037): the default refuses; ``strict=False`` is kept only as the
    PostgreSQL-faithful reproducer.

    **Cycle guard (CYC-1, G3 D-024 Q7, both paths).** An attach whose subject is
    the target row's parent branch or one of its ancestors in the tree would make
    the subject its own ancestor; it raises ``CyclicAttachError`` (PostgreSQL
    commits it).

    **Item registration (G3 D-024 Q4).** For a leaf attach (``new_edge_ix`` 1),
    ``item_id`` names the item's external id; ``new_node_id`` may then be None
    (the item's CONTRACTS UUID, INV-5, is used) or must be that UUID. An item
    not yet in ``treedb.item_registry`` is registered in the same transaction,
    with ``source_ref``, and listed in ``AttachResult.new_items``. An item
    already registered under the same UUID is used as is. ``ItemConflictError``
    when the registry holds a different identity (the item under another UUID,
    the UUID under another item, a different ``source_ref``) or the UUID given
    for a new item is not its CONTRACTS UUID (the import refuses that too).
    ``ItemRegistrationError`` for ``item_id`` on an ``edge_ix`` 2/3 attach, an
    invalid ``item_id`` (INV-1) or ``source_ref`` without ``item_id``. Without
    ``item_id``, an unregistered leaf UUID is still refused
    (``UnregisteredItemError``). The PG-faithful unknown-target no-op
    (``strict=False``) registers nothing.

    Raises a ``treedb.errors.MutationError`` subclass and writes nothing on
    any failure.
    """
    strict = D016B_STRICT_ATTACH if strict is None else bool(strict)
    p_node = _uuid_param(new_node_id, "new_node_id")
    p_target = _uuid_param(target_orchard_id, "target_orchard_id")
    p_edge = _int_param(new_edge_ix, "new_edge_ix")
    p_coph = _distance_param(coph_distance)
    # TreeDB G3 D-024 Q4: argument rules of the optional item registration.
    p_node = _check_item_args(item_id, source_ref, p_edge, p_node)

    def body() -> dict[str, Any]:
        f = "attach_node"
        # at:8      DROP TABLE IF EXISTS new_linkage;
        _step(con, f, "drop_new_linkage")
        # at:9-14   CREATE TEMP TABLE new_linkage AS SELECT p_new_node_id, ...;
        _step(con, f, "create_new_linkage", new_node_id=p_node, target_orchard_id=p_target,
              new_edge_ix=p_edge, target_coph=p_coph)
        # at:16     DROP TABLE IF EXISTS target_node_row;
        _step(con, f, "drop_target_node_row")
        # at:17-32  CREATE TEMP TABLE target_node_row AS WITH working_row AS (...) SELECT ...;
        _step(con, f, "create_target_node_row")
        # TreeDB CYC-1 (G3 D-024 Q7): the cycle guard, on BOTH paths (not part of D-016b).
        _cycle_guard(con, p_target, p_node)
        # TreeDB G3 D-024 Q4: check the registry, register a new item named by item_id (only
        # when one is given). Before the D-016b path, so the switch never changes the outcome
        # of a registry conflict; a later refusal rolls the registration back.
        new_items = _register_item(con, item_id, source_ref, p_node) if item_id is not None else ()
        # TreeDB D-016b (final, G6 D-037): the one isolated strict path.
        if strict:
            _d016b_strict_checks(con, p_target, p_node)
        # at:34-35  INSERT INTO graph.tree_branches (tree_id, branch_id) SELECT DISTINCT ...;
        _step(con, f, "insert_branch")
        # at:37-40  UPDATE graph.orchard SET node_id = (...new_branch_id), edge_ix = 2 WHERE id = (...);
        _step(con, f, "rewire_target")
        # at:42-45  INSERT INTO target_node_row SELECT ... FROM new_linkage nl, target_node_row tn;
        _step(con, f, "add_subject_row")
        # at:47-49  INSERT INTO graph.orchard (...) SELECT ... FROM target_node_row;
        # at:51-52  GET DIAGNOSTICS v_rowcount = ROW_COUNT; RETURN v_rowcount;
        v_rowcount = int(_step(con, f, "insert_rows").fetchone()[0])
        rep = _step(con, f, "report_new_branch").fetchone()
        tree_id, new_branch = (rep[0], rep[1]) if rep else (None, None)
        return {"rowcount": v_rowcount, "tree_id": tree_id, "new_branch_id": new_branch,
                "changed": tree_id is not None, "new_items": new_items}

    out = _run_in_transaction(con, body)
    result = AttachResult(
        rowcount=out["rowcount"], changed=out["changed"], target_orchard_id=p_target, new_node_id=p_node,
        new_branch_id=out["new_branch_id"], tree_id=out["tree_id"], tree_name=out.get("tree_name"),
        state_version=out.get("state_version"), strict=strict,
        orphaned_annotations=out["orphaned_annotations"],
        item_id=item_id, new_items=out["new_items"],
    )
    if result.new_items:
        logger.info("attach_node: registered new item(s) %s as %s in item_registry (D-024 Q4), same transaction",
                    list(result.new_items), result.new_node_id)
    if result.changed:
        logger.info("attach_node: tree %s (%s) state_version %s: node %s attached above orchard row %s "
                    "on new branch %s", result.tree_name, result.tree_id, result.state_version,
                    result.new_node_id, result.target_orchard_id, result.new_branch_id)
    else:
        logger.info("attach_node: orchard row %s not found; nothing changed (PG-faithful path, D-016b)",
                    result.target_orchard_id)
    _report_orphans("attach_node", result.tree_name, result.orphaned_annotations)
    return result


# ---------------------------------------------------------------------------
# detach_node
# ---------------------------------------------------------------------------

def detach_node(con: duckdb.DuckDBPyConnection, target_orchard_id: uuid.UUID | str | None) -> DetachResult:
    """Remove orchard row ``target_orchard_id`` together with its parent branch and
    rewire the grandparent row to the sibling, with the sibling's ``edge_ix`` (019).

    Same parameter and meaning as ``graph.orchard_detach_node`` (dt:1) and
    ``OrchardRepo.detach_node`` (repo.py:291-325). A detached branch stays in
    the tree without a parent (it can be re-attached by its id); a detached
    leaf leaves the tree.

    **D-016c.** When the parent branch has no parent itself (the target is a
    child of a root), PostgreSQL deletes that branch and rewires nothing: the
    tree splits. TreeDB does exactly that, and warns (``TreeSplitWarning``),
    logs a WARNING and sets ``split``. A sibling or target that is a leaf then
    leaves the tree. No new registry row is created: every part keeps the
    cultivar's ``tree_id`` (as in PG), and the cultivar's ``state_version``
    rises by one.

    **Cycle guard (CYC-3, G6 D-038, a divergence).** The rewire has no tree
    filter (GRAFT-1), so it can close a loop in another tree; PostgreSQL commits
    that. After the rewire, every tree whose rows the call changed (this tree
    and each collateral tree) is checked; a cycle raises ``CyclicDetachError``
    (no SQLSTATE) and the call rolls back, with no warning and no log entry.
    Every other detach, splits and acyclic collateral rewires included, is
    unchanged.

    Raises ``OrchardRowNotFoundError`` / ``NoSiblingError`` with PostgreSQL's
    messages (dt:17-19, dt:28-31), ``CyclicDetachError``, or another
    ``MutationError``; nothing is written on any failure.
    """
    p_target = _uuid_param(target_orchard_id, "target_orchard_id")

    def body() -> dict[str, Any]:
        f = "detach_node"
        # dt:12-15  SELECT branch_id, node_id INTO v_branch_id, v_target_node_id
        #           FROM graph.v_orchard WHERE orchard_id = p_target_orchard_id;
        row = _step(con, f, "select_target", target_orchard_id=p_target).fetchone()
        v_branch_id, v_target_node_id, tree_id = row if row is not None else (None, None, None)
        # dt:17-19  IF v_branch_id IS NULL THEN
        #               RAISE EXCEPTION 'No orchard row found for id %', p_target_orchard_id;
        #           END IF;
        if v_branch_id is None:
            raise OrchardRowNotFoundError(f"No orchard row found for id {_pg_text(p_target)}",
                                          target_orchard_id=p_target)
        # dt:21-26  SELECT node_id, edge_ix INTO v_sibling_node_id, v_sibling_edge_ix
        #           FROM graph.v_orchard WHERE branch_id = v_branch_id
        #           AND orchard_id <> p_target_orchard_id LIMIT 1;
        sib = _step(con, f, "select_sibling", v_branch_id=v_branch_id, target_orchard_id=p_target).fetchone()
        v_sibling_node_id, v_sibling_edge_ix = sib if sib is not None else (None, None)
        # dt:28-31  IF v_sibling_node_id IS NULL THEN
        #               RAISE EXCEPTION 'No sibling node found on branch_id % for orchard id %',
        #                   v_branch_id, p_target_orchard_id;
        #           END IF;
        if v_sibling_node_id is None:
            raise NoSiblingError(
                f"No sibling node found on branch_id {_pg_text(v_branch_id)} for orchard id {_pg_text(p_target)}",
                branch_id=v_branch_id, target_orchard_id=p_target)

        # TreeDB reporting (D-016c), read before any write.
        roots_before = tuple(r[0] for r in _step(con, "common", "tree_roots", tree_id=tree_id).fetchall())
        items_before = [r[0] for r in _step(con, "common", "tree_items", tree_id=tree_id).fetchall()]

        # dt:33-34  DELETE FROM graph.tree_branches WHERE branch_id = v_branch_id;
        # ... whose ON DELETE CASCADE (014:116) deletes the branch's orchard rows.
        # Explicit, children first, then the branch [DT-1, CASCADE-1].
        _step(con, f, "cascade_delete_orchard", v_branch_id=v_branch_id)
        # dt:36     GET DIAGNOSTICS v_rowcount = ROW_COUNT;  (the tree_branches DELETE)
        v_rowcount = int(_step(con, f, "delete_branch", v_branch_id=v_branch_id).fetchone()[0])
        # dt:38-41  UPDATE graph.orchard SET node_id = v_sibling_node_id,
        #           edge_ix = v_sibling_edge_ix WHERE node_id = v_branch_id;
        # (+ TreeDB RETURNING id, branch_id: which rows, in which trees [GRAFT-1])
        rewired = _step(con, f, "rewire_parent", v_sibling_node_id=v_sibling_node_id,
                        v_sibling_edge_ix=v_sibling_edge_ix, v_branch_id=v_branch_id).fetchall()
        # dt:43     RETURN v_rowcount;

        rewired_trees: tuple[uuid.UUID, ...] = ()
        if rewired:
            rewired_trees = tuple(r[0] for r in _step(
                con, "common", "trees_of_branches",
                branch_ids=sorted({r[1] for r in rewired}, key=str)).fetchall())
        # TreeDB CYC-3 (G6 D-038): every tree whose rows changed (this tree: its branch and
        # rows were deleted; each tree of a rewired row) must be acyclic; else raise and roll
        # back. Before the state_version bumps and before any report (all after COMMIT).
        _detach_cycle_guard(con, tuple(sorted({tree_id, *rewired_trees}, key=str)), p_target, v_branch_id,
                            frozenset(r[0] for r in rewired))
        roots_after = tuple(r[0] for r in _step(con, "common", "tree_roots", tree_id=tree_id).fetchall())
        items_after = {r[0] for r in _step(con, "common", "tree_items", tree_id=tree_id).fetchall()}
        return {
            "rowcount": v_rowcount, "tree_id": tree_id, "changed": True,
            "removed_branch_id": v_branch_id, "detached_node_id": v_target_node_id,
            "sibling_node_id": v_sibling_node_id, "sibling_edge_ix": int(v_sibling_edge_ix),
            "rewired_rows": len(rewired), "rewired_row_ids": tuple(sorted((r[0] for r in rewired), key=str)),
            "collateral_tree_ids": tuple(t for t in rewired_trees if t != tree_id),
            "roots_before": roots_before, "roots_after": roots_after,
            "dropped_item_ids": tuple(i for i in items_before if i not in items_after),
        }

    out = _run_in_transaction(con, body)
    new_roots = tuple(r for r in out["roots_after"] if r not in out["roots_before"])
    result = DetachResult(
        rowcount=out["rowcount"], target_orchard_id=p_target, tree_id=out["tree_id"],
        tree_name=out["tree_name"], state_version=out["state_version"],
        removed_branch_id=out["removed_branch_id"], detached_node_id=out["detached_node_id"],
        sibling_node_id=out["sibling_node_id"], sibling_edge_ix=out["sibling_edge_ix"],
        rewired_rows=out["rewired_rows"],
        # D-016c: the deleted branch had no parent row in its own tree. NOT "the rewire
        # updated 0 rows": a graft row of another tree can name a root (GRAFT-1, review M1).
        split=out["removed_branch_id"] in out["roots_before"],
        stored_root_deleted=out["removed_branch_id"] == out["stored_root_id"],
        roots_before=out["roots_before"], roots_after=out["roots_after"], new_root_ids=new_roots,
        detached_root_id=out["detached_node_id"] if out["detached_node_id"] in out["roots_after"] else None,
        dropped_item_ids=out["dropped_item_ids"], orphaned_annotations=out["orphaned_annotations"],
        rewired_row_ids=out["rewired_row_ids"], collateral_tree_ids=out["collateral_tree_ids"],
        collateral_state_versions=out["collateral_state_versions"],
    )
    logger.info("detach_node: tree %s (%s) state_version %s: orchard row %s detached, branch %s removed",
                result.tree_name, result.tree_id, result.state_version, result.target_orchard_id,
                result.removed_branch_id)
    if result.collateral_tree_ids:
        _report_collateral(result)
    if result.split:
        _announce_split(result)
    _report_orphans("detach_node", result.tree_name, result.orphaned_annotations)
    return result


def _report_collateral(r: DetachResult) -> None:
    """GRAFT-1: the dt:38-41 rewire (no tree filter, PG-faithful) changed rows of other trees."""
    logger.warning(
        "detach_node on cultivar %r (%s) also rewired graft row(s) of %s other tree(s) that named the deleted "
        "branch %s: %s; those rows now have node_id %s and edge_ix %s (an external_graft edge_ix 3 is "
        "overwritten, as in PostgreSQL dt:38-41). state_version bumped for those trees (D-007): %s. "
        "The cross-tree change is PostgreSQL behaviour, kept by G3 D-024 Q5 as amended by G6 D-038 "
        "(PG-faithful rewire; a rewire that would leave a tree cyclic is refused, CYC-3; "
        "INV-24: no mutation changes another cultivar).",
        r.tree_name, r.tree_id, len(r.collateral_tree_ids), r.removed_branch_id,
        [str(t) for t in r.collateral_tree_ids], r.sibling_node_id, r.sibling_edge_ix,
        {str(k): v for k, v in r.collateral_state_versions.items()},
    )


def _warn_every_time(message: str, category: type[Warning], depth: int) -> None:
    """``warnings.warn`` without Python's per-call-site dedup registry.

    ``warnings.warn`` records each (text, category, line) in the caller module's
    ``__warningregistry__`` and, under the default action, does not show the
    same text from the same line again. D-016c asks for every split to be
    loud, so the warning goes through ``warn_explicit`` with ``registry=None``
    (a fresh registry each time). The active filters still apply: ``ignore``,
    ``error`` and ``once`` set by the caller keep working, and no global filter
    is changed. ``depth`` is the frame, counted from the caller of this
    function, that the warning is attributed to (like ``stacklevel - 1``).
    """
    try:
        frame = sys._getframe(depth + 1)
    except ValueError:                                     # shallower stack than expected
        frame = sys._getframe(1)
    warnings.warn_explicit(message, category, frame.f_code.co_filename, frame.f_lineno,
                           module=frame.f_globals.get("__name__", "<unknown>"), registry=None,
                           module_globals=frame.f_globals)


def _announce_split(r: DetachResult) -> None:
    """D-016c: be loud (warning + log), on EVERY split. Called only after COMMIT."""
    msg = (
        f"tree split in cultivar {r.tree_name!r} ({r.tree_id}): detaching orchard row {r.target_orchard_id} "
        f"deleted branch {r.removed_branch_id}, which had no parent row in its own tree"
        f"{' (the stored import root, tree_registry.root_branch_id)' if r.stored_root_deleted else ''}. "
        f"Roots before: {[str(x) for x in r.roots_before]}; roots after: {[str(x) for x in r.roots_after]}; "
        f"new roots: {[str(x) for x in r.new_root_ids]}; items no longer in the cultivar: "
        f"{list(r.dropped_item_ids)}. All parts keep tree_id {r.tree_id}; state_version is now "
        f"{r.state_version}."
        + (f" The deleted branch was also named by graft row(s) of other tree(s) "
           f"{[str(t) for t in r.collateral_tree_ids]}, which now point at {r.sibling_node_id} (GRAFT-1)."
           if r.collateral_tree_ids else "")
        + " This is PostgreSQL's behaviour (D-016c)."
    )
    logger.warning(msg)
    # depth 2: _announce_split <- detach_node <- the caller of detach_node
    _warn_every_time(msg, TreeSplitWarning, depth=2)


def _report_orphans(op: str, tree_name: str | None, orphans: dict[str, list[tuple]]) -> None:
    if orphans:
        logger.warning("%s on %r left %s annotation row(s) without their node (kept, not deleted): %s",
                       op, tree_name, sum(len(v) for v in orphans.values()), sorted(orphans))


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def find_orchard_id(con: duckdb.DuckDBPyConnection, node_id: uuid.UUID | str,
                    tree_id: uuid.UUID | str) -> uuid.UUID | None:
    """Port of ``OrchardRepo.find_orchard_id`` (repo.py:331-351): the orchard row
    whose child is ``node_id`` in tree ``tree_id``, or None."""
    row = _step(con, "common", "find_orchard_id", node_id=_uuid_param(node_id, "node_id"),
                tree_id=_uuid_param(tree_id, "tree_id")).fetchone()
    return None if row is None else row[0]


def current_roots(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID | str) -> tuple[uuid.UUID, ...]:
    """Branches of the tree that have no parent row (one per component).

    After import this is ``(tree_registry.root_branch_id,)``. A detached branch
    adds a root; a D-016c split removes the old one. ``root_branch_id`` itself
    is never changed by a mutation: it stays the import root (P3C ROOT-2).
    """
    return tuple(r[0] for r in _step(con, "common", "tree_roots",
                                     tree_id=_uuid_param(tree_id, "tree_id")).fetchall())


__all__ = [
    "ANNOTATION_CHECKS", "AttachResult", "CyclicDetachError", "D016B_STRICT_ATTACH", "DetachResult",
    "TREEDB_ONLY_CHECKS",
    "attach_node", "current_roots", "detach_node", "find_orchard_id",
]
