"""Cohorts: immutable item-ID sets with a handle, and their algebra (phase P4A).

A cohort is a bounded set of shared leaf ``item_id`` values (INV-1) plus an
opaque handle (INV-17). It is not a cultivar. Membership never changes after
creation; every operation produces a **new** cohort that records its kind,
operands and the cultivar ``state_version`` values at creation (INV-18, D-007).

* ``install(con)`` creates the tables and SQL macros (idempotent). It does not
  call ``store.install_schema``; the items registry must already exist.
* ``all_items`` / ``from_ids`` create root cohorts. ``from_ids`` raises
  ``UnknownItemsError`` listing every unknown id (INV-20, D-016d); ids are
  never dropped silently. Duplicate ids in the input are collapsed (a cohort is
  a set); that is not a drop of any item.
* ``union`` / ``intersect`` / ``difference`` record kind ``union`` /
  ``intersect`` / ``difference`` and the operand handles. The base is the left
  operand. ``difference`` and ``intersect`` results are subsets of the base
  (INV-19). A ``union`` whose other operands are not subsets of the base is
  *explicit broadening*: its ``operands["broadens_base"]`` is true and
  ``Cohort.broadens_base`` reports it. It can never be confused with an
  ``execute`` result, which ``create`` only accepts as a subset of its base.
* ``create`` is the single insertion path, also for P5B (``execute``) and P6B
  (``external``).
* ``items`` returns the sorted members (or a DuckDB relation); ``describe``
  returns provenance plus how far each cultivar has moved since creation;
  ``get`` returns the header row.

Cohorts are never updated or deleted by this module (no such function exists).
In SQL: ``SELECT * FROM treedb.cohort_items('<cohort id>')``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Iterable, Sequence

import duckdb

KINDS = ("all_items", "from_ids", "union", "intersect", "difference", "execute", "external")


class CohortError(ValueError):
    """Invalid cohort operation."""


class UnknownCohortError(KeyError):
    """No cohort has this handle."""


class UnknownItemsError(CohortError):
    """Item ids are not in ``treedb.item_registry`` (INV-20: never dropped silently)."""

    def __init__(self, unknown: Sequence[str]):
        self.unknown = sorted(unknown)
        shown = ", ".join(repr(i) for i in self.unknown[:20])
        more = f" (and {len(self.unknown) - 20} more)" if len(self.unknown) > 20 else ""
        super().__init__(f"{len(self.unknown)} unknown item id(s): {shown}{more}")


@dataclass(frozen=True)
class Cohort:
    """Header row of ``treedb.cohort`` (members are read with ``items``)."""

    cohort_id: uuid.UUID
    membership_hash: str
    size: int
    kind: str
    base_cohort_id: uuid.UUID | None
    operands: dict[str, Any]
    expression: Any
    cultivar_state_versions: dict[str, dict[str, Any]]
    created_seq: int = field(compare=False)

    @property
    def broadens_base(self) -> bool:
        """True for an explicit ``union`` that adds items outside its base."""
        return self.kind == "union" and bool(self.operands.get("broadens_base"))


CohortRef = Cohort | uuid.UUID | str


def install(con: duckdb.DuckDBPyConnection) -> None:
    """Create the cohort tables and macros. Idempotent."""
    sql = resources.files("orchardlogic.treedb").joinpath("sql/cohort.sql").read_text(encoding="utf-8")
    con.execute(sql)


def membership_hash(item_ids: Iterable[str]) -> str:
    """SHA-256 hex of the sorted distinct ids as a compact JSON array.

    JSON framing keeps the hash unambiguous for any id text. Sorting is by
    code point, which equals DuckDB's byte order on UTF-8 text.
    """
    body = json.dumps(sorted(set(item_ids)), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _in_transaction(con: duckdb.DuckDBPyConnection) -> bool:
    """True inside an explicit transaction: autocommit gives each statement a new id."""
    q = "SELECT current_transaction_id()"
    return con.execute(q).fetchone()[0] == con.execute(q).fetchone()[0]


def _cid(ref: CohortRef) -> uuid.UUID:
    if isinstance(ref, Cohort):
        return ref.cohort_id
    if isinstance(ref, uuid.UUID):
        return ref
    return uuid.UUID(str(ref))


def _state_versions(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, Any]]:
    rows = con.execute("SELECT id, name, state_version FROM treedb.tree_registry ORDER BY name").fetchall()
    return {str(i): {"name": n, "state_version": int(v)} for i, n, v in rows}


def _row_to_cohort(row: tuple) -> Cohort:
    cid, h, size, kind, base, operands, expression, versions, seq = row
    return Cohort(
        cohort_id=cid,
        membership_hash=h,
        size=int(size),
        kind=kind,
        base_cohort_id=base,
        operands=json.loads(operands),
        expression=None if expression is None else json.loads(expression),
        cultivar_state_versions=json.loads(versions),
        created_seq=int(seq),
    )


_COLS = (
    "cohort_id, membership_hash, size, kind, base_cohort_id, "
    "operands::VARCHAR, expression::VARCHAR, cultivar_state_versions::VARCHAR, created_seq"
)


def get(con: duckdb.DuckDBPyConnection, cohort_id: CohortRef) -> Cohort:
    """The cohort with this handle. Raises ``UnknownCohortError``."""
    cid = _cid(cohort_id)
    rows = con.execute(f"SELECT {_COLS} FROM treedb.cohort WHERE cohort_id = ?", [cid]).fetchall()
    if not rows:
        raise UnknownCohortError(str(cid))
    return _row_to_cohort(rows[0])


def items(con: duckdb.DuckDBPyConnection, cohort: CohortRef, *, relation: bool = False):
    """Sorted member ids, or (``relation=True``) a DuckDB relation ``(item_id)``."""
    cid = _cid(cohort)
    get(con, cid)  # raise for an unknown handle rather than return an empty list
    if relation:
        return con.sql(f"SELECT item_id FROM treedb.cohort_items('{cid}') ORDER BY item_id")
    rows = con.execute("SELECT item_id FROM treedb.cohort_member WHERE cohort_id = ?", [cid]).fetchall()
    return sorted(r[0] for r in rows)


def is_subset(con: duckdb.DuckDBPyConnection, a: CohortRef, b: CohortRef) -> bool:
    """True when every member of ``a`` is a member of ``b``."""
    return set(items(con, a)) <= set(items(con, b))


def _check_known(con: duckdb.DuckDBPyConnection, ids: Sequence[str]) -> None:
    known = {
        r[0]
        for r in con.execute(
            "SELECT item_id FROM treedb.item_registry WHERE item_id IN (SELECT unnest(?::VARCHAR[]))",
            [list(ids)],
        ).fetchall()
    }
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise UnknownItemsError(unknown)


def create(
    con: duckdb.DuckDBPyConnection,
    item_ids: Iterable[str],
    kind: str,
    *,
    base: CohortRef | None = None,
    operands: dict[str, Any] | None = None,
    expression: Any = None,
) -> Cohort:
    """Insert a new cohort with a fresh handle and return it.

    The one insertion path. Checks: ``kind`` is known; every id is a string
    in the items registry (else ``UnknownItemsError``); an ``execute`` cohort
    has a base and is a subset of it (INV-19). Handles are random (uuid4), so
    equal membership gives equal ``membership_hash`` and distinct handles.
    """
    if kind not in KINDS:
        raise CohortError(f"unknown cohort kind {kind!r}; expected one of {KINDS}")
    ids = list(item_ids)
    bad = [i for i in ids if not isinstance(i, str)]
    if bad:
        raise TypeError(f"item ids must be str, got {bad[:5]!r}")
    members = sorted(set(ids))
    _check_known(con, members)

    base_id = None if base is None else _cid(base)
    if base_id is not None:
        base_members = set(items(con, base_id))  # also raises for an unknown base
    if kind == "execute":
        if base_id is None:
            raise CohortError("an execute cohort needs a base cohort (INV-19)")
        outside = [i for i in members if i not in base_members]
        if outside:
            raise CohortError(f"execute result is not a subset of its base; outside base: {outside[:20]}")

    cid = uuid.uuid4()
    own = not _in_transaction(con)  # inside the caller's transaction, it owns commit and rollback
    if own:
        con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "INSERT INTO treedb.cohort (cohort_id, membership_hash, size, kind, base_cohort_id, "
            "operands, expression, cultivar_state_versions) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                cid,
                membership_hash(members),
                len(members),
                kind,
                base_id,
                json.dumps(operands or {}, sort_keys=True),
                None if expression is None else json.dumps(expression, sort_keys=True),
                json.dumps(_state_versions(con), sort_keys=True),
            ],
        )
        con.execute(
            "INSERT INTO treedb.cohort_member (cohort_id, item_id) SELECT ?, unnest(?::VARCHAR[])",
            [cid, members],
        )
        if own:
            con.execute("COMMIT")
    except BaseException:
        if own:
            con.execute("ROLLBACK")
        raise
    return get(con, cid)


def all_items(con: duckdb.DuckDBPyConnection) -> Cohort:
    """Every registered item. Same membership whichever cultivar is shown."""
    ids = [r[0] for r in con.execute("SELECT item_id FROM treedb.item_registry").fetchall()]
    return create(con, ids, "all_items", operands={"source": "item_registry"})


def from_ids(con: duckdb.DuckDBPyConnection, ids: Iterable[str], *, on_unknown: str = "error") -> Cohort:
    """Cohort of the given ids. Unknown ids raise ``UnknownItemsError`` (listing them).

    ``on_unknown`` accepts only ``"error"``: silently dropping ids is forbidden
    (INV-20), and TreeDB offers no other policy.
    """
    if on_unknown != "error":
        raise CohortError(f"on_unknown={on_unknown!r} is not supported; unknown ids always raise (INV-20)")
    ids = list(ids)
    return create(con, ids, "from_ids", operands={"requested": len(ids), "distinct": len(set(ids))})


def _operands(con: duckdb.DuckDBPyConnection, refs: Sequence[CohortRef]) -> list[tuple[uuid.UUID, set[str]]]:
    return [(_cid(r), set(items(con, r))) for r in refs]


def union(con: duckdb.DuckDBPyConnection, left: CohortRef, right: CohortRef, *more: CohortRef) -> Cohort:
    """Union; the base is ``left``. Explicit broadening when the others reach outside it."""
    ops = _operands(con, (left, right, *more))
    base = ops[0][1]
    result = set().union(*(m for _, m in ops))
    added = len(result - base)
    return create(
        con, result, "union", base=ops[0][0],
        operands={"cohorts": [str(c) for c, _ in ops], "broadens_base": added > 0, "added_outside_base": added},
    )


def intersect(con: duckdb.DuckDBPyConnection, left: CohortRef, right: CohortRef, *more: CohortRef) -> Cohort:
    """Intersection; the base is ``left`` and the result is a subset of it."""
    ops = _operands(con, (left, right, *more))
    result = set.intersection(*(m for _, m in ops))
    return create(con, result, "intersect", base=ops[0][0], operands={"cohorts": [str(c) for c, _ in ops]})


def difference(con: duckdb.DuckDBPyConnection, left: CohortRef, right: CohortRef) -> Cohort:
    """``left`` minus ``right``; the base is ``left`` and the result is a subset of it."""
    (lid, lm), (rid, rm) = _operands(con, (left, right))
    return create(con, lm - rm, "difference", base=lid, operands={"cohorts": [str(lid), str(rid)]})


def describe(con: duckdb.DuckDBPyConnection, cohort: CohortRef) -> dict[str, Any]:
    """Provenance of a cohort (INV-18), with how each cultivar has moved since.

    ``cultivar_state_now`` is the live ``state_version`` per cultivar;
    ``moved_cultivars`` lists cultivars whose version differs from the recorded
    one (or that did not exist at creation, or no longer exist). Membership is
    not affected; this is evidence for OD-4.
    """
    c = get(con, cohort)
    now = _state_versions(con)
    moved = []
    for tid in sorted(set(now) | set(c.cultivar_state_versions)):
        rec = c.cultivar_state_versions.get(tid)
        cur = now.get(tid)
        if rec is None or cur is None or rec["state_version"] != cur["state_version"]:
            moved.append(tid)
    return {
        "cohort_id": str(c.cohort_id),
        "kind": c.kind,
        "size": c.size,
        "membership_hash": c.membership_hash,
        "base_cohort_id": None if c.base_cohort_id is None else str(c.base_cohort_id),
        "operands": c.operands,
        "expression": c.expression,
        "broadens_base": c.broadens_base,
        "created_seq": c.created_seq,
        "cultivar_state_versions": c.cultivar_state_versions,
        "cultivar_state_now": now,
        "moved_cultivars": moved,
    }


def integrity_violations(con: duckdb.DuckDBPyConnection) -> dict[str, list[tuple]]:
    """Undeclared references of the cohort tables (no FKs; see cohort.sql). Empty when all hold."""
    queries = {
        "cohort_member.cohort_id -> cohort": (
            "SELECT m.cohort_id, m.item_id FROM treedb.cohort_member m "
            "LEFT JOIN treedb.cohort c USING (cohort_id) WHERE c.cohort_id IS NULL"
        ),
        "cohort_member.item_id -> item_registry": (
            "SELECT m.cohort_id, m.item_id FROM treedb.cohort_member m "
            "LEFT JOIN treedb.item_registry i USING (item_id) WHERE i.item_id IS NULL"
        ),
        "cohort.base_cohort_id -> cohort": (
            "SELECT c.cohort_id FROM treedb.cohort c LEFT JOIN treedb.cohort b ON b.cohort_id = c.base_cohort_id "
            "WHERE c.base_cohort_id IS NOT NULL AND b.cohort_id IS NULL"
        ),
        "cohort.size / membership count": (
            "SELECT c.cohort_id FROM treedb.cohort c WHERE c.size <> "
            "(SELECT count(*) FROM treedb.cohort_member m WHERE m.cohort_id = c.cohort_id)"
        ),
    }
    out = {}
    for name, sql in queries.items():
        rows = con.execute(sql).fetchall()
        if rows:
            out[name] = rows
    return out
