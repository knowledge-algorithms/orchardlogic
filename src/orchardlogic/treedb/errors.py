"""Exceptions and warnings of TreeDB structural mutation (phase P3C).

Each exception records the PostgreSQL behaviour it corresponds to: the
SQLSTATE (``sqlstate``) and, for ``RAISE EXCEPTION``, the exact message
format of the installed function (``pg_message``). ``sqlstate`` is ``None``
when PostgreSQL raises nothing in that situation (a TreeDB-only rule); the
docstring then says what PostgreSQL does instead.

Installed texts cited (D-002, D-009):

* ``dt:<line>`` = ``reference/pg_installed/graph/function/orchard_detach_node_uuid.sql``
* ``at:<line>`` = ``reference/pg_installed/graph/function/orchard_attach_node_uuid_uuid_integer_numeric.sql``
* ``pgi:table/orchard.sql`` = ``reference/pg_installed/graph/table/orchard.sql``

The mapping table is also in ``docs/porting/P3C.md`` (ERR-1).
"""

from __future__ import annotations

from typing import Any


class TreeDBError(Exception):
    """Base class of TreeDB errors raised by ``treedb.mutate``."""


class MutationError(TreeDBError):
    """A mutation failed and was rolled back; nothing was written.

    ``sqlstate``: the PostgreSQL SQLSTATE of the equivalent failure, or
    ``None`` when PostgreSQL does not fail. ``pg_message``: PostgreSQL's
    message for the same input, when it raises one.
    """

    sqlstate: str | None = None

    def __init__(self, message: str, *, pg_message: str | None = None, **details: Any):
        super().__init__(message)
        self.pg_message = pg_message
        self.details = details


class TransactionStateError(MutationError):
    """The connection is already inside a transaction.

    No PostgreSQL counterpart: plpgsql runs inside the caller's transaction,
    while each TreeDB mutation opens and commits its own (P3C TX-1), as
    ``repo.py`` does by committing after every call (repo.py:268-278, 314-318).
    """


class InvalidParameterError(MutationError, ValueError):
    """A parameter cannot be converted to its SQL type (for example a bad UUID).

    PostgreSQL: SQLSTATE ``22P02`` (invalid_text_representation) when the
    text bound for a ``uuid`` / ``integer`` parameter does not parse.
    """

    sqlstate = "22P02"


class OrchardRowNotFoundError(MutationError, LookupError):
    """``detach_node``: no orchard row has this id (also the root, which has none).

    PostgreSQL: ``RAISE EXCEPTION 'No orchard row found for id %'`` (dt:17-19),
    SQLSTATE ``P0001`` (raise_exception). ``str(exc)`` is that exact message.
    """

    sqlstate = "P0001"


class NoSiblingError(MutationError):
    """``detach_node``: the target's branch has no other child row.

    PostgreSQL: ``RAISE EXCEPTION 'No sibling node found on branch_id % for
    orchard id %'`` (dt:28-31), SQLSTATE ``P0001``. ``str(exc)`` is that exact
    message.
    """

    sqlstate = "P0001"


class UnknownAttachTargetError(MutationError, LookupError):
    """``attach_node`` (D-016b strict path): the target orchard row does not exist.

    **PostgreSQL raises nothing**: the function returns 0 and changes nothing
    (at:17-49 join an empty working row; PG_BASELINE §8,
    ``reference/golden/*/*.invalid_attach_unknown_edge.json``). TreeDB raises
    (D-016b, final since G6 D-037; ``strict=False`` is the PostgreSQL reproducer)
    and reports the unknown id (D-016d, INV-20).
    """

    sqlstate = None


class DuplicateAttachError(MutationError):
    """``attach_node`` (D-016b strict path): the subject already has a parent row
    in the target's tree (an item would appear twice in one tree).

    **PostgreSQL raises nothing**: the subject gets a second parent and the
    function returns 2 (``UNIQUE (branch_id, node_id)`` is the only key,
    pgi:table/orchard.sql:8; PG_BASELINE §8,
    ``reference/golden/*/*.invalid_attach_duplicate_item.json``). TreeDB raises
    (D-016b, final since G6 D-037; ``strict=False`` is the PostgreSQL reproducer).
    """

    sqlstate = None


class CyclicDetachError(MutationError):
    """``detach_node``: after the dt:38-41 rewire, a tree whose rows the call
    changed (the detached tree, or a collateral tree of the GRAFT-1 rewire) would
    hold a cycle (a node its own ancestor). TreeDB guard CYC-3 (G6 D-038, amending
    G3 D-024 Q5; a divergence), run on every detach whatever the D-016b setting.

    **PostgreSQL raises nothing**: detach has no ancestry check (dt:5-44) and its
    rewire has no tree filter, so it commits the cycle and returns the deleted
    branch count. TreeDB refuses and rolls back (no row and no ``state_version``
    change; no split or collateral warning and no log entry), so
    ``sqlstate = None``. ``details``: ``target_orchard_id``, ``removed_branch_id``,
    ``tree_ids`` (the cyclic trees, in name order), ``tree_names``, ``rows`` (each
    row on a cycle as ``(tree_id, orchard_id, branch_id, node_id, edge_ix)``).

    Re-exported by ``treedb.mutate``.
    """

    sqlstate = None


class UniqueViolationError(MutationError):
    """A UNIQUE or PRIMARY KEY constraint rejected a row.

    PostgreSQL: SQLSTATE ``23505`` (unique_violation), for example
    ``orchard_branch_node_key`` when one new branch would get the same child
    twice (attach of a node above its own row, PG-faithful path).
    """

    sqlstate = "23505"


class NotNullViolationError(MutationError):
    """A NOT NULL column received NULL (for example a NULL subject node id).

    PostgreSQL: SQLSTATE ``23502`` (not_null_violation).
    """

    sqlstate = "23502"


class IntegrityViolationError(MutationError):
    """The mutation would leave a reference that a PostgreSQL foreign key rejects
    (``treedb.store.integrity_violations``, PORTING_NOTES FK-1; P3C INT-1).
    ``details["violations"]`` lists the new offending rows per check.

    PostgreSQL: SQLSTATE ``23503`` (foreign_key_violation): at least one of the
    new violations is of a PG foreign-key equivalent, so PostgreSQL would fail
    too. When the only new violations are of TreeDB-only rules,
    ``UnregisteredItemError`` (``sqlstate = None``) is raised instead.
    """

    sqlstate = "23503"


class UnregisteredItemError(IntegrityViolationError):
    """``attach_node``: a leaf row (``edge_ix`` 1) would name a uuid that is not in
    ``treedb.item_registry`` (the TreeDB-only rule of P3C INT-1), and the call gave
    no ``item_id`` to register it under.

    **PostgreSQL raises nothing**: ``orchard.node_id`` has no foreign key
    (pgi:table/orchard.sql), so PG inserts the row and returns 2. TreeDB refuses
    it, rolls back, and carries ``sqlstate = None`` so that no caller maps it to
    a PostgreSQL failure. Asymmetry (documented, INT-1): an ``edge_ix`` 2 or 3
    row naming a uuid that is not a branch is **not** refused (no such rule, as
    in PG). G3 (D-024 Q4): a leaf attach that passes the item's external
    ``item_id`` registers a new item in the same transaction instead; a bare
    unknown UUID is still refused with this error.
    """

    sqlstate = None


class ItemRegistrationError(MutationError, ValueError):
    """``attach_node`` with ``item_id`` / ``source_ref`` (D-024 Q4): the request
    cannot register or use the item as asked (for example ``item_id`` on an
    ``edge_ix`` 2/3 attach, an empty ``item_id``, or ``source_ref`` without
    ``item_id``). Raised before or inside the transaction; nothing is written.

    **No PostgreSQL counterpart** (PG has no item registry; P3C INT-1), so
    ``sqlstate = None``.
    """

    sqlstate = None


class ItemConflictError(ItemRegistrationError):
    """``attach_node`` with ``item_id`` (D-024 Q4): the item registry already holds
    a different identity for this request: the ``item_id`` is registered under
    another UUID, the UUID is registered under another ``item_id``, the UUID is
    not the item's CONTRACTS UUID (INV-5, which the import enforces too), or a
    given ``source_ref`` differs from the stored one. Rolled back; nothing is
    written. ``details`` names the stored row.

    **No PostgreSQL counterpart**, so ``sqlstate = None``.
    """

    sqlstate = None


class CyclicAttachError(MutationError):
    """``attach_node``: the subject is the target row's parent branch or one of
    its ancestors in the target's tree, so the attach would make a node its own
    ancestor (a cycle). TreeDB guard CYC-1 (D-024 Q7, a divergence), applied on
    the strict and on the ``strict=False`` path alike (not part of D-016b).

    **PostgreSQL raises nothing**: attach has no ancestry check (at:8-52), it
    commits the cycle and returns 2; the tree is left without a root and a
    recursive read over it does not terminate. TreeDB refuses and rolls back
    (no row and no ``state_version`` change), so ``sqlstate = None``.
    ``details``: ``new_node_id``, ``target_orchard_id``, ``tree_id``.
    """

    sqlstate = None


class TreeSplitWarning(UserWarning):
    """``detach_node`` deleted a branch that had no parent: the tree split (D-016c).

    Faithful to PostgreSQL, which raises nothing (PG_BASELINE §8,
    ``reference/golden/*/p2b.*.detach_root_child.json``). TreeDB is loud about
    it: this warning, a ``logging`` WARNING on logger ``treedb.mutate`` and
    ``DetachResult.split``. Every split is warned, also repeated identical
    messages from one call site (``treedb.mutate`` bypasses Python's
    per-call-site registry; the active warning filters still apply, so a
    caller can ``ignore`` or ``error`` this category as usual).
    """


__all__ = [
    "CyclicAttachError", "CyclicDetachError", "DuplicateAttachError", "IntegrityViolationError", "InvalidParameterError",
    "ItemConflictError", "ItemRegistrationError", "MutationError",
    "NoSiblingError", "NotNullViolationError", "OrchardRowNotFoundError", "TransactionStateError",
    "TreeDBError", "TreeSplitWarning", "UniqueViolationError", "UnknownAttachTargetError",
    "UnregisteredItemError",
]
