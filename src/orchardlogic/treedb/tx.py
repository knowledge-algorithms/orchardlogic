"""Nestable transactions for the v4 writers (docs/DATA_ORCHARD_V4.md, amendment A3).

The DuckDB Python connection cannot report whether it is inside a transaction and a
failed nested ``BEGIN`` poisons the outer one. ``transaction(con)`` tracks the nesting
depth per connection; only the outermost level issues BEGIN / COMMIT / ROLLBACK.

There are no savepoints: an exception in an inner level cannot be undone on its own. It propagates,
and if the caller catches it inside the outer block the outer transaction continues with the inner
level's writes in place (nothing is rolled back until the outermost level exits with an error).

``ingest.import_cultivar``, ``mutate.*`` and ``store.install_schema`` open their own
transaction and must only be called at depth 0 (``depth(con) == 0``).
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from contextlib import contextmanager

import duckdb

_DEPTH: "weakref.WeakKeyDictionary[duckdb.DuckDBPyConnection, int]" = weakref.WeakKeyDictionary()


def depth(con: duckdb.DuckDBPyConnection) -> int:
    """Current ``transaction`` nesting depth of ``con`` (0 = none opened through this module)."""
    return _DEPTH.get(con, 0)


@contextmanager
def transaction(con: duckdb.DuckDBPyConnection) -> Iterator[duckdb.DuckDBPyConnection]:
    """Run the block in one transaction; nested uses join the outermost one.

    An exception at any level propagates; the outermost level rolls everything back.
    """
    level = _DEPTH.get(con, 0)
    if level == 0:
        con.execute("BEGIN TRANSACTION")
    _DEPTH[con] = level + 1
    try:
        yield con
    except BaseException:
        _DEPTH[con] = level
        if level == 0:
            con.execute("ROLLBACK")
        raise
    else:
        _DEPTH[con] = level
        if level == 0:
            try:
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise


__all__ = ["depth", "transaction"]
