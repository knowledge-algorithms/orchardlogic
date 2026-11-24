"""Shared helpers for tests/ingest (P3A). Not a conftest: P1 tests import
``conftest`` by module name, so a second conftest module could shadow it."""

from __future__ import annotations

import uuid
from typing import Any

import duckdb

from conftest import fixture_paths
from orchardlogic.treedb import canonical, ingest, store
from orchardlogic.treedb.fixtures import cultivar_item_ids, load_fixture

FIXTURE_PATHS = fixture_paths()
FIXTURES = {p.stem: load_fixture(p) for p in FIXTURE_PATHS}

#: (fixture stem, cultivar name) for every cultivar of every fixture (INV-26 set).
CULTIVARS = [(stem, cv["name"]) for stem, fx in FIXTURES.items() for cv in fx["cultivars"]]


def cultivar(stem: str, name: str) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    fx = FIXTURES[stem]
    cv = next(c for c in fx["cultivars"] if c["name"] == name)
    return fx, cv, cultivar_item_ids(fx, cv)


def load_all(stem: str, con: duckdb.DuckDBPyConnection | None = None):
    """Import every cultivar of a fixture into ``con`` (a new in-memory store by default)."""
    con = store.connect() if con is None else con
    fx = FIXTURES[stem]
    results = {cv["name"]: ingest.import_fixture_cultivar(con, fx, cv["name"]) for cv in fx["cultivars"]}
    return con, results


def snapshot_all(con: duckdb.DuckDBPyConnection, names) -> dict[str, Any]:
    return {n: canonical.canonical_snapshot(con, n) for n in names}


def tree_rows(con, tree_id: uuid.UUID):
    """(branch_id, node_id, edge_ix, coph_distance) of one tree's orchard rows."""
    return con.execute(
        "SELECT o.branch_id, o.node_id, o.edge_ix, o.coph_distance FROM treedb.orchard o "
        "JOIN treedb.tree_branches b USING (branch_id) WHERE b.tree_id = ?",
        [tree_id],
    ).fetchall()
