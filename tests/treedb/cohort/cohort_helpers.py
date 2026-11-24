"""Helpers for tests/cohort (P4A). Not a conftest."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from orchardlogic.treedb import cohort, ingest, store

REPO = Path(__file__).resolve().parents[1]
TWOCULT8_TOPOLOGY = REPO / "fixtures" / "query_cases" / "twocult8_topology.json"


def new_store() -> duckdb.DuckDBPyConnection:
    con = store.connect()
    cohort.install(con)
    return con


def twocult8_topology() -> dict:
    return json.loads(TWOCULT8_TOPOLOGY.read_text(encoding="utf-8"))


def topology_to_z(items: list[str], nodes: dict[str, list[str]], root: str):
    """SciPy-style Z for a nested topology. Heights are strictly increasing in creation order."""
    index = {it: i for i, it in enumerate(items)}
    z: list[list[float]] = []
    counts: dict[int, int] = {i: 1 for i in range(len(items))}

    def build(name: str) -> int:
        if name in index:
            return index[name]
        l, r = (build(c) for c in nodes[name])
        l, r = min(l, r), max(l, r)
        idx = len(items) + len(z)
        counts[idx] = counts[l] + counts[r]
        z.append([l, r, 0.1 * (len(z) + 1), counts[idx]])
        return idx

    build(root)
    return z


def import_twocult8(con: duckdb.DuckDBPyConnection) -> None:
    """Import P5A's inline ``twocult8`` (items i1..i8, cultivars A and B)."""
    topo = twocult8_topology()
    for name, cv in topo["cultivars"].items():
        z = topology_to_z(topo["items"], cv["nodes"], cv["root"])
        ingest.import_cultivar(con, z, topo["items"], name)
