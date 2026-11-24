"""Helpers for tests/query (P5B). Not a conftest."""

from __future__ import annotations

import functools
import json
import random
from pathlib import Path

import duckdb

from conftest import load_named
from orchardlogic.treedb import cohort, ingest, store

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "fixtures" / "query_cases"
LABEL_SET = "p5a_nodes"


def new_store() -> duckdb.DuckDBPyConnection:
    con = store.connect()
    cohort.install(con)
    return con


@functools.lru_cache(maxsize=None)
def shared_store(stem: str) -> duckdb.DuckDBPyConnection:
    """One store per fixture for the whole session. Queries only add cohorts (never change cultivars)."""
    return fixture_store(stem)


def fixture_store(stem: str) -> duckdb.DuckDBPyConnection:
    con = new_store()
    fx = load_named(stem)
    for cv in fx["cultivars"]:
        ingest.import_fixture_cultivar(con, fx, cv["name"])
    return con


def node_by_label(con, cultivar: str, label: str) -> str:
    row = con.execute(
        "SELECT l.node_id FROM treedb.node_labels l JOIN treedb.tree_registry t ON t.id = l.tree_id "
        "WHERE t.name = ? AND l.label_set = ? AND l.label = ?", [cultivar, LABEL_SET, label]).fetchone()
    return str(row[0])


def case_files() -> list[Path]:
    return sorted(CASE_DIR.glob("Q*.json"))


def load_case(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def translate(con, query: dict) -> dict:
    """Case query -> treedb.query/1 JSON: drop ``base``, map node labels to branch ids, set the version.

    Rules, ``any_of`` and every other field are otherwise passed through, so the
    errors a case expects come from the library, not from this helper.
    """
    out = {k: v for k, v in query.items() if k not in ("base", "version")}

    def rule(r):
        if "node" in r:
            return {**r, "node": node_by_label(con, r["cultivar"], r["node"])}
        return dict(r)

    if "rules" in out:
        out["rules"] = [rule(r) for r in out["rules"]]
    return out


def branch_members(con, cultivar: str) -> dict[str, frozenset[str]]:
    """{branch id text: items below it} for one cultivar, from plain row walks."""
    tid = con.execute("SELECT id FROM treedb.tree_registry WHERE name = ?", [cultivar]).fetchone()[0]
    rows = con.execute(
        "SELECT o.branch_id, o.node_id, r.item_id FROM treedb.v_orchard o "
        "LEFT JOIN treedb.item_registry r ON r.item_uuid = o.node_id WHERE o.tree_id = ?", [tid]).fetchall()
    kids: dict = {}
    for b, n, item in rows:
        kids.setdefault(b, []).append((n, item))

    memo: dict = {}

    def members(b) -> frozenset[str]:
        if b not in memo:
            s: set[str] = set()
            for n, item in kids[b]:
                s |= {item} if item is not None else members(n)
            memo[b] = frozenset(s)
        return memo[b]

    return {str(b): members(b) for b in kids}


def sorted_nodes(con, cultivar: str) -> list[str]:
    """Branch ids ordered by their item sets, so seeded picks do not depend on random UUIDs."""
    bm = branch_members(con, cultivar)
    return sorted(bm, key=lambda b: (sorted(bm[b]), len(bm[b])))


def random_query(con, rng: random.Random, cultivars: list[str], items: list[str], *, max_rules: int = 6) -> dict:
    """A conflict-free random query (JSON), reproducible for a seeded ``rng``."""
    nodes = {c: sorted_nodes(con, c) for c in cultivars}
    rules, used_nodes, used_items = [], set(), set()
    for _ in range(rng.randint(0, max_rules)):
        op = rng.choice(["include", "exclude"])
        if rng.random() < 0.25:
            item = rng.choice(items)
            if item in used_items:
                continue
            used_items.add(item)
            rules.append({"item": item, "op": op})
        else:
            c = rng.choice(cultivars)
            n = rng.choice(nodes[c])
            if (c, n) in used_nodes:
                continue
            used_nodes.add((c, n))
            rules.append({"cultivar": c, "node": n, "op": op})
    return {"version": "treedb.query/1", "rules": rules}

