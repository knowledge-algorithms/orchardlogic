"""Root identity is stable (P3A acceptance criterion 3; PORTING_NOTES ROOT-1, OD-3).

PostgreSQL generates the root doc_id of ``graph.v_tree_map`` with VOLATILE
``gen_random_uuid()`` (reference/pg_installed/graph/view/v_tree_map.sql:33).
TreeDB stores it once at import (``tree_registry.root_branch_id``). These tests
read it repeatedly and also reproduce, on DuckDB, why the literal port is not
stable. They do not settle OD-3 for PostgreSQL (P2B measures that).
"""

from __future__ import annotations

import pytest

from ingest_helpers import FIXTURES, load_all
from orchardlogic.treedb import ingest, store

ROOT_SQL = "SELECT doc_id FROM treedb.v_tree_map WHERE tree_id = ? AND b_side = 'root'"


@pytest.mark.parametrize("stem", sorted(FIXTURES))
def test_root_id_read_twice_is_identical(stem):
    con, results = load_all(stem)
    for res in results.values():
        first = con.execute(ROOT_SQL, [res.tree_id]).fetchall()
        second = con.execute(ROOT_SQL, [res.tree_id]).fetchall()
        assert first == second == [(res.root_branch_id,)]
        # both references of one query agree too (create_orchard joins v_tree_map twice, co:19-23)
        same = con.execute(
            "SELECT count(*), count(DISTINCT a.doc_id), bool_and(a.doc_id = b.doc_id) "
            "FROM treedb.v_tree_map a JOIN treedb.v_tree_map b USING (tree_id, b_side) "
            "WHERE a.tree_id = ? AND a.b_side = 'root'", [res.tree_id]).fetchone()
        assert same == (1, 1, True)


def test_root_id_stable_across_reopen(tmp_path):
    path = tmp_path / "root.duckdb"
    con, results = load_all("F2_balanced8", store.connect(path))
    before = {n: con.execute(ROOT_SQL, [r.tree_id]).fetchone()[0] for n, r in results.items()}
    con.close()
    con = store.connect(path)
    after = {n: con.execute(ROOT_SQL, [r.tree_id]).fetchone()[0] for n, r in results.items()}
    assert before == after == {n: r.root_branch_id for n, r in results.items()}
    con.close()


def test_root_id_unchanged_by_later_writes():
    """An unrelated import (new staging rows, new registry row) does not move the root."""
    con, results = load_all("F1_tiny3")
    res = results["main"]
    fx = FIXTURES["F2_balanced8"]
    ingest.import_cultivar(con, fx["cultivars"][0]["z"], fx["item_ids"], "other")
    assert con.execute(ROOT_SQL, [res.tree_id]).fetchall() == [(res.root_branch_id,)]


def test_literal_volatile_root_is_unstable_in_duckdb():
    """Reproducer for ROOT-1: the literal PG expression is re-evaluated on every read in DuckDB.

    A view whose root row is ``gen_random_uuid() AS doc_id`` (v_tree_map.sql:33)
    returns a different id on each read, so it cannot serve as a node identity.
    """
    con, results = load_all("F1_tiny3")
    con.execute(
        """
        CREATE TEMP VIEW literal_root AS
        SELECT v_tree_z.tree_id, gen_random_uuid() AS doc_id
        FROM treedb.v_tree_z GROUP BY v_tree_z.tree_id
        """
    )
    a = con.execute("SELECT doc_id FROM literal_root").fetchone()[0]
    b = con.execute("SELECT doc_id FROM literal_root").fetchone()[0]
    assert a != b
    assert results["main"].root_branch_id not in (a, b)
