import duckdb

import orchardlogic.treedb as treedb


def test_package_imports():
    assert treedb.__version__


def test_duckdb_recursive_cte_and_table_macro_available():
    con = duckdb.connect()
    con.execute(
        "CREATE MACRO up_to(n) AS TABLE "
        "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r WHERE i < n) "
        "SELECT i FROM r"
    )
    assert con.execute("SELECT sum(i) FROM up_to(4)").fetchone()[0] == 10
