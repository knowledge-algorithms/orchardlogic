# SQL API

`TreeDB` installs a small set of read-only table macros, named `treedb_*`, on its DuckDB
connection (`db.connection`). They are for SQL users and for joining TreeDB data with other
tables. Each one returns what the matching Python call returns; `tests/api/test_api_sql.py`
checks that on the F5 fixture.

The macros live in `src/orchardlogic/treedb/sql/api_macros.sql` and are re-created every time a `TreeDB`
opens, so a reopened file always has the current definitions. The underlying objects
(`treedb.v_orchard`, `treedb.get_ancestry_paths`, `treedb.get_compressed_subtree`,
`treedb.cohort_items`, ...) remain available.

Conventions: a cultivar is named by its name; a node is a UUID (text or `UUID`); a cohort is its
handle (text or `UUID`). The macros are plain SQL, so an item that a cultivar does not
contain is ignored. The Python calls report such items (`missing_items`, `unknown_items`,
D-022); use them when that matters.

| Macro | Returns | Python equivalent |
|---|---|---|
| `treedb_cultivars()` | `tree_id, cultivar, state_version` | `db.cultivars()` |
| `treedb_node(cultivar, node)` | `cultivar, state_version, node_id, kind, item_id, orchard_id, parent_branch_id, edge_ix, edge_coph_distance` (no row for an unknown node) | `db.node(cultivar, node)` |
| `treedb_ancestry(cultivar, node)` | the `graph.get_ancestry_paths` rows from the node's incoming row up to the root (none for a root) | `db.node_ancestry(cultivar, node)` |
| `treedb_subtree(cultivar, cohort)` | the raw `graph.get_compressed_subtree` rows of the cohort's items in that cultivar | `db.subtree(cultivar, items).rows` |
| `treedb_cohort_items(cohort)` | `item_id`, sorted | `db.items(cohort)` |
| `treedb_cohort(cohort)` | the cohort header row (`kind`, `size`, `membership_hash`, provenance, `cultivar_state_versions`) | `db.describe(cohort)` |

## Examples

`tests/api/test_sql_api_doc.py` runs every Python block below.

```python
from orchardlogic.treedb import TreeDB
from orchardlogic.treedb.fixtures import item_uuid

db = TreeDB()
items = ["a", "b", "c", "d"]
db.import_cultivar([[0, 1, 1.0, 2], [2, 3, 1.0, 2], [4, 5, 2.0, 4]], items, "demo")   # (a b)(c d)
sql = db.connection.execute

# Cultivars and the state_version that tree-dependent results carry.
assert sql("SELECT cultivar, state_version FROM treedb_cultivars()").fetchall() == [("demo", 0)]

# A node: item "a" is a leaf; its parent is a branch.
a = item_uuid("a")
kind, item = sql("SELECT kind, item_id FROM treedb_node('demo', ?)", [str(a)]).fetchone()
assert (kind, item) == ("leaf", "a")

# Ancestry: the rows from the leaf up to the root (here two: the leaf row and the (a b) branch row).
rows = sql("SELECT * FROM treedb_ancestry('demo', ?)", [str(a)]).fetchall()
assert len(rows) == 2

# A cohort, its items and its Steiner subtree in the cultivar.
cohort = db.cohort(["a", "b", "c"])
cid = str(cohort.cohort_id)
assert sql("SELECT item_id FROM treedb_cohort_items(?)", [cid]).fetchall() == [("a",), ("b",), ("c",)]
assert len(sql("SELECT * FROM treedb_subtree('demo', ?)", [cid]).fetchall()) > 0

# They compose with ordinary SQL.
n = sql("SELECT count(*) FROM treedb_cohort_items(?) i JOIN treedb.item_registry r USING (item_id)", [cid]).fetchone()[0]
assert n == 3
db.close()
```

## Why mutation is Python only

There are no `treedb_attach` or `treedb_detach` macros. D-020 (OD-10) decided that mutations use
the Python coordinator: the steps are SQL (`sql/mutate/*.sql`), but one call is one DuckDB
transaction with checks, the D-016b and D-016c rules, the `state_version` bump and the loud
`TreeSplitWarning`. A SQL macro cannot open a transaction, raise a Python warning or return the
`AttachResult` and `DetachResult` details. Call `db.attach` and `db.detach`. Labels and cards
(`db.set_label`, `db.set_card`) are also Python only: they are metadata edits that never change
topology or `state_version`.

## Notes

- The macros are read-only and take no lock of their own. Run several statements that must agree
  inside `BEGIN TRANSACTION` ... `COMMIT`.
- A cyclic store makes `treedb_ancestry` and `treedb_subtree` raise the depth-guard error from
  `treedb.get_ancestry_paths` (CYC-2); the Python calls turn it into `read.AncestryCycleError`.
- `execute`, `explain` and `project` are not macros. Their results are structured (provenance,
  notices, warnings) and a cohort is created by `execute`. Read an `execute` result back with
  `treedb_cohort_items`.
