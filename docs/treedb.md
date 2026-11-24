# TreeDB

Storage and query operators for mutable hierarchies over shared items, on embedded DuckDB. Import it as `orchardlogic.treedb`.

## Quickstart


`tests/treedb/api/test_readme.py` runs this block.

```python
import warnings

from orchardlogic.treedb import TreeDB
from orchardlogic.treedb.fixtures import item_uuid
from orchardlogic.treedb.query import Query

with TreeDB() as db:                      # TreeDB("my.duckdb") keeps everything in a file
    items = ["a", "b", "c", "d"]
    # Two cultivars (trees) over the same items. Z is a SciPy linkage: leaf i is items[i].
    db.import_cultivar([[0, 1, 1.0, 2], [2, 3, 1.0, 2], [4, 5, 2.0, 4]], items, "by_name")   # (a b)(c d)
    db.import_cultivar([[0, 2, 1.0, 2], [1, 3, 1.0, 2], [4, 5, 2.0, 4]], items, "by_size")   # (a c)(b d)

    # A cohort is a bounded set of item ids with a handle; the algebra makes new cohorts.
    cohort = db.cohort(["a", "b", "c"])
    assert db.items(db.difference(cohort, db.cohort(["c"]))) == ["a", "b"]

    # Project it onto any cultivar: the compressed subtree, normalized to a SciPy linkage.
    p = db.project(cohort, "by_name", output="scipy")
    assert p.z.shape == (2, 4) and p.leaf_item_ids == ["a", "b", "c"] and p.state_version == 0

    # Query: a deterministic include/exclude over a base cohort gives a new, smaller cohort.
    branch = db.node("by_name", item_uuid("a")).parent_branch_id     # the (a b) branch
    result = db.execute(db.all_items(), Query.from_json(
        {"rules": [{"cultivar": "by_name", "node": str(branch), "op": "include"}]}))
    assert result.items == ("a", "b") and db.explain(result, "a").verdict is True

    # SQL: the same data through treedb_* macros (docs/treedb_sql_api.md).
    rows = db.connection.execute("SELECT item_id FROM treedb_cohort_items(?)", [str(cohort.cohort_id)]).fetchall()
    assert [r[0] for r in rows] == ["a", "b", "c"]

    # Mutate (Python only). Cultivars carry a state_version; a cohort made earlier is still
    # usable, and says so when a cultivar has moved since (D-021).
    db.detach(db.node("by_name", item_uuid("a")).orchard_id)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        moved = db.project(cohort, "by_name")
    assert moved.moved is not None and moved.missing_items == ("a",) and len(caught) == 1
```
