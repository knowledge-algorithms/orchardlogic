# OrchardLogic

OrchardLogic builds and maintains reusable hierarchical structures over a collection of
documents or records, such as trees of topics, functions or domains. Decision models, LLMs, agents and other
software can then navigate, filter and query those structures. OrchardLogic prepares
the structures. It is not a decision model or an orchestration framework.

It has two parts in one distribution:

- **`orchardlogic.cultivate`** builds trees over a corpus. It computes several similarity
  layers (TF-IDF, optional MiniLM embeddings, taxonomy distributions), fuses them with
  explicit weights, clusters them hierarchically, cuts the result into readable levels,
  labels nodes, places new items into a fitted tree, and saves everything as versioned
  artifacts.
- **`orchardlogic.treedb`** stores those trees ("cultivars") over a shared set of item ids
  in embedded DuckDB. It supports cohorts (bounded item sets), projection onto any tree,
  deterministic include/exclude queries with explanations, attach/detach mutations, and
  read-only SQL macros. No server is needed.

Status: alpha (0.1.0). The APIs come from two libraries that were used in research
projects. Expect changes before 1.0.

## Install

```bash
uv add orchardlogic        # or: pip install orchardlogic
```

Python 3.11 or newer. Base dependencies: numpy, scipy, scikit-learn, duckdb.

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `embeddings` | torch, transformers | MiniLM description layer |
| `taxonomy-ml` | torch, transformers | ModernBERT taxonomy heads |
| `labeling` | openai | live LLM node labeling |

Without the neural extras, `cultivate` runs offline on TF-IDF and taxonomy cue sets
when you opt in with `allow_offline_fallback=True`.

## Cultivate: build trees

```python
from orchardlogic.cultivate import Orchard, OrchardBuilder

documents = [
    "Schedule a calendar reminder for tomorrow morning.",
    "Send an email summary of the weekly status notes.",
    "Create a task to review the draft proposal.",
    "Find documents mentioning quarterly budget planning.",
]
orchard = OrchardBuilder(allow_offline_fallback=True).build(documents)
tree = orchard.tree("semantic")
print(tree.linkage)                      # SciPy linkage over the 4 documents
orchard.save("my_orchard")               # versioned artifact directory
assert Orchard.load("my_orchard").tree_ids == ("semantic",)
```

With no taxonomies, the builder makes one `semantic` tree. Pass your own
`TaxonomyModel` objects (`TaxonomyModel.from_definition(...)`, then `.fit(...)` or
`.load_head(...)`) to get one named tree per taxonomy. 0.1.0 ships no packaged
taxonomy definitions.

## TreeDB: store, query and mutate trees

```python
from orchardlogic.treedb import TreeDB
from orchardlogic.treedb.fixtures import item_uuid
from orchardlogic.treedb.query import Query

with TreeDB() as db:                       # TreeDB("my.duckdb") persists to a file
    items = ["a", "b", "c", "d"]
    db.import_cultivar([[0, 1, 1.0, 2], [2, 3, 1.0, 2], [4, 5, 2.0, 4]], items, "by_name")
    db.import_cultivar([[0, 2, 1.0, 2], [1, 3, 1.0, 2], [4, 5, 2.0, 4]], items, "by_size")

    cohort = db.cohort(["a", "b", "c"])
    p = db.project(cohort, "by_name", output="scipy")          # compressed subtree
    branch = db.node("by_name", item_uuid("a")).parent_branch_id
    result = db.execute(db.all_items(), Query.from_json(
        {"rules": [{"cultivar": "by_name", "node": str(branch), "op": "include"}]}))
    assert result.items == ("a", "b")
```

More: [docs/treedb.md](docs/treedb.md) (full quickstart, tested) and
[docs/treedb_sql_api.md](docs/treedb_sql_api.md) (SQL macros).

## Using them together

`orchardlogic.cultivate.cultivate_into_db` writes a cultivated tree, its cuts,
enrichments and vectors into a TreeDB database. `insert_items` then places new
documents into that stored tree. See `examples/cultivate_into_treedb.py`.

## License

Apache License 2.0. Copyright 2026 Energy Algorithms, LLC. See [LICENSE](LICENSE),
[NOTICE](NOTICE) and [PROVENANCE.md](PROVENANCE.md).

## Contributing

Issues and pull requests: https://github.com/doc1000/orchardlogic/issues.
See [CONTRIBUTING.md](CONTRIBUTING.md).
