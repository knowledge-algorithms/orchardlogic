# Provenance

OrchardLogic 0.1.0 was assembled from two existing libraries. Their code was copied
at the commits below. Algorithms and serialization formats were not changed; the
edits are listed under "Changes made during the port".

| Subpackage | Source repository | Branch | Commit |
|---|---|---|---|
| `orchardlogic.cultivate` | https://github.com/doc1000/orchard (package `orchard`) | `main` | `611de12` |
| `orchardlogic.treedb` | https://github.com/doc1000/treeDB (package `treedb`) | `main` | `a1e0a93` |

Both libraries were ported from prior work owned by Energy Algorithms, LLC
(owner confirmation, 2026-11-24). The source repositories keep their full history.
This repository starts with a fresh history.

## Changes made during the port

- Import paths: `orchard` became `orchardlogic.cultivate` and `treedb` became
  `orchardlogic.treedb`, including `importlib.resources` package names and
  install hints such as `orchardlogic[embeddings]`.
- Removed `orchard.mcp_server` and its tests (MCP is out of scope for 0.1.0).
  `navigation`, `catalog` and `render` are kept.
- Removed the `mcp` and `db` extras of orchard. TreeDB is now part of the same
  distribution, so `cultivate_into_db` and `insert_items` work without an extra.
- Removed the packaged default taxonomies and everything derived from AppWorld
  data: `assets/taxonomies/` (Domain and Function definitions and their ModernBERT
  logistic heads) and the `domain`/`function` similarity profiles. Removed with them:
  `DomainTaxonomy`, `FunctionTaxonomy`, `default_taxonomies`,
  `TaxonomyModel.load_default`, `TaxonomyModel.attach_packaged_head` and the
  packaged-head loaders in `backends.taxonomy_heads`. `OrchardBuilder()` without
  `taxonomies` now builds one `semantic` tree (it used to build `domain` and
  `function`). Custom taxonomies (`TaxonomyModel.from_definition`, `fit`,
  `load_head`) and caller-supplied profiles work as before. The `semantic.json`
  weight pair (0.66 MiniLM / 0.34 TF-IDF) is kept.
- Not carried over: TreeDB's `reference/` snapshots, PostgreSQL golden outputs and
  harness, the AppWorld fixture, the porting, golden, acceptance, oracle and perf
  test suites, the Orchard docs site, developer tools, plans, prompts, reviews and
  design docs. The test suite was cut to a fast offline set (about 550 tests).
- Enricher records (`code.module`, `code.sha256`) now name the `orchardlogic`
  module paths, so their hashes differ from records written by the old `orchard`
  package. Cultivar, cut, label and TreeDB formats are unchanged.
- Kept unchanged: SQL schema names (`treedb.*` inside DuckDB), artifact and schema
  version strings (`orchard_cultivar_v1`, `treedb.query/1`, ...), logger names,
  ledger actor names, and the subpackages' own `__version__` values.

Some docstrings cite `reference/...` paths or decision ids (`D-0xx`). Those refer to
files in the source repositories at the commits above.
