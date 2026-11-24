"""The TreeDB facade (phase P6A): one small class over the modules already built.

Every method delegates to the module that owns the behaviour and adds none. Results are the
modules' own result objects, so the ``state_version`` they carry is the module's:

* ``node`` -> ``read.NodeRecord`` (``state_version``), ``cultivars`` -> ``read.tree_info`` dicts;
* ``ancestry`` / ``subtree(output="raw")`` -> ``read.ItemReadResult`` (``state_version`` through its
  resolution); ``subtree(output="scipy")`` -> ``(Z, leaves, context)`` with ``context["state_version"]``;
* ``attach`` / ``detach`` -> ``mutate.AttachResult`` / ``DetachResult`` (``state_version`` after commit);
* ``cohort`` / ``union`` / ... -> ``cohort.Cohort`` (``cultivar_state_versions``, INV-18);
* ``project`` -> ``project.Projection`` (``state_version`` and the D-021 ``moved`` notice);
* ``execute`` -> ``query.ExecuteResult``; ``explain`` -> ``query.Explanation``.

``on_moved`` and ``on_missing`` pass straight through, so ``CohortMovedWarning`` (D-021) reaches the
caller. (The warning is attributed to the line in this module that calls ``project``; the active
warning filters still apply.) Mutation is Python only (D-020, OD-10). Nothing here imports
scikit-learn (the ``cuts`` extra, D-028, stays optional).

``TreeDB.connection`` is the open DuckDB connection, for the ``treedb_*`` SQL macros (docs/SQL_API.md).
"""

from __future__ import annotations

import uuid
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb

from orchardlogic.treedb import checkpoints, cutstore, documents, enrichment, ledger, registry, vectors
from orchardlogic.treedb import cohort as cohort_mod
from orchardlogic.treedb import metadata, mutate, normalize, read, store
from orchardlogic.treedb import project as project_mod
from orchardlogic.treedb import query as query_mod
from orchardlogic.treedb.adapters import orchard_artifact
from orchardlogic.treedb.ingest import ImportedCultivar, import_cultivar

SUBTREE_OUTPUTS = ("raw", "scipy")


class TreeDB:
    """An in-memory (``path=None``) or file-backed TreeDB. Reopening a file keeps cultivars and cohorts."""

    def __init__(self, path: str | Path | None = None, *, migrate: bool = False):
        self.path = None if path is None else Path(path)
        self._con: duckdb.DuckDBPyConnection | None = store.connect(path, migrate=migrate)   # installs the schema (idempotent)
        try:
            read.install(self._con)           # always CREATE OR REPLACE: a reopened file gets current macros
            cohort_mod.install(self._con)
            self._con.execute(resources.files("orchardlogic.treedb").joinpath("sql/api_macros.sql").read_text(encoding="utf-8"))
        except BaseException:
            self._con.close()
            self._con = None
            raise

    # -- connection and lifecycle -----------------------------------------------------------

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise ValueError("this TreeDB is closed")
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "TreeDB":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- import ----------------------------------------------------------------------------

    def import_cultivar(self, Z: Any, item_ids: Sequence[str], name: str,
                        labels: Mapping[str, Mapping[str, str]] | None = None,
                        cards: Mapping[str, Any] | None = None, source: Any = None) -> ImportedCultivar:
        """``ingest.import_cultivar``."""
        return import_cultivar(self.connection, Z, item_ids, name, labels=labels, cards=cards, source=source)

    def import_orchard_artifact(self, path: str | Path) -> orchard_artifact.ArtifactImport:
        """``adapters.orchard_artifact.import_orchard_artifact``."""
        return orchard_artifact.import_orchard_artifact(self.connection, path)

    def import_cultivar_artifact(self, path: str | Path, **kw: Any) -> registry.CultivarImport:
        """``registry.import_cultivar_artifact``."""
        return registry.import_cultivar_artifact(self.connection, path, **kw)

    # -- Data Orchard v4 (thin delegates; docs/DATA_ORCHARD_V4.md section 5.1) --------------

    def put_perspective(self, record: Any) -> str:
        return registry.put_perspective(self.connection, record)

    def put_enricher(self, record: Any) -> str:
        return registry.put_enricher(self.connection, record)

    def set_alias(self, alias: str, tree_id: Any) -> None:
        registry.set_alias(self.connection, alias, tree_id)

    def resolve(self, alias_or_name_or_uuid: Any) -> uuid.UUID:
        return registry.resolve(self.connection, alias_or_name_or_uuid)

    def cultivar_info(self, alias_or_tree: Any) -> dict[str, Any]:
        return registry.cultivar_info(self.connection, alias_or_tree)

    def upsert_documents(self, docs: Sequence[Mapping[str, Any]]) -> documents.UpsertResult:
        return documents.upsert_documents(self.connection, docs)

    def recent_items(self, **kw: Any) -> list[dict[str, Any]]:
        return documents.recent_items(self.connection, **kw)

    def put_vectors(self, space: str, rows: Any) -> int:
        return vectors.put_vectors(self.connection, space, rows)

    def link_items(self, space: str, pairs: Any) -> None:
        vectors.link_items(self.connection, space, pairs)

    def item_matrix(self, space: str, item_ids: Any = None):
        return vectors.item_matrix(self.connection, space, item_ids)

    def nearest(self, space: str, query: Any, k: int = 10, item_ids: Any = None):
        return vectors.nearest(self.connection, space, query, k, item_ids)

    def read_cut(self, alias_or_tree: Any, cut_name: str, **kw: Any) -> list[dict[str, Any]]:
        return cutstore.read_cut(self.connection, registry.resolve(self.connection, alias_or_tree), cut_name, **kw)

    def add_cut_items(self, alias_or_tree: Any, cut_name: str, placements: Sequence[Mapping[str, Any]]) -> int:
        return cutstore.add_items(self.connection, registry.resolve(self.connection, alias_or_tree), cut_name,
                                  placements)

    def ledger_append(self, action: str, **fields: Any) -> uuid.UUID:
        return ledger.append(self.connection, action, **fields)

    def ledger_events(self, **filters: Any) -> list[dict[str, Any]]:
        return ledger.events(self.connection, **filters)

    def put_checkpoint(self, namespace: str, key: str, payload: Any) -> None:
        checkpoints.put_checkpoint(self.connection, namespace, key, payload)

    def get_checkpoint(self, namespace: str, key: str) -> Any:
        return checkpoints.get_checkpoint(self.connection, namespace, key)

    # -- read ------------------------------------------------------------------------------

    def cultivars(self) -> list[dict[str, Any]]:
        """``read.tree_info`` of every cultivar, ordered by name."""
        con = self.connection
        names = [r[0] for r in con.execute("SELECT name FROM treedb.tree_registry ORDER BY name").fetchall()]
        return [read.tree_info(con, n) for n in names]

    def node(self, cultivar: str | uuid.UUID, node_id: Any) -> read.NodeRecord:
        """``read.get_node``."""
        return read.get_node(self.connection, cultivar, node_id)

    def ancestry(self, cultivar: str | uuid.UUID, items: Iterable[str], *,
                 on_missing: str = "report") -> read.ItemReadResult:
        """``read.ancestry_paths_for_items``."""
        return read.ancestry_paths_for_items(self.connection, cultivar, items, on_missing=on_missing)

    def node_ancestry(self, cultivar: str | uuid.UUID, node_id: Any) -> list[tuple]:
        """``read.node_ancestry``: the chain above one node."""
        return read.node_ancestry(self.connection, cultivar, node_id)

    def subtree(self, cultivar: str | uuid.UUID, items: Iterable[str], *, output: str = "raw",
                on_missing: str = "report", leaf_order: str = "raw"):
        """Compressed Steiner subtree of ``items``.

        ``output="raw"``: ``read.compressed_subtree_for_items`` (an ``ItemReadResult``).
        ``output="scipy"``: ``normalize.scipy_linkage_for_items`` (``(Z, leaves, context)``).
        """
        if output not in SUBTREE_OUTPUTS:
            raise ValueError(f"output must be one of {SUBTREE_OUTPUTS}, got {output!r}")
        if output == "raw":
            return read.compressed_subtree_for_items(self.connection, cultivar, items, on_missing=on_missing)
        return normalize.scipy_linkage_for_items(self.connection, cultivar, items, on_missing=on_missing,
                                                 leaf_order=leaf_order)

    # -- mutation (Python only, D-020) --------------------------------------------------------

    def attach(self, new_node_id: uuid.UUID | str | None, target_orchard_id: uuid.UUID | str | None,
               new_edge_ix: int | None, coph_distance: float | None = None, *, strict: bool | None = None,
               item_id: str | None = None, source_ref: str | None = None) -> mutate.AttachResult:
        """``mutate.attach_node``."""
        return mutate.attach_node(self.connection, new_node_id, target_orchard_id, new_edge_ix, coph_distance,
                                  strict=strict, item_id=item_id, source_ref=source_ref)

    def detach(self, target_orchard_id: uuid.UUID | str | None) -> mutate.DetachResult:
        """``mutate.detach_node``."""
        return mutate.detach_node(self.connection, target_orchard_id)

    # -- metadata (8.9) -----------------------------------------------------------------------

    def set_label(self, cultivar: str | uuid.UUID, node_id: Any, label_set: str, label: str) -> None:
        """``metadata.set_label``: no change to topology, lineage or ``state_version``."""
        metadata.set_label(self.connection, cultivar, node_id, label_set, label)

    def set_card(self, cultivar: str | uuid.UUID, node_id: Any, card: Any) -> None:
        """``metadata.set_card``: no change to topology, lineage or ``state_version``."""
        metadata.set_card(self.connection, cultivar, node_id, card)

    # -- cohorts ------------------------------------------------------------------------------

    def all_items(self) -> cohort_mod.Cohort:
        """``cohort.all_items``."""
        return cohort_mod.all_items(self.connection)

    def cohort(self, ids: Iterable[str]) -> cohort_mod.Cohort:
        """``cohort.from_ids``: unknown ids raise ``UnknownItemsError``."""
        return cohort_mod.from_ids(self.connection, ids)

    def union(self, left: cohort_mod.CohortRef, right: cohort_mod.CohortRef, *more: cohort_mod.CohortRef) -> cohort_mod.Cohort:
        """``cohort.union``."""
        return cohort_mod.union(self.connection, left, right, *more)

    def intersect(self, left: cohort_mod.CohortRef, right: cohort_mod.CohortRef, *more: cohort_mod.CohortRef) -> cohort_mod.Cohort:
        """``cohort.intersect``."""
        return cohort_mod.intersect(self.connection, left, right, *more)

    def difference(self, left: cohort_mod.CohortRef, right: cohort_mod.CohortRef) -> cohort_mod.Cohort:
        """``cohort.difference``."""
        return cohort_mod.difference(self.connection, left, right)

    def items(self, cohort: cohort_mod.CohortRef) -> list[str]:
        """``cohort.items``: the sorted members."""
        return cohort_mod.items(self.connection, cohort)

    def describe(self, cohort: cohort_mod.CohortRef) -> dict[str, Any]:
        """``cohort.describe``: provenance and how far each cultivar has moved since creation."""
        return cohort_mod.describe(self.connection, cohort)

    def export_cohort(self, cohort: cohort_mod.CohortRef) -> dict[str, Any]:
        """Cohort transfer (D-033): the ID list plus provenance. Never any linkage.

        Keys: ``cohort_id``, ``membership_hash``, ``item_ids`` and ``cultivar_state_versions``
        (the ``state_version`` per cultivar recorded when the cohort was made).
        """
        c = cohort_mod.get(self.connection, cohort)
        return {
            "cohort_id": str(c.cohort_id),
            "membership_hash": c.membership_hash,
            "item_ids": cohort_mod.items(self.connection, c),
            "cultivar_state_versions": c.cultivar_state_versions,
        }

    # -- projection and queries ---------------------------------------------------------------

    def project(self, cohort: cohort_mod.CohortRef, cultivar: str | uuid.UUID, *, output: str = "raw",
                on_missing: str = "report", on_moved: str = "report") -> project_mod.Projection:
        """``project.project``."""
        return project_mod.project(self.connection, cohort, cultivar, output=output,
                                   on_missing=on_missing, on_moved=on_moved)

    def execute(self, base_cohort: cohort_mod.CohortRef,
                query: query_mod.Query | query_mod.NormalQuery) -> query_mod.ExecuteResult:
        """``query.execute``: a new subset cohort of ``base_cohort``."""
        return query_mod.execute(self.connection, base_cohort, query)

    def explain(self, subject: query_mod.ExecuteResult | query_mod.Query | query_mod.NormalQuery,
                item_id: str, *, base: cohort_mod.CohortRef | None = None) -> query_mod.Explanation:
        """``query.explain``."""
        return query_mod.explain(self.connection, subject, item_id, base=base)


__all__ = ["TreeDB"]
