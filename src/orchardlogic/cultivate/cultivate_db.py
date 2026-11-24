"""DB half of cultivate: ``cultivate_into_db`` and incremental ``insert_items`` (spec 6.1, A4).

treedb is imported lazily inside the functions (optional extra ``db``); orchard imports fine
without it. Convention for new item rows: ``node_key = "item:<item_id>"`` (passed to
``cutstore.add_items``; cultivation-time items keep their ``member_...`` leaf keys).
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchardlogic.cultivate.document import Document
from orchardlogic.cultivate.enrichers import TreeContext, get_enricher, handle_of
from orchardlogic.cultivate.perspectives import validate_perspective
from orchardlogic.cultivate.placement import place_items, recompute_display


DEFAULT_PLACEMENT = "knn"  # chosen by scripts/eval_insert_forte.py (insert_eval.json)


@contextmanager
def _connection(db: Any, *, read_only: bool = False):
    """Yield a treedb connection for a path (opened with retry, closed after) or a connection."""
    if isinstance(db, (str, Path)):
        from orchardlogic.treedb import store

        con = store.connect_with_retry(db, read_only=read_only)
        try:
            yield con
        finally:
            con.close()
    else:
        yield db


@dataclass
class InsertResult:
    """Outcome of ``insert_items``."""

    placements: list[dict[str, Any]] = field(default_factory=list)  # new items only
    new_item_ids: list[str] = field(default_factory=list)
    recaptured_item_ids: list[str] = field(default_factory=list)
    refreshed_nodes: list[str] = field(default_factory=list)  # ancestors whose display was redone
    cut_name: str = ""
    cut_revision: int | None = None
    tree_attached: bool = False  # True only if every new item was attached to the orchard tree
    attach: dict[str, str | None] = field(default_factory=dict)  # item_id -> None | failure reason
    ledger_event_id: str | None = None
    write_lock_seconds: float | None = None  # read-write open of the main write transaction
    attach_lock_seconds: float | None = None  # longest read-write hold of the per-item attach steps


def cultivate_into_db(
    docs: Sequence[Document | Mapping[str, Any]],
    perspective: Mapping[str, Any],
    *,
    name: str,
    db: Any,
    alias: str | None,
    out_dir: str | Path | None = None,
    encoder: Any = None,
    created_at: str | None = None,
    documents: str = "keep",
):
    """``cultivate`` then ``treedb.registry.import_cultivar_artifact``; returns ``CultivarImport``.

    ``out_dir=None`` writes the cultivar to a temporary directory that is removed afterwards.
    """
    from orchardlogic.treedb import registry

    from orchardlogic.cultivate.cultivate import cultivate

    def run(path: Path):
        cultivate(docs, perspective, name=name, out_dir=path, encoder=encoder, created_at=created_at)
        with _connection(db) as con:
            return registry.import_cultivar_artifact(con, path, alias=alias, documents=documents)

    if out_dir is not None:
        return run(Path(out_dir))
    with tempfile.TemporaryDirectory(prefix="cultivar-") as tmp:
        return run(Path(tmp) / "cultivar")


def _open(db: Any, *, read_only: bool):
    """Context manager: a short connection for a path; the caller's connection otherwise."""
    return _connection(db, read_only=read_only)


@dataclass
class _Snapshot:
    tid: Any
    record: dict
    cut_name: str
    space: str
    rows: list
    revision: int | None
    stored: dict  # item_id -> vector for the cut's items
    docs: dict  # item_id -> Document for the cut's items
    item_outputs: dict  # handle -> item_id -> output
    node_outputs: dict  # handle -> node_key -> output
    have: dict  # sha -> vector for the new texts that are already cached


def _read_phase(con, alias, documents, perspective, cut_name) -> _Snapshot:
    """Everything the compute phase needs, read in one short (read-only) connection."""
    from orchardlogic.treedb import cutstore, documents as tdocs, enrichment, registry, vectors

    from orchardlogic.cultivate.cultivate import embedding_texts, text_sha256

    tid = registry.resolve(con, alias)
    record = dict(perspective) if perspective else dict(registry.cultivar_info(con, tid)["perspective"])
    validate_perspective(record)
    cut_name = cut_name or record["cuts"][0]["name"]
    space = record["embedding"]["space"]
    rows = cutstore.read_cut(con, tid, cut_name)
    revision = cutstore.cut_revision(con, tid, cut_name)
    cut_ids = sorted(r["item_id"] for r in rows if r["kind"] == "item")
    wanted = set(cut_ids)
    docs = {
        rec["item_id"]: Document(
            text=rec["text"], item_id=rec["item_id"], title=rec["title"], metadata=rec["metadata"] or {}
        )
        for rec in tdocs.recent_items(con, limit=10**9, with_text=True)
        if rec["item_id"] in wanted
    }
    ids, matrix = vectors.item_matrix(con, space, cut_ids)
    shas = sorted({text_sha256(t) for t in embedding_texts(documents, record)})
    return _Snapshot(
        tid=tid,
        record=record,
        cut_name=cut_name,
        space=space,
        rows=rows,
        revision=revision,
        stored={i: matrix[n] for n, i in enumerate(ids)},
        docs=docs,
        item_outputs={h: enrichment.get_item_outputs(con, h, cut_ids) for h in record["item_enrichers"]},
        node_outputs={h: enrichment.get_node_outputs(con, tid, h) for h in record["tree_enrichers"]},
        have=vectors.get_vectors(con, space, shas),
    )


@dataclass
class _Plan:
    result: InsertResult
    documents: list
    shas: list
    input_sha: dict
    item_out: dict  # handle -> outputs for the batch
    new_vec_rows: list  # (sha, vector) to store
    fresh_nodes: dict  # handle -> node outputs
    display: dict


def _compute(snap: _Snapshot, documents, encoder, placement) -> _Plan:
    """All CPU work, in memory: enrichment, embedding, placement, tree enrichers, display."""
    from orchardlogic.cultivate.cultivate import embed_texts, embedding_texts, text_sha256

    record, rows = snap.record, [dict(r) for r in snap.rows]
    result = InsertResult(cut_name=snap.cut_name)
    item_row = {r["item_id"]: r for r in rows if r["kind"] == "item"}
    result.new_item_ids = [d.item_id for d in documents if d.item_id not in item_row]
    result.recaptured_item_ids = [d.item_id for d in documents if d.item_id in item_row]

    texts = embedding_texts(documents, record)
    shas = [text_sha256(t) for t in texts]
    input_sha = {d.item_id: s for d, s in zip(documents, shas, strict=True)}
    item_out = {h: get_enricher(h).enrich(documents) for h in record["item_enrichers"]}
    have = dict(snap.have)
    todo = [i for i, s in enumerate(shas) if s not in have]
    new_vec_rows = []
    if todo:
        fresh = embed_texts([texts[i] for i in todo], encoder)
        for n, i in enumerate(todo):
            have[shas[i]] = fresh[n]
            new_vec_rows.append((shas[i], fresh[n]))
    doc_vec = {d.item_id: have[s] for d, s in zip(documents, shas, strict=True)}

    if result.new_item_ids:
        strategy = placement or record.get("placement") or DEFAULT_PLACEMENT
        result.placements = place_items(
            rows, snap.stored, {i: doc_vec[i] for i in result.new_item_ids}, strategy=strategy
        )
        by_key = {r["node_key"]: r for r in rows}
        next_pos: dict[str, int] = {}
        for r in rows:
            if r["parent_key"] is not None:
                next_pos[r["parent_key"]] = max(next_pos.get(r["parent_key"], 0), r["position"] + 1)
        for p in result.placements:
            parent = by_key[p["parent_key"]]
            pos = next_pos.get(p["parent_key"], 0)
            next_pos[p["parent_key"]] = pos + 1
            row = {
                "node_key": p["node_key"], "parent_key": p["parent_key"], "kind": "item",
                "depth": parent["depth"] + 1, "position": pos, "item_id": p["item_id"],
                "descendant_count": 1, "path": list(p["path"]), "title": None, "summary": None,
                "tags": None,
            }  # fmt: skip
            rows.append(row)
            by_key[row["node_key"]] = row
            for anc in p["path"]:
                by_key[anc]["descendant_count"] += 1

    affected: set[str] = set()
    for p in result.placements:
        affected.update(p["path"])
    for item_id in result.recaptured_item_ids:
        row = item_row[item_id]
        affected.update([*row["path"], row["parent_key"]])
    fresh_nodes: dict = {}
    display: dict = {}
    if affected:
        result.refreshed_nodes = sorted(affected)
        only = frozenset(affected)
        docs = {**snap.docs, **{d.item_id: d for d in documents}}
        vecs = {**snap.stored, **doc_vec}
        item_outputs = {
            h: {**snap.item_outputs.get(h, {}), **item_out.get(h, {})} for h in record["item_enrichers"]
        }
        node_outputs = {h: dict(v) for h, v in snap.node_outputs.items()}
        cut_doc = {"tree_id": str(snap.tid), "name": snap.cut_name, "nodes": rows}
        for handle in record["tree_enrichers"]:
            ctx = TreeContext(
                tree_id=str(snap.tid), cut=cut_doc, docs=docs, item_outputs=item_outputs,
                node_outputs=node_outputs, vectors=vecs, only=only,
            )  # fmt: skip
            fresh_nodes[handle] = get_enricher(handle).enrich(ctx)
            node_outputs[handle] = {**node_outputs[handle], **fresh_nodes[handle]}
        existing = {
            r["node_key"]: {"title": r["title"], "summary": r["summary"], "tags": r["tags"]}
            for r in rows
            if r["kind"] != "item"
        }
        display = recompute_display(rows, node_outputs, record["display"], only=only, existing=existing)
    return _Plan(result, documents, shas, input_sha, item_out, new_vec_rows, fresh_nodes, display)


def _write_phase(con, alias, snap: _Snapshot, plan: _Plan, actor) -> bool:
    """One short read-write transaction; False if the alias or the cut changed since the read."""
    from orchardlogic.treedb import cutstore, documents as tdocs, enrichment, ledger, registry, vectors
    from orchardlogic.treedb.tx import transaction

    result = plan.result
    with transaction(con):
        if registry.resolve(con, alias) != snap.tid:
            return False  # the alias was re-pointed (rebuild/set_alias) since the read phase
        if cutstore.cut_revision(con, snap.tid, snap.cut_name) != snap.revision:
            return False
        tdocs.upsert_documents(
            con,
            [
                {"item_id": d.item_id, "title": d.title, "text": d.text, "metadata": dict(d.metadata)}
                for d in plan.documents
            ],
        )
        for handle, outputs in plan.item_out.items():
            enrichment.put_item_outputs(con, handle, outputs, input_sha=plan.input_sha)
        if plan.new_vec_rows:
            vectors.put_vectors(con, snap.space, plan.new_vec_rows)
        links = [(d.item_id, s) for d, s in zip(plan.documents, plan.shas, strict=True)]
        vectors.link_items(con, snap.space, links)
        if result.placements:
            cutstore.add_items(
                con, snap.tid, snap.cut_name,
                [{"item_id": p["item_id"], "parent_key": p["parent_key"], "node_key": p["node_key"]}
                 for p in result.placements],
            )  # fmt: skip
        for handle, fresh in plan.fresh_nodes.items():
            enrichment.put_node_outputs(con, snap.tid, handle, fresh)
        if plan.display:
            cutstore.update_display(con, snap.tid, snap.cut_name, plan.display)
        result.cut_revision = cutstore.cut_revision(con, snap.tid, snap.cut_name)
        result.ledger_event_id = str(
            ledger.append(
                con, "insert", actor=actor, tree_id=str(snap.tid),
                payload={"alias": alias, "cut": snap.cut_name, "new": result.new_item_ids,
                         "recaptured": result.recaptured_item_ids,
                         "refreshed_nodes": len(result.refreshed_nodes)},
            )  # fmt: skip
        )
    return True


def insert_items(
    db: Any,
    alias: str,
    docs: Sequence[Document | Mapping[str, Any]],
    *,
    perspective: Mapping[str, Any] | None = None,
    encoder: Any = None,
    cut_name: str | None = None,
    placement: str | None = None,
    attach: bool = True,
    actor: str | None = "orchard.insert_items",
    max_attempts: int = 3,
) -> InsertResult:
    """Incremental insert (spec 6.1): documents, item enrichment, vectors, placement, display.

    Three phases so the DuckDB write lock is held briefly when ``db`` is a path:
    (1) a short read-only open reads the cut, vectors, documents and stored enrichment;
    (2) with no connection open, enrichment, embedding, placement, tree enrichers and display
    are computed in memory; (3) one short read-write open writes everything in a single
    transaction, first re-checking ``cut.revision`` (if the cut changed meanwhile, phases 1-2
    are redone, up to ``max_attempts``). (4) ``attach_node`` runs afterwards in its own short
    open (``attach=False`` skips it); failures are per item in ``InsertResult.attach`` and never
    undo the cut insert. Items already in the cut are re-captured: re-enriched and re-embedded,
    their ancestors refreshed, no second row. With a connection, the same phases run on it.
    ``write_lock_seconds`` / ``attach_lock_seconds`` on the result time the read-write opens.
    """
    import time

    from orchardlogic.cultivate.cultivate import as_documents

    documents = list({d.item_id: d for d in as_documents(docs)}.values())  # last one wins
    for _ in range(max_attempts):
        with _open(db, read_only=True) as con:
            snap = _read_phase(con, alias, documents, perspective, cut_name)
        plan = _compute(snap, documents, encoder, placement)
        with _open(db, read_only=False) as con:
            start = time.perf_counter()
            committed = _write_phase(con, alias, snap, plan, actor)
            lock = time.perf_counter() - start
        if committed:
            break
    else:
        raise RuntimeError(f"insert_items: {alias!r}/{snap.cut_name!r} kept changing; gave up")
    result = plan.result
    result.write_lock_seconds = lock

    if attach and result.placements:
        holds = []
        for p in result.placements:  # one short read-write open per item
            with _open(db, read_only=False) as con:
                start = time.perf_counter()
                result.attach[p["item_id"]] = _attach(con, snap.tid, p)
                holds.append(time.perf_counter() - start)
        result.attach_lock_seconds = max(holds)  # longest single hold
        result.tree_attached = all(v is None for v in result.attach.values())
    return result


def _attach(con, tid, placement: Mapping[str, Any]) -> str | None:
    """Attach via ``treedb.registry.attach_item`` (A4); None on success, else the failure reason."""
    try:
        from orchardlogic.treedb import registry

        outcome = registry.attach_item(
            con, tid, placement["parent_key"], placement["node_key"], placement["item_id"]
        )
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return f"{type(exc).__name__}: {exc}"
    return None if outcome.ok else f"{outcome.reason}: {outcome.detail}"


__all__ = ["InsertResult", "cultivate_into_db", "insert_items"]
