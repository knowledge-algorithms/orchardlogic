"""Import an Orchard artifact directory (``Orchard.save`` output) into a store.

Phase P6B. Reads the JSON files directly; there is **no import of the
``orchard`` package**. The checks follow ``orchard.py:Orchard.load`` (lines
cited below are from ``reference/orchard/src/orchard/orchard.py``, commit
``cd88c2de01dca01bf134656538810f3191e105af`` in ``reference/manifest.json``):

=====================================================  ================================
check here                                              orchard source
=====================================================  ================================
manifest ``schema_version == orchard_artifact_v1``      orchard.py:162-165
``document_count`` equals ``documents.json``            orchard.py:170-172
document ``schema_version`` (``orchard_document_v1``)   document.py (``from_mapping``)
unique, non-empty ``item_id``                           identity.py ``ensure_unique_item_ids``
linkage ``scipy_linkage_artifact_v1``                   tree.py ``from_persisted``
canonical tree ``canonical_linkage_tree_v1``            tree.py ``validate_canonical_tree``
label set ``orchard_label_set_v1``                      orchard.py:187-191
card set ``orchard_card_set_v1`` + name + card schema   orchard.py ``_load_card_sets``
root id equals the manifest and the canonical file      orchard.py:204-207
canonical node registry equals the one rebuilt from Z   tree.py ``from_persisted``
each tree's leaves equal the corpus                     orchard.py:67-73
=====================================================  ================================

The tree id syntax (``validate_tree_id``) is checked by ``ingest.import_cultivar``.
Everything is validated **before** the first write, including ``ingest``'s own
input checks for every tree (the public ``ingest.validate_cultivar``) and the
documents, so a bad artifact leaves the store untouched. The writes themselves
are one ``ingest.import_cultivar`` call per tree (each is atomic), then one
transaction for the documents (they name items the trees register). Only a
failure that no input check can see (for example a database error between
trees) could leave earlier trees committed.

Limits, all loud (``ArtifactError``) rather than silent:

* Orchard keeps several named card sets per tree; ``treedb.node_cards`` keeps
  one card per node, so a tree with more than one card set is refused.
* ``layer_matrices.npz`` is not read (the store has no place for it).

Documents (D-027) are written to ``treedb.documents`` (``item_id``, ``title``,
``text``, ``metadata``); the artifact's other document fields (``source``,
``schema_version``) are not kept. The full records also come back in
``ArtifactImport.documents``. A store that already has a document for any item of
the artifact refuses the import (the same rule as an existing cultivar name).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from orchardlogic.treedb import ingest
from orchardlogic.treedb.fixtures import canonical_json, node_keys

ARTIFACT_SCHEMA_VERSION = "orchard_artifact_v1"
DOCUMENT_SCHEMA_VERSION = "orchard_document_v1"
CANONICAL_TREE_SCHEMA_VERSION = "canonical_linkage_tree_v1"
LINKAGE_SCHEMA_VERSION = "scipy_linkage_artifact_v1"
LABEL_SET_SCHEMA_VERSION = "orchard_label_set_v1"
CARD_SET_SCHEMA_VERSION = "orchard_card_set_v1"
CARD_SCHEMA_VERSION = "orchard_node_card_v1"
_CARD_REQUIRED = ("schema_version", "title", "summary", "scope", "capabilities", "concepts", "uncertainty")


class ArtifactError(ValueError):
    """The artifact is invalid or unsupported. Nothing was written.

    That holds for every failure the pre-write checks can see. A database error
    raised during the writes (after the first tree committed, before the documents
    are written) is not an ``ArtifactError`` and may leave earlier trees committed.
    """


@dataclass(frozen=True)
class ArtifactImport:
    """Result of ``import_orchard_artifact``."""

    cultivars: dict[str, ingest.ImportedCultivar]   # tree_id -> imported cultivar, manifest order
    item_ids: tuple[str, ...]                       # corpus, in documents.json order
    documents: dict[str, dict[str, Any]]            # item_id -> document record (text, title, metadata, ...)
    metadata: dict[str, Any] = field(default_factory=dict)  # manifest "metadata"
    documents_written: int = 0                      # rows added to treedb.documents


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"missing artifact file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid JSON in {path}: {exc}") from exc


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ArtifactError(msg)


def _load_documents(root: Path, manifest: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    rows = _read_json(root / "documents.json")
    _check(isinstance(rows, list), "documents.json must be a list")
    docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        _check(isinstance(row, dict) and "text" in row, "each document needs a 'text' field")
        item_id = str(row.get("item_id") or "").strip()
        _check(bool(item_id), "document with an empty item_id")
        _check(item_id not in docs, f"duplicate item_id: {item_id}")
        version = row.get("schema_version", DOCUMENT_SCHEMA_VERSION)
        _check(version == DOCUMENT_SCHEMA_VERSION, f"unsupported document schema_version: {version!r}")
        title, metadata = row.get("title"), row.get("metadata")
        _check(isinstance(row["text"], str), f"document {item_id!r}: text must be a string")
        _check(title is None or isinstance(title, str), f"document {item_id!r}: title must be a string")
        _check(metadata is None or isinstance(metadata, dict), f"document {item_id!r}: metadata must be an object")
        docs[item_id] = row
    _check(len(docs) == manifest.get("document_count"), "document_count does not match documents.json")
    return list(docs), docs


def _load_tree(root: Path, entry: dict[str, Any], corpus: set[str]) -> dict[str, Any]:
    tree_id = entry["tree_id"]
    tree_dir = root / "trees" / tree_id
    canonical = _read_json(tree_dir / "canonical_tree.json")
    linkage = _read_json(tree_dir / "linkage.json")
    _check(linkage.get("schema_version") == LINKAGE_SCHEMA_VERSION, f"{tree_id}: unsupported linkage schema")
    _check(canonical.get("schema_version") == CANONICAL_TREE_SCHEMA_VERSION,
           f"{tree_id}: unsupported canonical tree schema")
    items = list(linkage["item_ids"])
    z = linkage["z_matrix"]
    _check(set(items) == corpus and len(items) == len(corpus),
           f"{tree_id}: tree leaf item_ids must exactly match the document corpus")
    _check(len(items) >= 2, f"{tree_id}: a cultivar needs at least 2 leaves")
    _check(len(z) == len(items) - 1, f"{tree_id}: z_matrix must have n-1 rows")

    keys = node_keys(z, items)                       # INV-3: same hash as orchard's canonical node ids
    root_key = keys[2 * len(items) - 2]
    _check(canonical.get("root_node_id") == root_key, f"{tree_id}: canonical root_node_id mismatch")
    _check(entry.get("root_node_id") == root_key, f"manifest root_node_id mismatch for tree {tree_id!r}")
    _check(set(canonical["nodes"]) == set(keys.values()), f"{tree_id}: persisted node registry mismatch")

    labels: dict[str, dict[str, str]] = {}
    for name in entry.get("label_sets", []):
        payload = _read_json(tree_dir / "labels" / f"{name}.json")
        _check(payload.get("schema_version") == LABEL_SET_SCHEMA_VERSION,
               f"unsupported label set schema for {name!r}")
        labels[name] = {str(k): v for k, v in payload["labels"].items()}

    card_names = list(entry.get("card_sets", []))
    _check(len(card_names) <= 1,
           f"{tree_id}: {len(card_names)} card sets; treedb.node_cards holds one card per node "
           "(refusing rather than dropping card sets)")
    cards: dict[str, Any] = {}
    for name in card_names:
        payload = _read_json(tree_dir / "cards" / f"{name}.json")
        _check(payload.get("schema_version") == CARD_SET_SCHEMA_VERSION, f"unsupported card set schema for {name!r}")
        _check(payload.get("name") == name, f"card set name mismatch for {name!r}")
        _check(isinstance(payload.get("cards"), dict), f"card set {name!r} is missing a cards object")
        for node_id, card in payload["cards"].items():
            missing = [k for k in _CARD_REQUIRED if k not in card]
            _check(not missing, f"card {node_id!r}: missing {', '.join(missing)}")
            _check(card["schema_version"] == CARD_SCHEMA_VERSION, f"card {node_id!r}: unsupported schema_version")
            cards[str(node_id)] = card
    return {"name": tree_id, "z": z, "items": items, "labels": labels, "cards": cards,
            "method": linkage.get("method"), "card_set": card_names[0] if card_names else None,
            "active_label_set": entry.get("active_label_set")}


def _existing_documents(con: duckdb.DuckDBPyConnection, item_ids: list[str]) -> list[str]:
    rows = con.execute("SELECT item_id FROM treedb.documents WHERE item_id IN (SELECT unnest(?))",
                       [item_ids]).fetchall()
    return sorted(r[0] for r in rows)


def _write_documents(con: duckdb.DuckDBPyConnection, item_ids: list[str], docs: dict[str, dict[str, Any]]) -> None:
    """One transaction; the items are registered by the tree imports that precede it."""
    rows = [(i, docs[i].get("title"), docs[i]["text"],
             None if docs[i].get("metadata") is None else canonical_json(docs[i]["metadata"])) for i in item_ids]
    con.execute("BEGIN TRANSACTION")
    try:
        con.executemany("INSERT INTO treedb.documents (item_id, title, text, metadata) VALUES (?, ?, ?, ?)", rows)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


@dataclass(frozen=True)
class ParsedArtifact:
    """An Orchard artifact directory, fully read and checked (no store involved).

    ``trees`` entries: ``name`` (tree id), ``z``, ``items`` (leaf order), ``labels``, ``cards``,
    ``method``, ``card_set``, ``active_label_set``.
    """

    root: Path
    manifest: dict[str, Any]
    item_ids: tuple[str, ...]                 # corpus, documents.json order
    documents: dict[str, dict[str, Any]]
    trees: tuple[dict[str, Any], ...]         # manifest order


def read_artifact(path: str | Path) -> ParsedArtifact:
    """Every pre-write check of ``import_orchard_artifact`` that needs no store.

    Raises ``ArtifactError`` for an invalid artifact (schema versions, corpus equality,
    node registry, ingest's own input checks for every tree, duplicate tree ids).
    """
    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    try:
        _check(isinstance(manifest, dict), "manifest.json must be an object")
        _check(manifest.get("schema_version") == ARTIFACT_SCHEMA_VERSION,
               f"unsupported orchard artifact schema: {manifest.get('schema_version')!r}")
        item_ids, docs = _load_documents(root, manifest)
        corpus = set(item_ids)
        trees = [_load_tree(root, entry, corpus) for entry in manifest["trees"]]
        for t in trees:
            # Run ingest's own input checks for every tree BEFORE the first write
            # (name regex, item_id form, Z validity/integrality, INV-11 label and
            # card keys, label value types), so a later tree cannot fail after an
            # earlier one has committed.
            try:
                ingest.validate_cultivar(t["z"], t["items"], t["name"], t["labels"] or None, t["cards"] or None)
            except ingest.IngestError as exc:
                raise ArtifactError(f"{t['name']}: {exc}") from exc
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ArtifactError(f"malformed artifact: {type(exc).__name__}: {exc}") from exc
    names = [t["name"] for t in trees]
    _check(len(names) == len(set(names)), "duplicate tree ids in the manifest")
    return ParsedArtifact(root, manifest, tuple(item_ids), docs, tuple(trees))


def import_orchard_artifact(con: duckdb.DuckDBPyConnection, path: str | Path) -> ArtifactImport:
    """Import every tree of an Orchard artifact directory as a cultivar.

    ``path`` holds ``manifest.json``, ``documents.json`` and ``trees/<id>/``
    (``canonical_tree.json``, ``linkage.json``, ``labels/``, ``cards/``).
    Items are registered by ``ingest.import_cultivar`` (INV-1, INV-5); the tree
    id becomes the cultivar name (INV-2). The artifact's documents are written to
    ``treedb.documents`` (D-027). Raises ``ArtifactError`` (nothing written) for an
    invalid artifact, a cultivar name that already exists, or an item that already
    has a document.

    "Nothing written" covers every failure the pre-write checks can see. A database
    error during the writes is not an ``ArtifactError``: trees are committed one at a
    time and then the documents, so such an error between them could leave the
    earlier trees (and no documents) in the store.
    """
    parsed = read_artifact(path)
    root, manifest, docs = parsed.root, parsed.manifest, parsed.documents
    item_ids, trees = list(parsed.item_ids), list(parsed.trees)
    names = [t["name"] for t in trees]
    existing = {r[0] for r in con.execute("SELECT name FROM treedb.tree_registry").fetchall()}
    clash = sorted(set(names) & existing)
    _check(not clash, f"cultivar name(s) already in the store: {clash}")
    have = _existing_documents(con, item_ids)
    _check(not have, f"{len(have)} item(s) already have a document in the store, e.g. {have[:3]}")

    imported: dict[str, ingest.ImportedCultivar] = {}
    for t in trees:
        source = {"orchard_artifact": root.name, "tree_id": t["name"], "method": t["method"],
                  "active_label_set": t["active_label_set"], "card_set": t["card_set"]}
        try:
            imported[t["name"]] = ingest.import_cultivar(
                con, t["z"], t["items"], t["name"],
                labels=t["labels"] or None, cards=t["cards"] or None, source=source)
        except ingest.IngestError as exc:
            raise ArtifactError(f"{t['name']}: {exc}") from exc
    _write_documents(con, item_ids, docs)
    return ArtifactImport(imported, tuple(item_ids), docs, dict(manifest.get("metadata") or {}), len(docs))


__all__ = ["ArtifactError", "ArtifactImport", "ParsedArtifact", "import_orchard_artifact", "read_artifact"]
