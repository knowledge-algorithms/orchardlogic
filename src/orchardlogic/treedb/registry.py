"""Perspective / enricher registry and cultivar artifact import (docs/DATA_ORCHARD_V4.md section 5.1, A2-A4).

``import_cultivar_artifact`` does not call ``import_orchard_artifact``. It reads the directory with
``orchard_artifact.read_artifact``, verifies every hash of ``cultivar.json``, imports each tree with
``ingest.import_cultivar`` (own transaction each, depth 0 only), then writes everything else in one
``tx.transaction``. If that second phase fails, a compensating transaction deletes the new trees.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np

from orchardlogic.treedb import cutstore, enrichment, ingest, mutate, tx, vectors
from orchardlogic.treedb import documents as docs_mod
from orchardlogic.treedb.adapters import orchard_artifact
from orchardlogic.treedb.adapters.orchard_artifact import ArtifactError
from orchardlogic.treedb.errors import MutationError
from orchardlogic.treedb.fixtures import canonical_json, item_uuid, node_keys

CULTIVAR_SCHEMA_VERSION = "orchard_cultivar_v1"
PERSPECTIVE_SCHEMA_VERSION = "orchard_perspective_v1"
ENRICHER_SCHEMA_VERSION = "orchard_enricher_v1"
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class RegistryConflictError(ValueError):
    """Same ``name@version`` with a different record id."""


class CultivarIntegrityError(ValueError):
    """A cultivar artifact does not match its ``cultivar.json`` (names the first mismatch)."""


@dataclass(frozen=True)
class CultivarImport:
    artifact_sha256: str
    name: str
    already_imported: bool
    trees: dict[str, uuid.UUID]            # artifact tree id -> treedb tree_registry.id
    tree_names: dict[str, str]             # artifact tree id -> tree_registry.name
    alias: str | None
    documents_written: int = 0
    vectors_written: int = 0
    cut_nodes: int = 0
    imported: dict[str, ingest.ImportedCultivar] = field(default_factory=dict)   # empty when idempotent


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# perspectives and enrichers
# ---------------------------------------------------------------------------

def _put_record(con: duckdb.DuckDBPyConnection, table: str, record: Mapping[str, Any], schema: str) -> str:
    if record.get("schema_version") != schema:
        raise ValueError(f"record schema_version must be {schema!r}, got {record.get('schema_version')!r}")
    name, version = record.get("name"), record.get("version")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ValueError("record needs non-empty string name and version")
    rid = sha256_json(record)
    id_col = "perspective_id" if table == "perspective" else "enricher_id"
    with tx.transaction(con):
        row = con.execute(f"SELECT {id_col} FROM treedb.{table} WHERE name = ? AND version = ?",
                          [name, version]).fetchone()
        if row is not None:
            if row[0] != rid:
                raise RegistryConflictError(
                    f"{table} {name}@{version} already registered with id {row[0][:12]}, new record has {rid[:12]}")
            return rid
        if table == "perspective":
            con.execute("INSERT INTO treedb.perspective (perspective_id, name, version, record) VALUES (?,?,?,?)",
                        [rid, name, version, canonical_json(record)])
        else:
            code = record.get("code") or {}
            con.execute(
                "INSERT INTO treedb.enricher (enricher_id, name, version, scope, method, record, code_sha256, "
                "code_link) VALUES (?,?,?,?,?,?,?,?)",
                [rid, name, version, record.get("scope"), record.get("method"), canonical_json(record),
                 code.get("sha256"), code.get("link")])
    return rid


def put_perspective(con: duckdb.DuckDBPyConnection, record: Mapping[str, Any]) -> str:
    """Register a perspective; returns its id (sha256 of the canonical record). Idempotent."""
    return _put_record(con, "perspective", record, PERSPECTIVE_SCHEMA_VERSION)


def put_enricher(con: duckdb.DuckDBPyConnection, record: Mapping[str, Any]) -> str:
    """Register an enricher record; returns its id. ``scope`` must be item|tree."""
    if record.get("scope") not in ("item", "tree"):
        raise ValueError("enricher scope must be 'item' or 'tree'")
    if record.get("method") not in ("extractive", "statistical", "llm"):
        raise ValueError("enricher method must be extractive, statistical or llm")
    return _put_record(con, "enricher", record, ENRICHER_SCHEMA_VERSION)


def get_perspective(con: duckdb.DuckDBPyConnection, id_or_handle: str) -> dict[str, Any]:
    if _HEX64.match(id_or_handle):
        row = con.execute("SELECT record FROM treedb.perspective WHERE perspective_id = ?", [id_or_handle]).fetchone()
    else:
        name, _, version = id_or_handle.partition("@")
        row = con.execute("SELECT record FROM treedb.perspective WHERE name = ? AND version = ?",
                          [name, version]).fetchone()
    if row is None:
        raise KeyError(f"unknown perspective {id_or_handle!r}")
    return json.loads(row[0])


# ---------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------

def _tree_exists(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID) -> bool:
    return con.execute("SELECT 1 FROM treedb.tree_registry WHERE id = ?", [str(tree_id)]).fetchone() is not None


def set_alias(con: duckdb.DuckDBPyConnection, alias: str, tree_id: uuid.UUID | str) -> None:
    """Point ``alias`` (may contain ``/``) at a tree; replaces an existing alias."""
    if not alias:
        raise ValueError("alias must be non-empty")
    tid = uuid.UUID(str(tree_id))
    with tx.transaction(con):
        if not _tree_exists(con, tid):
            raise KeyError(f"no tree {tid}")
        con.execute("INSERT OR REPLACE INTO treedb.cultivar_alias (alias, tree_id, updated_at) "
                    "VALUES (?, ?, CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP))", [alias, str(tid)])


def resolve(con: duckdb.DuckDBPyConnection, alias_or_name_or_uuid: str | uuid.UUID) -> uuid.UUID:
    """Alias, then tree name, then UUID text. ``KeyError`` when nothing matches."""
    key = alias_or_name_or_uuid
    if isinstance(key, uuid.UUID):
        if _tree_exists(con, key):
            return key
        raise KeyError(f"no tree {key}")
    row = con.execute("SELECT tree_id FROM treedb.cultivar_alias WHERE alias = ?", [key]).fetchone()
    if row is None:
        row = con.execute("SELECT id FROM treedb.tree_registry WHERE name = ?", [key]).fetchone()
    if row is None:
        try:
            u = uuid.UUID(key)
        except ValueError:
            raise KeyError(f"unknown alias, tree name or id {key!r}") from None
        if not _tree_exists(con, u):
            raise KeyError(f"no tree {u}")
        return u
    return uuid.UUID(str(row[0]))


def cultivar_info(con: duckdb.DuckDBPyConnection, alias_or_tree: str | uuid.UUID) -> dict[str, Any]:
    """Tree identity, aliases, the cultivar artifact row, perspective and enricher records."""
    tid = resolve(con, alias_or_tree)
    name = con.execute("SELECT name FROM treedb.tree_registry WHERE id = ?", [str(tid)]).fetchone()[0]
    aliases = [r[0] for r in con.execute("SELECT alias FROM treedb.cultivar_alias WHERE tree_id = ? ORDER BY alias",
                                         [str(tid)]).fetchall()]
    info: dict[str, Any] = {"tree_id": tid, "tree_name": name, "aliases": aliases, "artifact": None,
                            "artifact_tree_id": None, "perspective": None, "enrichers": []}
    row = con.execute(
        "SELECT a.artifact_sha256, a.name, a.schema_version, a.manifest, a.orchard_manifest, a.perspective_id, "
        "a.enricher_ids, a.cultivate_version, a.source_path, a.imported_at, t.artifact_tree_id "
        "FROM treedb.cultivar_tree t JOIN treedb.cultivar_artifact a USING (artifact_sha256) WHERE t.tree_id = ?",
        [str(tid)]).fetchone()
    if row is None:
        return info
    info["artifact"] = {"artifact_sha256": row[0], "name": row[1], "schema_version": row[2],
                        "manifest": json.loads(row[3]), "orchard_manifest": json.loads(row[4]),
                        "perspective_id": row[5], "enricher_ids": list(row[6]), "cultivate_version": row[7],
                        "source_path": row[8], "imported_at": row[9]}
    info["artifact_tree_id"] = row[10]
    info["perspective"] = get_perspective(con, row[5])
    for eid in row[6]:
        rec = con.execute("SELECT record FROM treedb.enricher WHERE enricher_id = ?", [eid]).fetchone()
        info["enrichers"].append(json.loads(rec[0]))
    return info


# ---------------------------------------------------------------------------
# attaching a new leaf under a cultivar node (A4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttachOutcome:
    ok: bool
    reason: str | None = None          # None on success, else a short machine-readable reason
    detail: str | None = None
    node_id: uuid.UUID | None = None   # the new leaf's orchard node id (item UUID) on success
    result: Any = None                 # mutate.AttachResult on success


def node_id_for(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID | str, node_key: str) -> uuid.UUID | None:
    """orchard ``node_id`` of a node_key from ``cultivar_node`` (None when unknown)."""
    row = con.execute("SELECT node_id FROM treedb.cultivar_node WHERE tree_id = ? AND node_key = ?",
                      [str(tree_id), node_key]).fetchone()
    return None if row is None else uuid.UUID(str(row[0]))


def attach_item(con: duckdb.DuckDBPyConnection, tree_id: uuid.UUID | str, parent_key: str, node_key: str,
                item_id: str) -> AttachOutcome:
    """Attach ``item_id`` as a leaf next to cluster ``parent_key`` with ``mutate.attach_node`` (A4).

    Looks up the parent's ``cultivar_node.node_id``, finds its ``treedb.orchard`` row, calls
    ``attach_node(new_edge_ix=1, item_id=...)`` (its own transaction; tx depth must be 0) and on
    success records ``cultivar_node(tree_id, node_key, item uuid)``. Expected refusals (unknown
    node_key, root, mutation refusals, depth != 0) come back as ``ok=False`` with a ``reason``.
    """
    tid = uuid.UUID(str(tree_id))
    if tx.depth(con) != 0:
        return AttachOutcome(False, "inside_transaction", "attach_node needs its own transaction (depth 0)")
    target = node_id_for(con, tid, parent_key)
    if target is None:
        return AttachOutcome(False, "unknown_parent_key", parent_key)
    if node_id_for(con, tid, node_key) is not None:
        return AttachOutcome(False, "node_key_exists", node_key)
    row = con.execute(
        "SELECT o.id FROM treedb.orchard o JOIN treedb.tree_branches b ON b.branch_id = o.branch_id "
        "WHERE b.tree_id = ? AND o.node_id = ? ORDER BY o.id LIMIT 1", [str(tid), str(target)]).fetchone()
    if row is None:
        return AttachOutcome(False, "no_orchard_row", "the node has no edge above it (root?)")
    try:
        res = mutate.attach_node(con, None, row[0], 1, item_id=item_id)
    except (MutationError, LookupError, ValueError) as exc:
        return AttachOutcome(False, type(exc).__name__, str(exc))
    leaf = item_uuid(item_id)
    with tx.transaction(con):
        con.execute("INSERT INTO treedb.cultivar_node (tree_id, node_key, node_id) VALUES (?, ?, ?)",
                    [str(tid), node_key, str(leaf)])
    return AttachOutcome(True, None, None, leaf, res)


# ---------------------------------------------------------------------------
# cultivar artifact import
# ---------------------------------------------------------------------------

def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_cultivar_dir(root: Path) -> dict[str, Any]:
    """Read ``cultivar.json`` and verify every file hash and ``artifact_sha256``.

    Raises ``CultivarIntegrityError`` naming the first mismatch (files are checked in sorted path
    order: missing, changed, then unlisted extras; then ``artifact_sha256``).
    """
    cj = root / "cultivar.json"
    try:
        manifest = json.loads(cj.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CultivarIntegrityError(f"missing cultivar.json in {root}") from None
    except json.JSONDecodeError as exc:
        raise CultivarIntegrityError(f"cultivar.json is not valid JSON: {exc}") from None
    if manifest.get("schema_version") != CULTIVAR_SCHEMA_VERSION:
        raise CultivarIntegrityError(f"unsupported cultivar schema_version: {manifest.get('schema_version')!r}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise CultivarIntegrityError("cultivar.json has no files map")
    for rel in sorted(files):
        p = root / rel
        if not p.is_file():
            raise CultivarIntegrityError(f"{rel}: file listed in cultivar.json is missing")
        if _sha_file(p) != files[rel]:
            raise CultivarIntegrityError(f"{rel}: sha256 mismatch")
    on_disk = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                     if p.is_file() and p.relative_to(root).as_posix() != "cultivar.json")
    for rel in on_disk:
        if rel not in files:
            raise CultivarIntegrityError(f"{rel}: file not listed in cultivar.json")
    if sha256_json(files) != manifest.get("artifact_sha256"):
        raise CultivarIntegrityError("artifact_sha256 does not match the files map")
    return manifest


def _load_json(root: Path, rel: str) -> Any:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _existing(con: duckdb.DuckDBPyConnection, sha: str, alias: str | None) -> CultivarImport:
    name = con.execute("SELECT name FROM treedb.cultivar_artifact WHERE artifact_sha256 = ?", [sha]).fetchone()[0]
    rows = con.execute(
        "SELECT t.artifact_tree_id, t.tree_id, r.name FROM treedb.cultivar_tree t "
        "JOIN treedb.tree_registry r ON r.id = t.tree_id WHERE t.artifact_sha256 = ?", [sha]).fetchall()
    manifest = json.loads(con.execute("SELECT manifest FROM treedb.cultivar_artifact WHERE artifact_sha256 = ?",
                                      [sha]).fetchone()[0])
    pos = {t["tree_id"]: i for i, t in enumerate(manifest.get("trees", []))}
    rows.sort(key=lambda r: pos.get(r[0], 0))
    trees = {r[0]: uuid.UUID(str(r[1])) for r in rows}
    if alias and trees:
        set_alias(con, alias, next(iter(trees.values())))
    return CultivarImport(sha, name, True, trees, {r[0]: r[2] for r in rows}, alias)


def _delete_trees(con: duckdb.DuckDBPyConnection, tree_ids: list[uuid.UUID]) -> None:
    """Compensating delete (A3): every row the tree imports created for these trees."""
    with tx.transaction(con):
        for t in map(str, tree_ids):
            con.execute("DELETE FROM treedb.orchard WHERE branch_id IN "
                        "(SELECT branch_id FROM treedb.tree_branches WHERE tree_id = ?)", [t])
            for table in ("tree_z", "tree_branches", "node_labels", "node_cards", "cultivar_node",
                          "cultivar_tree", "cut_node", "cut", "node_enrichment"):
                con.execute(f"DELETE FROM treedb.{table} WHERE tree_id = ?", [t])
            con.execute("DELETE FROM treedb.tree_registry WHERE id = ?", [t])


def import_cultivar_artifact(con: duckdb.DuckDBPyConnection, path: str | Path, *, tree_name: str | None = None,
                             alias: str | None = None,
                             documents: Literal["refuse", "keep", "replace"] = "keep") -> CultivarImport:
    """Import an ``orchard_cultivar_v1`` directory (see the module docstring and A2-A4).

    ``documents`` controls existing document
    rows: ``refuse`` raises ``ArtifactError`` if any item has one, ``keep`` leaves them, ``replace``
    overwrites. Idempotent on ``artifact_sha256``: a second import returns the existing record
    (``already_imported=True``) and, with ``alias``, repoints the alias at the artifact's first tree.
    ``tree_name`` defaults to ``<cultivar.name>-<artifact_tree_id>-<sha8>``; with a given
    ``tree_name`` and several trees each name gets ``-<artifact_tree_id>`` appended. ``alias`` points at
    the first tree of the manifest. Must be called at transaction depth 0.
    """
    documents_mode = documents
    if documents_mode not in ("refuse", "keep", "replace"):
        raise ValueError("documents must be 'refuse', 'keep' or 'replace'")
    if tx.depth(con) != 0:
        raise RuntimeError("import_cultivar_artifact must be called outside a transaction (depth 0, A3)")
    root = Path(path)
    cv = verify_cultivar_dir(root)
    sha = cv["artifact_sha256"]
    name = cv.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ArtifactError(f"cultivar name {name!r} must match {_NAME_RE.pattern}")
    if con.execute("SELECT 1 FROM treedb.cultivar_artifact WHERE artifact_sha256 = ?", [sha]).fetchone():
        return _existing(con, sha, alias)

    parsed = orchard_artifact.read_artifact(root)
    # declared identity checks
    if sha256_json(cv["perspective"]) != cv.get("perspective_id"):
        raise CultivarIntegrityError("perspective_id does not match the perspective record")
    enrichers = list(cv.get("enrichers") or [])
    for e in enrichers:
        if sha256_json(e["record"]) != e["enricher_id"]:
            raise CultivarIntegrityError(f"enricher_id mismatch for {e['record'].get('name')}@{e['record'].get('version')}")
    declared = cv.get("items") or {}
    if declared.get("count") != len(parsed.item_ids) or declared.get("ids_sha256") != sha256_json(sorted(parsed.item_ids)):
        raise CultivarIntegrityError("items.count / items.ids_sha256 do not match documents.json")
    handle_to_id = {f"{e['record']['name']}@{e['record']['version']}": e["enricher_id"] for e in enrichers}

    # tree names and per-tree payloads, all validated before the first write
    tree_entries = {t["tree_id"]: t for t in cv.get("trees", [])}
    ptrees = {t["name"]: t for t in parsed.trees}
    if set(tree_entries) != set(ptrees):
        raise CultivarIntegrityError("cultivar.json trees differ from manifest.json trees")
    order = [t["name"] for t in parsed.trees]
    names: dict[str, str] = {}
    for tid in order:
        if tree_name is None:
            names[tid] = f"{name}-{tid}-{sha[:8]}"
        else:
            names[tid] = tree_name if len(order) == 1 else f"{tree_name}-{tid}"
    existing = {r[0] for r in con.execute("SELECT name FROM treedb.tree_registry").fetchall()}
    clash = sorted(set(names.values()) & existing)
    if clash:
        raise ArtifactError(f"cultivar name(s) already in the store: {clash}")
    for tid, t in ptrees.items():
        try:
            ingest.validate_cultivar(t["z"], t["items"], names[tid], t["labels"] or None, t["cards"] or None)
        except ingest.IngestError as exc:
            raise ArtifactError(f"{names[tid]}: {exc}") from exc
    have_docs = docs_mod.existing_item_ids(con, list(parsed.item_ids))
    if documents_mode == "refuse" and have_docs:
        raise ArtifactError(f"{len(have_docs)} item(s) already have a document in the store, e.g. {have_docs[:3]}")
    # registry conflicts before any write
    for table, rec in [("perspective", cv["perspective"])] + [("enricher", e["record"]) for e in enrichers]:
        id_col = f"{table}_id"
        row = con.execute(f"SELECT {id_col} FROM treedb.{table} WHERE name = ? AND version = ?",
                          [rec["name"], rec["version"]]).fetchone()
        new_id = sha256_json(rec)
        if row is not None and row[0] != new_id:
            raise RegistryConflictError(f"{table} {rec['name']}@{rec['version']} already registered with another id")

    # payload files (all read before writing)
    vec_payload = []
    for v in cv.get("vectors") or []:
        with np.load(root / v["file"], allow_pickle=False) as z:
            ids, mat, shas = [str(x) for x in z["ids"]], np.asarray(z["vectors"], dtype=np.float32), [str(x) for x in z["sha256"]]
        if mat.ndim != 2 or not (len(ids) == len(shas) == mat.shape[0]) or mat.shape[1] != v.get("dim") \
                or len(ids) != v.get("count"):
            raise CultivarIntegrityError(f"{v['file']}: shape does not match cultivar.json vectors entry")
        vec_payload.append((v["space"], ids, mat, shas))
    item_enr = [(handle_to_id[x["handle"]], _load_json(root, x["file"])["outputs"])
                for x in cv.get("item_enrichment") or []]
    tree_payload: dict[str, dict[str, Any]] = {}
    for tid in order:
        te = tree_entries[tid]
        tree_payload[tid] = {
            "cuts": [_load_json(root, c["file"]) for c in te.get("cuts") or []],
            "node_enr": [(handle_to_id[x["handle"]], _load_json(root, x["file"])["outputs"])
                         for x in te.get("node_enrichment") or []],
        }

    # step 1: trees, each in its own transaction
    created: dict[str, ingest.ImportedCultivar] = {}
    try:
        for tid in order:
            t = ptrees[tid]
            source = {"cultivar_artifact": root.name, "artifact_sha256": sha, "tree_id": tid,
                      "method": t["method"]}
            try:
                created[tid] = ingest.import_cultivar(con, t["z"], t["items"], names[tid],
                                                      labels=t["labels"] or None, cards=t["cards"] or None,
                                                      source=source)
            except ingest.IngestError as exc:
                raise ArtifactError(f"{names[tid]}: {exc}") from exc
        # step 2: everything else, one transaction
        docs_written = vec_written = cut_nodes = 0
        with tx.transaction(con):
            skip = set(have_docs) if documents_mode == "keep" else set()
            to_write = [parsed.documents[i] for i in parsed.item_ids if i not in skip]
            docs_mod.upsert_documents(con, [{"item_id": d["item_id"], "title": d.get("title"),
                                             "text": d["text"], "metadata": d.get("metadata")} for d in to_write])
            docs_written = len(to_write)
            pid = put_perspective(con, cv["perspective"])
            eids = [put_enricher(con, e["record"]) for e in enrichers]
            con.execute(
                "INSERT INTO treedb.cultivar_artifact (artifact_sha256, name, schema_version, manifest, "
                "orchard_manifest, perspective_id, enricher_ids, cultivate_version, source_path) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                [sha, name, CULTIVAR_SCHEMA_VERSION, canonical_json(cv), canonical_json(parsed.manifest), pid, eids,
                 (cv.get("cultivate") or {}).get("version"), str(root)])
            for tid, imp in created.items():
                t = ptrees[tid]
                con.execute("INSERT INTO treedb.cultivar_tree (tree_id, artifact_sha256, artifact_tree_id) "
                            "VALUES (?,?,?)", [str(imp.tree_id), sha, tid])
                keys = node_keys(t["z"], t["items"])
                idx = sorted(keys)
                con.execute(
                    "INSERT INTO treedb.cultivar_node (tree_id, node_key, node_id) "
                    "SELECT ?::UUID, unnest(?::VARCHAR[]), unnest(?::UUID[])",
                    [str(imp.tree_id), [keys[i] for i in idx], [str(imp.node_ids[i]) for i in idx]])
                for cut_doc in tree_payload[tid]["cuts"]:
                    cutstore.put_cut(con, imp.tree_id, cut_doc)
                    cut_nodes += len(cut_doc["nodes"])
                for eid, outs in tree_payload[tid]["node_enr"]:
                    enrichment.put_node_outputs(con, imp.tree_id, eid, outs)
            for space, ids, mat, shas in vec_payload:
                vec_written += vectors.put_vectors(con, space, zip(shas, mat))
                vectors.link_items(con, space, zip(ids, shas))
            for eid, outs in item_enr:
                enrichment.put_item_outputs(con, eid, outs)
            if alias:
                set_alias(con, alias, created[order[0]].tree_id)
    except BaseException as exc:
        if created:
            try:
                _delete_trees(con, [c.tree_id for c in created.values()])
            except BaseException as cleanup:        # keep the original error; report the cleanup failure
                exc.add_note(f"compensating delete of {[str(c.tree_id) for c in created.values()]} failed: "
                             f"{type(cleanup).__name__}: {cleanup}")
        raise
    return CultivarImport(sha, name, False, {t: c.tree_id for t, c in created.items()}, names, alias,
                          docs_written, vec_written, cut_nodes, created)


__all__ = ["AttachOutcome", "attach_item", "node_id_for", "CultivarImport", "CultivarIntegrityError", "RegistryConflictError", "cultivar_info", "get_perspective",
           "import_cultivar_artifact", "put_enricher", "put_perspective", "resolve", "set_alias", "sha256_json",
           "verify_cultivar_dir"]
