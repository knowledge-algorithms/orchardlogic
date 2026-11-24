"""Cultivate: documents + perspective -> ``orchard_cultivar_v1`` directory (spec section 4).

Pipeline: text -> embedding (MiniLM or an injected encoder) -> row-normalise -> Ward linkage ->
``Tree`` -> ``Orchard.save`` (the ``orchard_artifact_v1`` part) -> dynamic cut(s) mapped to cut
nodes (A6) -> item enrichers -> tree enrichers (with ``TreeContext``) -> display -> vectors npz,
enrichment and cut files -> ``cultivar.json`` (written last, with file hashes).

Database import (``cultivate_into_db``) is a later phase; this module never imports treedb.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import re
import shutil
import subprocess
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage

from orchardlogic.cultivate import __version__
from orchardlogic.cultivate.backends.minilm import MiniLMEmbeddingBackend
from orchardlogic.cultivate.cuts import build_dynamic_cut
from orchardlogic.cultivate.document import Document
from orchardlogic.cultivate.enrichers import TreeContext, enricher_id, get_enricher, handle_filename, handle_of
from orchardlogic.cultivate.exceptions import CultivarIntegrityError, InvalidIdentityError
from orchardlogic.cultivate.identity import canonical_json, validate_tree_id
from orchardlogic.cultivate.orchard import Orchard
from orchardlogic.cultivate.perspectives import perspective_id, sha256_json, validate_perspective
from orchardlogic.cultivate.placement import (  # noqa: F401  (re-exported)
    place_items,
    recompute_display,
)
from orchardlogic.cultivate.tree import Tree

CULTIVAR_SCHEMA_VERSION = "orchard_cultivar_v1"
CUT_SCHEMA_VERSION = "orchard_cultivar_cut_v1"
ITEM_ENRICHMENT_SCHEMA_VERSION = "orchard_item_enrichment_v1"
NODE_ENRICHMENT_SCHEMA_VERSION = "orchard_node_enrichment_v1"
CULTIVAR_FILENAME = "cultivar.json"
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

Encoder = Any  # object with .encode(texts) -> ndarray, or a callable texts -> ndarray


# ---------------------------------------------------------------- small helpers


def _utc_now_z() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Deterministic ``.npz`` (fixed zip timestamps; np.savez would embed the clock)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, payload.getvalue())
    return buffer.getvalue()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def as_documents(docs: Sequence[Document | Mapping[str, Any]]) -> list[Document]:
    return [d if isinstance(d, Document) else Document.from_mapping(d) for d in docs]


def embedding_texts(docs: Sequence[Document], perspective: Mapping[str, Any]) -> list[str]:
    """The exact UTF-8 texts embedded (template then ``max_chars``); also the vector cache key."""
    emb = perspective["embedding"]
    template, limit = emb["text_template"], int(emb["max_chars"])
    return [template.format(title=d.title, text=d.text)[:limit] for d in docs]


def text_sha256(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def embed_texts(texts: Sequence[str], encoder: Encoder | None = None) -> np.ndarray:
    """Encode and row-normalise; float32 so that artifact and linkage see identical numbers."""
    if encoder is None:
        encoder = MiniLMEmbeddingBackend()
    encode = encoder.encode if hasattr(encoder, "encode") else encoder
    raw = np.asarray(encode(list(texts)), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != len(texts):
        raise ValueError(f"encoder returned shape {raw.shape} for {len(texts)} texts")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("embedding has a zero or non-finite row")
    return (raw / norms).astype(np.float32)


# ---------------------------------------------------------------- cut -> rows (A6)


def cut_rows(cut: Mapping[str, Any], canonical: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map ``build_dynamic_cut`` output to ``orchard_cultivar_cut_v1`` node rows."""
    canon_nodes = canonical["nodes"]
    rows: list[dict[str, Any]] = []

    def visit(cluster: Mapping[str, Any], parent: dict[str, Any] | None, position: int) -> None:
        key = cluster["canonical_node_id"]
        row = {
            "node_key": key,
            "parent_key": None if parent is None else parent["node_key"],
            "kind": "root" if parent is None else "cluster",
            "depth": 0 if parent is None else parent["depth"] + 1,
            "position": position,
            "item_id": None,
            "descendant_count": int(cluster["descendant_count"]),
            "path": [] if parent is None else [*parent["path"], parent["node_key"]],
        }
        rows.append(row)
        children = cluster["children"]
        for index, child in enumerate(children):
            visit(child, row, index)
        for offset, leaf_key in enumerate(cluster["direct_leaf_node_ids"]):
            rows.append(
                {
                    "node_key": leaf_key,
                    "parent_key": key,
                    "kind": "item",
                    "depth": row["depth"] + 1,
                    "position": len(children) + offset,
                    "item_id": canon_nodes[leaf_key]["item_id"],
                    "descendant_count": 1,
                    "path": [*row["path"], key],
                }
            )

    visit(cut["root"], None, 0)
    return rows


# ---------------------------------------------------------------- cultivate


def _resolve_enrichers(handles: Sequence[str], scope: str) -> list[Any]:
    enrichers = []
    for handle in handles:
        enricher = get_enricher(handle)
        if enricher.record["scope"] != scope:
            raise ValueError(f"{handle} has scope {enricher.record['scope']!r}, expected {scope!r}")
        enrichers.append(enricher)
    return enrichers


def cultivate(
    docs: Sequence[Document | Mapping[str, Any]],
    perspective: Mapping[str, Any],
    *,
    name: str,
    out_dir: str | Path,
    encoder: Encoder | None = None,
    created_at: str | None = None,
) -> Path:
    """Write an ``orchard_cultivar_v1`` directory at ``out_dir`` and return its path.

    ``encoder`` is an object with ``.encode(texts)`` (e.g. ``MiniLMEmbeddingBackend`` with an
    ``encode_fn``) or a plain callable; default is the local MiniLM snapshot. ``created_at`` is
    an ISO ``Z`` string (default now); it is recorded in ``cultivar.json`` only and does not
    influence ``artifact_sha256``. Documents are processed in ``item_id`` order, so the result
    does not depend on input order. ``out_dir`` must be absent or empty.
    """
    perspective = json.loads(canonical_json(perspective))  # detached, JSON-only copy
    validate_perspective(perspective)
    if not _NAME_RE.match(name or ""):
        raise InvalidIdentityError(f"cultivar name {name!r} must match {_NAME_RE.pattern}")
    if len(perspective["features"]) != 1:
        raise NotImplementedError("multi-feature perspectives use the fusion path (not yet wired)")
    root = Path(out_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"cultivar output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    documents = sorted(as_documents(docs), key=lambda d: d.item_id)
    if len(documents) < 3:
        raise ValueError("cultivate needs at least 3 documents")
    item_ids = [d.item_id for d in documents]
    docs_by_id = {d.item_id: d for d in documents}
    tree_id = validate_tree_id(perspective["name"])

    # 1. vectors
    texts = embedding_texts(documents, perspective)
    vectors = embed_texts(texts, encoder)
    content_shas = [text_sha256(t) for t in texts]
    vectors_by_id = {i: vectors[n] for n, i in enumerate(item_ids)}

    # 2. linkage + tree (A5)
    method = perspective["linkage"]["method"]
    z = scipy_linkage(vectors.astype(np.float64), method=method, metric="euclidean")
    tree = Tree.from_linkage(z, item_ids=item_ids, tree_id=tree_id, method=method, documents=documents)
    orchard = Orchard.from_trees(
        documents=documents,
        trees={tree_id: tree},
        metadata={"cultivar": name, "perspective_id": perspective_id(perspective)},
    )
    orchard.save(root)
    canonical = tree.to_canonical_dict()
    linkage_payload = tree.linkage_payload()

    # 3. cuts (A6)
    cuts: dict[str, dict[str, Any]] = {}
    for entry in perspective["cuts"]:
        params = {k: v for k, v in entry.items() if k != "name"}
        built = build_dynamic_cut(canonical, linkage_payload, **params)
        settings = {**entry, "warnings": list(built.get("warnings", []))}
        cuts[entry["name"]] = {
            "schema_version": CUT_SCHEMA_VERSION,
            "tree_id": tree_id,
            "name": entry["name"],
            "settings": settings,
            "nodes": cut_rows(built, canonical),
            "display": {},
        }

    # 4. item enrichers
    item_enrichers = _resolve_enrichers(perspective["item_enrichers"], "item")
    item_outputs: dict[str, dict[str, dict]] = {}
    for enricher in item_enrichers:
        item_outputs[handle_of(enricher.record)] = enricher.enrich(documents)

    # 5. tree enrichers per cut, then display
    tree_enrichers = _resolve_enrichers(perspective["tree_enrichers"], "tree")
    merged_node_outputs: dict[str, dict[str, dict]] = {handle_of(e.record): {} for e in tree_enrichers}
    for cut_name, cut_doc in cuts.items():
        node_outputs: dict[str, dict[str, dict]] = {}
        for enricher in tree_enrichers:
            ctx = TreeContext(
                tree_id=tree_id,
                cut=cut_doc,
                docs=docs_by_id,
                item_outputs=item_outputs,
                node_outputs=node_outputs,
                vectors=vectors_by_id,
                tree=tree,
            )
            handle = handle_of(enricher.record)
            node_outputs[handle] = enricher.enrich(ctx)
            for key, out in node_outputs[handle].items():
                merged_node_outputs[handle].setdefault(key, out)
        cut_doc["display"] = recompute_display(
            cut_doc["nodes"], node_outputs, perspective["display"], root_title=name
        )

    # 6. files
    space = perspective["embedding"]["space"]
    vectors_rel = f"vectors/{space.replace('/', '_')}.npz"
    vectors_path = root / vectors_rel
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    vectors_path.write_bytes(
        _npz_bytes(
            {
                "ids": np.asarray(item_ids),
                "vectors": vectors.astype(np.float32),
                "sha256": np.asarray(content_shas),
            }
        )
    )
    item_files = []
    for handle, outputs in item_outputs.items():
        rel = f"enrichment/items/{handle_filename(handle)}.json"
        _write_json(
            root / rel,
            {"schema_version": ITEM_ENRICHMENT_SCHEMA_VERSION, "handle": handle, "outputs": outputs},
        )
        item_files.append({"handle": handle, "file": rel})
    node_files = []
    for handle, outputs in merged_node_outputs.items():
        rel = f"trees/{tree_id}/enrichment/{handle_filename(handle)}.json"
        _write_json(
            root / rel,
            {
                "schema_version": NODE_ENRICHMENT_SCHEMA_VERSION,
                "handle": handle,
                "tree_id": tree_id,
                "outputs": outputs,
            },
        )
        node_files.append({"handle": handle, "file": rel})
    cut_files = []
    for cut_name, cut_doc in cuts.items():
        rel = f"trees/{tree_id}/cuts/{cut_name}.json"
        _write_json(root / rel, cut_doc)
        cut_files.append({"name": cut_name, "file": rel})

    # 7. cultivar.json, last
    used = [*item_enrichers, *tree_enrichers]
    files = _hash_tree(root)
    manifest = {
        "schema_version": CULTIVAR_SCHEMA_VERSION,
        "name": name,
        "created_at": created_at or _utc_now_z(),
        "cultivate": {"package": "orchard", "version": __version__, "git_commit": _git_commit()},
        "perspective": perspective,
        "perspective_id": perspective_id(perspective),
        "enrichers": [
            {"enricher_id": enricher_id(e.record), "record": e.record} for e in used
        ],
        "items": {"count": len(item_ids), "ids_sha256": sha256_json(sorted(item_ids))},
        "vectors": [
            {"space": space, "file": vectors_rel, "dim": int(vectors.shape[1]), "count": len(item_ids)}
        ],
        "item_enrichment": item_files,
        "trees": [{"tree_id": tree_id, "cuts": cut_files, "node_enrichment": node_files}],
        "files": files,
        "artifact_sha256": sha256_json(files),
    }
    _write_json(root / CULTIVAR_FILENAME, manifest)
    return root


def _hash_tree(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel != CULTIVAR_FILENAME:
            files[rel] = _sha256_bytes(path.read_bytes())
    return files


# ---------------------------------------------------------------- reading / verification


@dataclass
class Cultivar:
    """A verified cultivar directory, parsed. For tests and round trips on the cultivate side."""

    path: Path
    manifest: dict[str, Any]
    item_ids: list[str]
    vector_ids: dict[str, list[str]] = field(default_factory=dict)  # space -> ids
    vectors: dict[str, np.ndarray] = field(default_factory=dict)  # space -> n x dim float32
    vector_sha256: dict[str, list[str]] = field(default_factory=dict)  # space -> row hashes
    item_enrichment: dict[str, dict[str, dict]] = field(default_factory=dict)
    node_enrichment: dict[str, dict[str, dict[str, dict]]] = field(default_factory=dict)
    cuts: dict[str, dict[str, dict]] = field(default_factory=dict)  # tree -> cut name -> doc

    @property
    def artifact_sha256(self) -> str:
        return self.manifest["artifact_sha256"]

    @property
    def perspective(self) -> dict[str, Any]:
        return self.manifest["perspective"]

    def orchard(self) -> Orchard:
        return Orchard.load(self.path)


def verify_cultivar(path: str | Path) -> dict[str, Any]:
    """Check every file hash and return ``cultivar.json``; raises ``CultivarIntegrityError``."""
    root = Path(path)
    manifest_path = root / CULTIVAR_FILENAME
    if not manifest_path.is_file():
        raise CultivarIntegrityError(f"{CULTIVAR_FILENAME} missing in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CULTIVAR_SCHEMA_VERSION:
        raise CultivarIntegrityError(
            f"unsupported cultivar schema_version {manifest.get('schema_version')!r}"
        )
    files: Mapping[str, str] = manifest["files"]
    for rel in sorted(files):
        target = root / rel
        if not target.is_file():
            raise CultivarIntegrityError(f"listed file is missing: {rel}")
        if _sha256_bytes(target.read_bytes()) != files[rel]:
            raise CultivarIntegrityError(f"hash mismatch: {rel}")
    on_disk = set(_hash_tree(root))
    extra = sorted(on_disk - set(files))
    if extra:
        raise CultivarIntegrityError(f"file not listed in cultivar.json: {extra[0]}")
    if sha256_json(dict(files)) != manifest["artifact_sha256"]:
        raise CultivarIntegrityError("artifact_sha256 does not match the file hashes")
    return manifest


def read_cultivar(path: str | Path) -> Cultivar:
    """Verify all hashes, then parse the cultivar's vectors, enrichment and cuts."""
    root = Path(path)
    manifest = verify_cultivar(root)

    def load(rel: str) -> Any:
        return json.loads((root / rel).read_text(encoding="utf-8"))

    cultivar = Cultivar(path=root, manifest=manifest, item_ids=[])
    cultivar.item_ids = [d["item_id"] for d in load("documents.json")]
    for entry in manifest["vectors"]:
        with np.load(root / entry["file"], allow_pickle=False) as archive:
            cultivar.vector_ids[entry["space"]] = archive["ids"].astype(str).tolist()
            cultivar.vectors[entry["space"]] = np.asarray(archive["vectors"], dtype=np.float32)
            cultivar.vector_sha256[entry["space"]] = archive["sha256"].astype(str).tolist()
    for entry in manifest["item_enrichment"]:
        cultivar.item_enrichment[entry["handle"]] = load(entry["file"])["outputs"]
    for tree in manifest["trees"]:
        tid = tree["tree_id"]
        cultivar.node_enrichment[tid] = {
            e["handle"]: load(e["file"])["outputs"] for e in tree["node_enrichment"]
        }
        cultivar.cuts[tid] = {c["name"]: load(c["file"]) for c in tree["cuts"]}
    return cultivar


def cultivate_into_db(*args: Any, **kwargs: Any):
    """Cultivate and import into orchard-db; see ``orchard.cultivate_db`` (needs extra ``db``)."""
    from orchardlogic.cultivate.cultivate_db import cultivate_into_db as impl

    return impl(*args, **kwargs)


def insert_items(*args: Any, **kwargs: Any):
    """Incremental insert into a cultivated orchard-db tree; see ``orchard.cultivate_db``."""
    from orchardlogic.cultivate.cultivate_db import insert_items as impl

    return impl(*args, **kwargs)


def clean_output_dir(path: str | Path) -> None:
    """Remove a previous cultivar directory (used by scripts with an explicit --force)."""
    if Path(path).exists():
        shutil.rmtree(path)


__all__ = [
    "Cultivar",
    "CultivarIntegrityError",
    "cultivate",
    "cultivate_into_db",
    "cut_rows",
    "embed_texts",
    "embedding_texts",
    "insert_items",
    "place_items",
    "read_cultivar",
    "recompute_display",
    "verify_cultivar",
]
