"""Writes a small valid ``orchard_cultivar_v1`` directory without importing ``orchard`` (spec section 4)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree

from orchardlogic.treedb.fixtures import canonical_json, node_keys

SPACE = "test-space@abc12345/mean/tt1"


def sha_json(v) -> str:
    return hashlib.sha256(canonical_json(v).encode("utf-8")).hexdigest()


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")


def perspective(name="topic", version="1") -> dict:
    return {"schema_version": "orchard_perspective_v1", "name": name, "version": version,
            "description": "test", "embedding": {"space": SPACE}, "features": [{"name": "embedding", "weight": 1.0}],
            "linkage": {"method": "ward", "metric": "euclidean"},
            "cuts": [{"name": "topics", "top_criterion": "cluster_count", "cluster_count": 3}],
            "item_enrichers": ["item.extract@1"], "tree_enrichers": ["tree.bottom_up_extractive@1"],
            "display": {"title": ["tree.bottom_up_extractive@1"]}}


def enricher(name, scope, version="1") -> dict:
    return {"schema_version": "orchard_enricher_v1", "name": name, "version": version, "scope": scope,
            "method": "extractive", "params": {}, "code": {"title": name, "description": "", "module": "x",
                                                            "sha256": "0" * 64, "link": None}}


def build_cultivar(out: Path, *, name="vb-topic", n=12, dim=8, seed=1, tweak_perspective=None) -> Path:
    rng = np.random.default_rng(seed)
    items = [f"vb:{i:04d}" for i in range(n)]
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    z = linkage(vecs, method="ward", metric="euclidean").tolist()
    keys = node_keys(z, items)
    base = dt.datetime(2026, 1, 1)
    docs = [{"schema_version": "orchard_document_v1", "item_id": it, "title": f"Title {i}",
             "text": f"body text number {i} about topic {i % 3}",
             "metadata": {"url": f"https://example.com/{i}",
                          "captured_at": (base + dt.timedelta(days=i * 10)).isoformat() + "Z"}}
            for i, it in enumerate(items)]
    root_key = keys[2 * n - 2]
    tree = to_tree(np.array(z))
    nodes = {keys[i]: {"item_id": items[i], "kind": "leaf", "children": []} for i in range(n)}
    nodes.update({keys[i]: {"item_id": None, "kind": "branch"} for i in range(n, 2 * n - 1)})

    cut_nodes = [{"node_key": root_key, "parent_key": None, "kind": "root", "depth": 0, "position": 0,
                  "item_id": None, "descendant_count": n, "path": []}]
    display = {root_key: {"title": "All", "summary": "everything", "tags": ["all"]}}
    pos = 0
    for child in (tree.get_left(), tree.get_right()):
        ck = keys[child.id]
        if child.is_leaf():
            cut_nodes.append({"node_key": ck, "parent_key": root_key, "kind": "item", "depth": 1, "position": pos,
                              "item_id": items[child.id], "descendant_count": 1, "path": [root_key]})
        else:
            cut_nodes.append({"node_key": ck, "parent_key": root_key, "kind": "cluster", "depth": 1,
                              "position": pos, "item_id": None, "descendant_count": child.get_count(),
                              "path": [root_key]})
            display[ck] = {"title": f"Cluster {pos}", "summary": "s", "tags": ["t1", "t2"]}
            for j, leaf in enumerate(child.pre_order()):
                cut_nodes.append({"node_key": keys[leaf], "parent_key": ck, "kind": "item", "depth": 2,
                                  "position": j, "item_id": items[leaf], "descendant_count": 1,
                                  "path": [root_key, ck]})
        pos += 1
    cut = {"schema_version": "orchard_cultivar_cut_v1", "tree_id": "topic", "name": "topics",
           "settings": {"cluster_count": 3, "warnings": []}, "nodes": cut_nodes, "display": display}

    persp = perspective()
    if tweak_perspective:
        persp.update(tweak_perspective)
    e_item, e_tree = enricher("item.extract", "item"), enricher("tree.bottom_up_extractive", "tree")

    shas = [hashlib.sha256(d["text"].encode()).hexdigest() for d in docs]
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "documents.json", docs)
    _dump(out / "trees/topic/canonical_tree.json",
          {"schema_version": "canonical_linkage_tree_v1", "root_node_id": root_key, "nodes": nodes})
    _dump(out / "trees/topic/linkage.json", {"schema_version": "scipy_linkage_artifact_v1", "item_ids": items,
                                              "z_matrix": z, "method": "ward"})
    _dump(out / "manifest.json", {"schema_version": "orchard_artifact_v1", "document_count": n,
                                  "document_schema_version": "orchard_document_v1", "tree_ids": ["topic"],
                                  "trees": [{"tree_id": "topic", "root_node_id": root_key, "leaf_count": n,
                                             "method": "ward", "label_sets": [], "card_sets": []}]})
    _dump(out / "trees/topic/cuts/topics.json", cut)
    _dump(out / "enrichment/items/item.extract__1.json",
          {"schema_version": "orchard_item_enrichment_v1", "handle": "item.extract@1",
           "outputs": {it: {"title": f"T{i}", "summary": "sum", "dates": [], "key_terms": ["a"], "word_count": 7}
                       for i, it in enumerate(items)}})
    _dump(out / "trees/topic/enrichment/tree.bottom_up_extractive__1.json",
          {"schema_version": "orchard_node_enrichment_v1", "handle": "tree.bottom_up_extractive@1",
           "tree_id": "topic", "outputs": {k: {"title": f"N-{i}"} for i, k in enumerate(display)}})
    (out / "vectors").mkdir(exist_ok=True)
    np.savez(out / f"vectors/{SPACE.replace('/', '_')}.npz", ids=np.array(items), vectors=vecs,
             sha256=np.array(shas))

    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*")) if p.is_file()}
    cj = {"schema_version": "orchard_cultivar_v1", "name": name, "created_at": "2026-11-17T03:00:00Z",
          "cultivate": {"package": "orchard", "version": "0.1.0", "git_commit": None},
          "perspective": persp, "perspective_id": sha_json(persp),
          "enrichers": [{"enricher_id": sha_json(e), "record": e} for e in (e_item, e_tree)],
          "items": {"count": n, "ids_sha256": sha_json(sorted(items))},
          "vectors": [{"space": SPACE, "file": f"vectors/{SPACE.replace('/', '_')}.npz", "dim": dim, "count": n}],
          "item_enrichment": [{"handle": "item.extract@1", "file": "enrichment/items/item.extract__1.json"}],
          "trees": [{"tree_id": "topic", "cuts": [{"name": "topics", "file": "trees/topic/cuts/topics.json"}],
                     "node_enrichment": [{"handle": "tree.bottom_up_extractive@1",
                                          "file": "trees/topic/enrichment/tree.bottom_up_extractive__1.json"}]}],
          "files": files, "artifact_sha256": sha_json(files)}
    _dump(out / "cultivar.json", cj)
    return out
