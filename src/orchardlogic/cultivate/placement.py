"""Pure placement and display recompute used by cultivation and incremental insert (spec 6.1).

No database, no model: both functions work on cut rows (``orchard_cultivar_cut_v1`` "nodes").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from orchardlogic.cultivate.enrichers import text as T

LEAF_KEY_PREFIX = "item:"


def item_row_key(item_id: str) -> str:
    """node_key of a newly placed item row (cultivation items use the canonical leaf id)."""
    return f"{LEAF_KEY_PREFIX}{item_id}"


def _unit(vector: np.ndarray) -> np.ndarray | None:
    vec = np.asarray(vector, dtype=np.float64).ravel()
    norm = np.linalg.norm(vec)
    return None if norm == 0 or not np.isfinite(norm) else vec / norm


def cluster_centroids(
    cut_nodes: Sequence[Mapping[str, Any]],
    item_vectors_by_id: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Mean of the stored item vectors beneath each root/cluster node (unnormalised mean)."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for node in cut_nodes:
        if node["kind"] != "item" or node["item_id"] not in item_vectors_by_id:
            continue
        vec = np.asarray(item_vectors_by_id[node["item_id"]], dtype=np.float64)
        for anc in node["path"]:
            sums[anc] = sums[anc] + vec if anc in sums else vec.copy()
            counts[anc] = counts.get(anc, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


PLACEMENT_STRATEGIES = ("knn", "centroid")
KNN_K = 5
KNN_MIN_SIMILARITY = 0.3


def place_items(
    cut_nodes: Sequence[Mapping[str, Any]],
    item_vectors_by_id: Mapping[str, np.ndarray],
    new_vectors: Mapping[str, np.ndarray],
    *,
    strategy: str = "centroid",
    k: int = KNN_K,
    min_similarity: float = KNN_MIN_SIMILARITY,
) -> list[dict[str, Any]]:
    """Place new items; ``strategy`` is ``"centroid"`` (greedy descent) or ``"knn"``.

    ``knn``: each item's ``k`` nearest existing items (within the top-level cluster that the
    centroid descent picks) vote for their parent cluster, weighted
    by cosine similarity; the winner is the parent. If the best similarity is below
    ``min_similarity`` the centroid descent decides. Result keys are as for ``centroid``
    plus ``method`` (``knn`` or ``centroid``).
    """
    if strategy not in PLACEMENT_STRATEGIES:
        raise ValueError(f"placement strategy must be one of {PLACEMENT_STRATEGIES}")
    base = _place_centroid(cut_nodes, item_vectors_by_id, new_vectors)
    if strategy == "centroid":
        return base
    nodes = {n["node_key"]: n for n in cut_nodes}
    parent_of = {n["item_id"]: n["parent_key"] for n in cut_nodes if n["kind"] == "item"}
    ids = [i for i in item_vectors_by_id if i in parent_of]
    if not ids:
        return base
    units = [_unit(item_vectors_by_id[i]) for i in ids]
    keep = [n for n, u in enumerate(units) if u is not None]
    ids = [ids[n] for n in keep]
    matrix = np.vstack([units[n] for n in keep])
    centroids = {
        key: unit
        for key, vec in cluster_centroids(cut_nodes, item_vectors_by_id).items()
        if (unit := _unit(vec)) is not None
    }
    top_of = np.array(
        [
            ([*nodes[parent_of[i]]["path"], parent_of[i]] + [None, None])[1]
            for i in ids
        ],
        dtype=object,
    )
    for placement in base:
        vec = _unit(new_vectors[placement["item_id"]])
        sims = matrix @ vec
        # neighbours only vote inside the top-level cluster the centroid descent chose
        # (global kNN improved leaf agreement but lost top-level agreement with a rebuild)
        top = placement["path"][1] if len(placement["path"]) > 1 else None
        allowed = np.flatnonzero(top_of == top) if top is not None else np.arange(len(ids))
        if allowed.size == 0:
            allowed = np.arange(len(ids))
        order = allowed[np.argsort(-sims[allowed], kind="stable")[: max(1, k)]]
        if float(sims[order[0]]) < min_similarity:
            placement["method"] = "centroid"
            continue
        votes: dict[str, float] = {}
        for idx in order:
            parent = parent_of[ids[idx]]
            votes[parent] = votes.get(parent, 0.0) + max(float(sims[idx]), 0.0)
        winner = min(votes, key=lambda key: (-round(votes[key], 12), key))
        placement["parent_key"] = winner
        placement["path"] = [*nodes[winner]["path"], winner]
        placement["similarity"] = float(centroids[winner] @ vec) if winner in centroids else None
        placement["method"] = "knn"
    return base


def _place_centroid(
    cut_nodes: Sequence[Mapping[str, Any]],
    item_vectors_by_id: Mapping[str, np.ndarray],
    new_vectors: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Greedy descent by centroid cosine (spec 6.1).

    Starting at the root, each new item moves to the child *cluster* whose centroid has the
    highest cosine similarity with the item vector, until the current node has no child
    clusters. Returns one dict per new item, in ``new_vectors`` order:

    - ``item_id``
    - ``node_key``: the new item row's key (``item:<item_id>``)
    - ``parent_key``: the cluster (or root) the item joins
    - ``path``: ancestor node_keys from the root to ``parent_key`` inclusive (the cut-row path
      convention for a row whose parent is ``parent_key``)
    - ``similarity``: cosine between the item and ``parent_key``'s centroid (``None`` when the
      item is placed directly under a root without clusters)
    """
    nodes = {n["node_key"]: n for n in cut_nodes}
    roots = [n for n in cut_nodes if n["kind"] == "root"]
    if len(roots) != 1:
        raise ValueError("cut must have exactly one root node")
    root = roots[0]["node_key"]
    centroids = {
        key: unit
        for key, vec in cluster_centroids(cut_nodes, item_vectors_by_id).items()
        if (unit := _unit(vec)) is not None
    }
    child_clusters: dict[str, list[str]] = {}
    for node in sorted(cut_nodes, key=lambda n: (n["depth"], n["position"])):
        if node["kind"] == "cluster":
            child_clusters.setdefault(node["parent_key"], []).append(node["node_key"])

    placements: list[dict[str, Any]] = []
    for item_id, raw in new_vectors.items():
        vec = _unit(raw)
        if vec is None:
            raise ValueError(f"item {item_id!r} has a zero or non-finite vector")
        current = root
        similarity: float | None = None
        while True:
            options = [k for k in child_clusters.get(current, []) if k in centroids]
            if not options:
                break
            scored = [(float(centroids[k] @ vec), k) for k in options]
            # highest similarity; ties go to the lexicographically smaller key
            similarity, current = min(scored, key=lambda s: (-round(s[0], 12), s[1]))
        placements.append(
            {
                "item_id": item_id,
                "node_key": item_row_key(item_id),
                "parent_key": current,
                "path": [*nodes[current]["path"], current],
                "similarity": similarity,
                "method": "centroid",
            }
        )
    return placements


def recompute_display(
    cut_nodes: Sequence[Mapping[str, Any]],
    node_outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    display_spec: Mapping[str, Sequence[str]],
    *,
    only: Sequence[str] | frozenset[str] | None = None,
    existing: Mapping[str, Mapping[str, Any]] | None = None,
    root_title: str = "All items",
) -> dict[str, dict[str, Any]]:
    """Fill ``{title, summary, tags}`` per root/cluster node using the perspective ``display``.

    For each field the first non-empty value across ``display_spec[field]`` handles wins
    (``node_outputs[handle][node_key][field]``). Cluster titles are made unique among siblings;
    siblings outside ``only`` keep their ``existing`` title and take precedence. Returns display
    for ``only`` when given, otherwise for every root/cluster node.
    """
    existing = existing or {}
    nodes = {n["node_key"]: n for n in cut_nodes if n["kind"] != "item"}
    wanted = [
        k
        for k, n in sorted(nodes.items(), key=lambda kv: (kv[1]["depth"], kv[1]["position"]))
        if only is None or k in only
    ]

    def pick(field: str, key: str) -> Any:
        for handle in display_spec.get(field, ()):
            value = node_outputs.get(handle, {}).get(key, {}).get(field)
            if value:
                return value
        return None

    result: dict[str, dict[str, Any]] = {}
    titles: dict[str, str] = {}
    candidates: dict[str, list[str]] = {}
    for key in wanted:
        title = pick("title", key) or ""
        terms: list[str] = []
        for outputs in node_outputs.values():
            node_out = outputs.get(key, {})
            terms += [t for t in node_out.get("distinct_terms", ()) if t not in terms]
            terms += [t for t in node_out.get("key_terms", ()) if t not in terms]
        if not title and terms:
            title = T.title_from_terms(terms[:3])
        if not title:
            title = root_title if nodes[key]["kind"] == "root" else "Cluster"
        titles[key] = title
        candidates[key] = terms
        result[key] = {
            "title": title,
            "summary": pick("summary", key) or "",
            "tags": list(pick("tags", key) or []),
        }
    groups: dict[str | None, list[str]] = {}
    for key in nodes:
        groups.setdefault(nodes[key].get("parent_key"), []).append(key)
    fixed = {
        k: existing[k]["title"]
        for k in nodes
        if k not in result and k in existing and existing[k].get("title")
    }
    ordered_groups = []
    for kids in groups.values():
        kids = sorted(kids, key=lambda k: nodes[k]["position"])
        ordered_groups.append([k for k in kids if k in fixed] + [k for k in kids if k in titles])
    unique = T.dedupe_sibling_titles({**fixed, **titles}, ordered_groups, candidates)
    for key in wanted:
        result[key]["title"] = unique[key]
    return result
