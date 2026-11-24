"""Contrast labeling for one Orchard tree.

Reads a finished bottom-up card set and writes a second set. Each call
rewrites the title and summary so the card is less like nearby cards.
A node with no selected neighbor keeps its source card and is not sent to
the model.

The source card set and the linkage are left unchanged. ``client=None``
builds the OpenAI adapter. Tests inject a fake client.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from orchardlogic.cultivate.exceptions import InvalidIdentityError, LabelingError
from orchardlogic.cultivate.labeling.bottom_up import (
    _bottom_up_layers,
    _require_client,
    _require_max_workers,
    _require_model,
    _require_set_name,
    card_from_response,
)
from orchardlogic.cultivate.labeling.cards import NodeCard
from orchardlogic.cultivate.labeling.client import DEFAULT_MAX_WORKERS, LabelingClient
from orchardlogic.cultivate.labeling.cost import DEFAULT_LABELING_MODEL
from orchardlogic.cultivate.labeling.dispatch import (
    LabelingErrorRecord,
    WaveItem,
    WaveStopped,
    load_checkpoint,
    merge_wave,
    run_wave,
    write_checkpoint,
)
from orchardlogic.cultivate.labeling.prompts import render_contrast_prompt
from orchardlogic.cultivate.tree import Tree

DEFAULT_CONTRAST_LAMBDA = 1.0
DEFAULT_CONTRAST_HEIGHT_CAP = 0.20
DEFAULT_CONTRAST_TOP_K = 8

_HEIGHT_TOLERANCE = 1e-12


def label_contrast(
    tree: Tree,
    *,
    source_set: str = "llm_bottom_up",
    model: str = DEFAULT_LABELING_MODEL,
    client: LabelingClient | None = None,
    name: str = "llm_contrast",
    lambda_value: float = DEFAULT_CONTRAST_LAMBDA,
    height_cap: float = DEFAULT_CONTRAST_HEIGHT_CAP,
    top_k: int = DEFAULT_CONTRAST_TOP_K,
    max_workers: int = DEFAULT_MAX_WORKERS,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, NodeCard]:
    """Rewrite titles and summaries away from nearby nodes.

    ``source_set`` must already hold one card per node. The default model
    id is ``gpt-6-luna``. ``name`` is both the new label set and the new
    card set. It must differ from ``source_set``. Titles go into
    ``tree.labels[name]`` and become the active label set. Full cards go
    into ``tree.cards[name]``.
    """
    before = tree.linkage_fingerprint()
    model_id = _require_model(model)
    set_name = _require_set_name(name)
    source_name = _require_set_name(source_set)
    if set_name == source_name:
        raise LabelingError(
            "contrast name must differ from the source set "
            "so the source cards stay unchanged"
        )
    source = _require_source_cards(tree, source_name)
    neighbors = select_neighbors(
        tree,
        lambda_value=lambda_value,
        height_cap=height_cap,
        top_k=top_k,
    )
    worker_count = _require_max_workers(max_workers)
    checkpoint_path = _checkpoint_path(checkpoint_dir, set_name)
    produced: dict[str, NodeCard] = {}
    request_indexes: dict[str, int] = {}
    all_errors: list[LabelingErrorRecord] = []
    if set_name in tree.cards:
        produced.update(tree.cards[set_name])
    if checkpoint_path is not None:
        loaded = load_checkpoint(checkpoint_path, set_name=set_name)
        produced.update(loaded.cards)
        request_indexes.update(loaded.request_indexes)
        all_errors.extend(loaded.errors)

    for layer in _bottom_up_layers(tree):
        for node in layer:
            node_id = node["node_id"]
            if node_id in produced:
                continue
            if not neighbors[node_id]:
                produced[node_id] = source[node_id]

    items: list[WaveItem] = []
    request_index = 0
    for layer in _bottom_up_layers(tree):
        for node in layer:
            node_id = node["node_id"]
            if node_id in produced:
                continue
            neighbor_ids = neighbors[node_id]
            if not neighbor_ids:
                continue
            items.append(
                WaveItem(
                    request_index=request_index,
                    node_id=node_id,
                    prompt=render_contrast_prompt(
                        source[node_id],
                        [source[neighbor_id] for neighbor_id in neighbor_ids],
                    ),
                )
            )
            request_index += 1

    def commit(*, make_active: bool = True) -> None:
        _require_same_linkage(tree, before)
        tree.set_cards(set_name, produced)
        tree.set_labels(
            set_name,
            {node_id: card.title for node_id, card in produced.items()},
            make_active=make_active,
        )
        _require_same_linkage(tree, before)
        if checkpoint_path is not None:
            write_checkpoint(
                checkpoint_path,
                set_name=set_name,
                cards=produced,
                request_indexes=request_indexes,
                errors=all_errors,
            )

    if items:
        labeling_client = _require_client(client)

        def fallback(item: WaveItem) -> NodeCard:
            return source[item.node_id]

        try:
            wave = run_wave(
                labeling_client,
                model=model_id,
                items=items,
                parse_response=card_from_response,
                fallback_for=fallback,
                failure_denominator=len(items),
                max_workers=worker_count,
            )
        except WaveStopped as stopped:
            merge_wave(produced, all_errors, request_indexes, stopped)
            commit(make_active=False)
            raise
        merge_wave(produced, all_errors, request_indexes, wave)

    if len(produced) != len(tree.nodes):
        missing = sorted(set(tree.nodes) - set(produced))
        raise LabelingError(
            f"contrast labeling incomplete; missing cards for {missing[:3]}"
        )
    commit()
    return dict(tree.cards[set_name])


def _checkpoint_path(
    checkpoint_dir: str | Path | None,
    set_name: str,
) -> Path | None:
    if checkpoint_dir is None:
        return None
    return Path(checkpoint_dir) / f"{set_name}.json"


def select_neighbors(
    tree: Tree,
    *,
    lambda_value: float = DEFAULT_CONTRAST_LAMBDA,
    height_cap: float = DEFAULT_CONTRAST_HEIGHT_CAP,
    top_k: int = DEFAULT_CONTRAST_TOP_K,
) -> dict[str, tuple[str, ...]]:
    """Return each node id and its selected neighbor ids, closest first.

    Every node is present. A node with no neighbor maps to an empty
    tuple and ``label_contrast`` does not call the model for it.
    Insertion order follows the bottom-up generations.
    """
    lam = _require_non_negative(lambda_value, "lambda")
    cap = _require_non_negative(height_cap, "height cap")
    limit = _require_top_k(top_k)
    parent, depths = _parents_and_depths(tree)
    ancestors = _ancestor_sets(parent, tree.nodes)
    heights = _normalized_heights(tree)
    root_id = tree.root_node_id
    root_children = set(tree.nodes[root_id]["children"])
    selected: dict[str, tuple[str, ...]] = {}
    for layer in _bottom_up_layers(tree):
        for node in layer:
            target_id = node["node_id"]
            if target_id == root_id:
                band = cap
            else:
                parent_id = parent[target_id]
                persistence = max(0.0, heights[parent_id] - heights[target_id])
                band = min(cap, lam * persistence)
            candidates: list[tuple[int, float, str, str]] = []
            for candidate_id, candidate in tree.nodes.items():
                if candidate_id == target_id:
                    continue
                is_root_child = target_id == root_id and candidate_id in root_children
                if target_id != root_id and (
                    candidate_id in ancestors[target_id]
                    or target_id in ancestors[candidate_id]
                ):
                    continue
                height_gap = abs(heights[candidate_id] - heights[target_id])
                in_band = height_gap <= band + _HEIGHT_TOLERANCE
                if not in_band and not is_root_child:
                    continue
                candidates.append(
                    (
                        _edge_distance(
                            target_id,
                            candidate_id,
                            ancestors=ancestors,
                            depths=depths,
                        ),
                        height_gap,
                        candidate["membership_hash"],
                        candidate_id,
                    )
                )
            ranked = sorted(candidates)[:limit]
            selected[target_id] = tuple(item[3] for item in ranked)
    return selected


def _require_source_cards(tree: Tree, source_set: str) -> dict[str, NodeCard]:
    if source_set not in tree.cards:
        raise LabelingError(f"contrast source set {source_set!r} is missing")
    source = tree.cards[source_set]
    for node_id in tree.nodes:
        if node_id not in source:
            raise LabelingError(
                f"contrast source set {source_set!r} is missing a card for {node_id}"
            )
    return source


def _require_same_linkage(tree: Tree, before: tuple[str, ...]) -> None:
    if tree.linkage_fingerprint() != before:
        raise InvalidIdentityError("contrast labeling mutated linkage")


def _require_non_negative(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelingError(f"{label} must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise LabelingError(f"{label} must be a non-negative number")
    return number


def _require_top_k(top_k: int) -> int:
    if type(top_k) is not int or top_k < 1:
        raise LabelingError("top_k must be a positive integer")
    return top_k


def _parents_and_depths(tree: Tree) -> tuple[dict[str, str], dict[str, int]]:
    parent: dict[str, str] = {}
    for node in tree.nodes.values():
        for child_id in node["children"]:
            parent[child_id] = node["node_id"]
    depths = {tree.root_node_id: 0}
    stack = [tree.root_node_id]
    while stack:
        node_id = stack.pop()
        for child_id in tree.nodes[node_id]["children"]:
            depths[child_id] = depths[node_id] + 1
            stack.append(child_id)
    return parent, depths


def _ancestor_sets(
    parent: dict[str, str],
    nodes: dict[str, Any],
) -> dict[str, set[str]]:
    ancestors: dict[str, set[str]] = {}
    for node_id in nodes:
        found: set[str] = set()
        current = node_id
        while current in parent:
            current = parent[current]
            found.add(current)
        ancestors[node_id] = found
    return ancestors


def _edge_distance(
    left: str,
    right: str,
    *,
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
) -> int:
    common = (ancestors[left] | {left}) & (ancestors[right] | {right})
    lca = max(common, key=lambda node_id: depths[node_id])
    return depths[left] + depths[right] - 2 * depths[lca]


def _normalized_heights(tree: Tree) -> dict[str, float]:
    root_height = float(tree.nodes[tree.root_node_id]["linkage_distance"])
    scale = root_height if root_height > 0 else 1.0
    return {
        node_id: float(node["linkage_distance"]) / scale
        for node_id, node in tree.nodes.items()
    }
