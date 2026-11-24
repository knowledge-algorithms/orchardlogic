"""Bottom-up LLM labeling for one Orchard tree.

Leaves are labeled from their own documents. Each parent is labeled from
its two child cards. One generation is labeled concurrently, then the
next generation waits until child cards exist. The title is stored in
``tree.labels`` and the full card in ``tree.cards``. Linkage is left
unchanged.

``client=None`` builds the OpenAI adapter. Tests inject a fake client.
Failed calls are retried once, then a fallback card may be used when the
failure rate stays below four percent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from orchardlogic.cultivate.exceptions import InvalidIdentityError, LabelingError
from orchardlogic.cultivate.labeling.cards import NodeCard, parse_model_card
from orchardlogic.cultivate.labeling.client import (
    DEFAULT_MAX_WORKERS,
    LabelingClient,
    OpenAILabelingClient,
)
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
from orchardlogic.cultivate.labeling.prompts import (
    format_document_for_prompt,
    render_leaf_prompt,
    render_parent_prompt,
)
from orchardlogic.cultivate.tree import Tree

# Leaf fallback keeps this many characters of the document text.
_FALLBACK_TEXT_CHARS = 240


def label_bottom_up(
    tree: Tree,
    *,
    model: str = DEFAULT_LABELING_MODEL,
    client: LabelingClient | None = None,
    name: str = "llm_bottom_up",
    max_workers: int = DEFAULT_MAX_WORKERS,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, NodeCard]:
    """Write one card per node, leaves first, then each parent generation.

    The default model id is ``gpt-6-luna``. ``name`` is both the label set
    and the card set. Titles go into ``tree.labels[name]`` and become the
    active label set. Full cards go into ``tree.cards[name]``.
    """
    before = tree.linkage_fingerprint()
    model_id = _require_model(model)
    set_name = _require_set_name(name)
    labeling_client = _require_client(client)
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
    node_count = len(tree.nodes)

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

    for layer in _bottom_up_layers(tree):
        items: list[WaveItem] = []
        for request_index, node in enumerate(layer):
            node_id = node["node_id"]
            if node_id in produced:
                continue
            if node["kind"] == "leaf":
                prompt = render_leaf_prompt(_document_block(tree, node["item_id"]))
            else:
                left, right = _require_child_cards(node, produced)
                prompt = render_parent_prompt(left, right)
            items.append(
                WaveItem(
                    request_index=request_index,
                    node_id=node_id,
                    prompt=prompt,
                )
            )
        if not items:
            continue

        def fallback(item: WaveItem) -> NodeCard:
            node = tree.nodes[item.node_id]
            if node["kind"] == "leaf":
                return _leaf_fallback(tree, node)
            left, right = _require_child_cards(node, produced)
            return _parent_fallback(left, right)

        try:
            wave = run_wave(
                labeling_client,
                model=model_id,
                items=items,
                parse_response=card_from_response,
                fallback_for=fallback,
                failure_denominator=node_count,
                max_workers=worker_count,
            )
        except WaveStopped as stopped:
            merge_wave(produced, all_errors, request_indexes, stopped)
            commit(make_active=False)
            raise
        merge_wave(produced, all_errors, request_indexes, wave)
        commit()

    if len(produced) != node_count:
        missing = sorted(set(tree.nodes) - set(produced))
        raise LabelingError(
            f"bottom-up labeling incomplete; missing cards for {missing[:3]}"
        )
    _require_same_linkage(tree, before)
    tree.set_cards(set_name, produced)
    tree.set_labels(
        set_name,
        {node_id: card.title for node_id, card in produced.items()},
    )
    _require_same_linkage(tree, before)
    return dict(tree.cards[set_name])


def card_from_response(text: str) -> NodeCard:
    """Parse one model response into a card."""
    if not isinstance(text, str) or not text.strip():
        raise LabelingError("label card schema mismatch: empty model response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LabelingError(
            "label card schema mismatch: response is not JSON"
        ) from exc
    return parse_model_card(payload)


def _leaf_fallback(tree: Tree, node: Mapping[str, Any]) -> NodeCard:
    item_id = node["item_id"]
    document = tree.documents_by_id.get(item_id)
    if document is None:
        title = item_id
        body = item_id
    else:
        title = document.title.strip() if document.title.strip() else item_id
        body = document.text.strip() if document.text.strip() else title
    summary = body[:_FALLBACK_TEXT_CHARS].strip() or title
    return NodeCard(
        title=title,
        summary=summary,
        scope=(),
        capabilities=(),
        concepts=(),
        uncertainty=(),
    )


def _parent_fallback(left: NodeCard, right: NodeCard) -> NodeCard:
    return NodeCard(
        title=f"{left.title} and {right.title}",
        summary=f"{left.title}. {right.title}.",
        scope=(),
        capabilities=(),
        concepts=(),
        uncertainty=(),
    )


def _require_max_workers(max_workers: int) -> int:
    if type(max_workers) is not int or max_workers < 1:
        raise LabelingError("max_workers must be a positive integer")
    return max_workers


def _checkpoint_path(
    checkpoint_dir: str | Path | None,
    set_name: str,
) -> Path | None:
    if checkpoint_dir is None:
        return None
    return Path(checkpoint_dir) / f"{set_name}.json"


def _require_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise LabelingError("model id must be a non-empty string")
    return model.strip()


def _require_set_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise InvalidIdentityError("label set name must be non-empty")
    return name


def _require_client(client: LabelingClient | None) -> LabelingClient:
    if client is None:
        return OpenAILabelingClient()
    complete = getattr(client, "complete", None)
    if not callable(complete):
        raise LabelingError("client must provide complete(model=, prompt=)")
    return client


def _require_same_linkage(tree: Tree, before: tuple[str, ...]) -> None:
    if tree.linkage_fingerprint() != before:
        raise InvalidIdentityError("bottom-up labeling mutated linkage")


def _document_block(tree: Tree, item_id: str) -> str:
    document = tree.documents_by_id.get(item_id)
    if document is None:
        return format_document_for_prompt(text=item_id, item_id=item_id)
    return format_document_for_prompt(
        text=document.text,
        title=document.title,
        item_id=document.item_id,
    )


def _require_child_cards(
    node: Mapping[str, Any],
    cards: Mapping[str, NodeCard],
) -> tuple[NodeCard, NodeCard]:
    children = node["children"]
    node_id = node["node_id"]
    if len(children) != 2:
        raise LabelingError(f"parent {node_id} must have two child cards")
    left_id, right_id = children
    if left_id not in cards or right_id not in cards:
        raise LabelingError(
            f"parent {node_id} run before both child cards exist"
        )
    return cards[left_id], cards[right_id]


def _bottom_up_layers(tree: Tree) -> list[list[dict[str, Any]]]:
    """Leaf generation, then each generation whose children are already done.

    Inside a generation, internal nodes are ordered by linkage row. A later
    linkage row that is still a shallow merge is labeled before an earlier
    row that already sits higher in the tree.
    """
    leaves = {
        node["item_id"]: node
        for node in tree.nodes.values()
        if node["kind"] == "leaf"
    }
    layers: list[list[dict[str, Any]]] = [
        [leaves[item_id] for item_id in tree.item_ids]
    ]
    pending = {
        node["node_id"]: node
        for node in tree.nodes.values()
        if node["kind"] == "internal"
    }
    done = {node["node_id"] for node in layers[0]}
    while pending:
        layer = [
            node
            for node in pending.values()
            if all(child in done for child in node["children"])
        ]
        if not layer:
            raise LabelingError(
                "parent run before both child cards exist: "
                "an internal node is missing a child"
            )
        layer.sort(key=lambda node: int(node["linkage_row"]))
        layers.append(layer)
        for node in layer:
            done.add(node["node_id"])
            del pending[node["node_id"]]
    return layers
