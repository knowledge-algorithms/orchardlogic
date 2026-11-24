"""Cost estimate for labeling. No card store, client, or network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchardlogic.cultivate import Document, Tree
from orchardlogic.cultivate.exceptions import LabelingError
from orchardlogic.cultivate.labeling.cards import NodeCard
from orchardlogic.cultivate.labeling.cost import (
    DEFAULT_LABELING_MODEL,
    DEFAULT_PRICE_TABLE,
    LEGACY_LABELING_MODEL,
    OUTPUT_TOKENS_PER_CARD,
    SIZING_STAND_IN_CARD,
    ModelPrice,
    estimate_labeling,
)
from orchardlogic.cultivate.labeling.prompts import (
    format_document_for_prompt,
    render_contrast_prompt,
    render_leaf_prompt,
    render_parent_prompt,
)

FIXTURE = Path(__file__).parent / "fixtures" / "labeling_16.json"


def _chain_linkage(n: int) -> list[list[float]]:
    if n < 2:
        return []
    rows: list[list[float]] = [[0, 1, 0.1, 2]]
    for index, leaf in enumerate(range(2, n)):
        rows.append(
            [float(n + index), float(leaf), 0.1 * (index + 2), float(leaf + 1)]
        )
    return rows


def _two_leaf_tree() -> Tree:
    documents = [
        Document(text="Knead dough until smooth.", item_id="bread", title="Dough"),
        Document(text="Patch the inner tube.", item_id="bike", title="Tube"),
    ]
    return Tree.from_linkage(
        [[0, 1, 0.2, 2]],
        item_ids=["bread", "bike"],
        tree_id="labeling",
        documents=documents,
    )


def _fixture_tree() -> Tree:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    documents = [
        Document(
            text=row["text"],
            item_id=row["item_id"],
            title=row["title"],
            source=row["source"],
        )
        for row in rows
    ]
    return Tree.from_linkage(
        _chain_linkage(len(documents)),
        item_ids=[document.item_id for document in documents],
        tree_id="labeling",
        documents=documents,
    )


def _ordered_nodes(tree: Tree) -> list[dict]:
    by_item = {
        node["item_id"]: node
        for node in tree.nodes.values()
        if node["kind"] == "leaf"
    }
    internals = [node for node in tree.nodes.values() if node["kind"] == "internal"]
    internals.sort(key=lambda node: int(node["linkage_row"]))
    return [by_item[item_id] for item_id in tree.item_ids] + internals


def _card_for(cards: dict[str, NodeCard] | None, node_id: str) -> NodeCard:
    if cards is not None and node_id in cards:
        return cards[node_id]
    return SIZING_STAND_IN_CARD


def _expected_bottom_up_characters(
    tree: Tree,
    cards: dict[str, NodeCard] | None = None,
) -> int:
    total = 0
    for node in _ordered_nodes(tree):
        if node["kind"] == "leaf":
            document = tree.documents_by_id[node["item_id"]]
            block = format_document_for_prompt(
                text=document.text,
                title=document.title,
                item_id=document.item_id,
            )
            total += render_leaf_prompt(block).character_count
            continue
        left_id, right_id = node["children"]
        total += render_parent_prompt(
            _card_for(cards, left_id),
            _card_for(cards, right_id),
        ).character_count
    return total


def _expected_contrast_characters(
    tree: Tree,
    neighbor_counts: dict[str, int],
    cards: dict[str, NodeCard] | None = None,
) -> int:
    total = 0
    for node in _ordered_nodes(tree):
        count = neighbor_counts.get(node["node_id"], 0)
        if count < 1:
            continue
        neighbors = (SIZING_STAND_IN_CARD,) * count
        total += render_contrast_prompt(
            _card_for(cards, node["node_id"]), neighbors
        ).character_count
    return total


def test_bottom_up_counts_and_token_formula() -> None:
    tree = _two_leaf_tree()
    before = tree.linkage_fingerprint()
    estimate = estimate_labeling(tree, model=LEGACY_LABELING_MODEL)
    assert tree.linkage_fingerprint() == before
    assert estimate.model == LEGACY_LABELING_MODEL
    assert estimate.mode == "bottom_up"
    assert estimate.leaf_count == 2
    assert estimate.internal_node_count == 1
    assert estimate.node_count == 3
    assert estimate.bottom_up_calls == 3
    assert estimate.contrast_calls == 0
    assert estimate.call_count == 3
    assert estimate.input_characters == _expected_bottom_up_characters(tree)
    assert estimate.input_tokens == estimate.input_characters // 4
    assert estimate.output_tokens_per_card == OUTPUT_TOKENS_PER_CARD == 256
    assert estimate.output_tokens == 3 * 256
    price = DEFAULT_PRICE_TABLE[LEGACY_LABELING_MODEL]
    assert price.input_usd_per_million == 0.40
    assert price.output_usd_per_million == 1.60
    assert price.as_of == "2026-07-25"
    assert price.is_estimate is True
    assert estimate.price_as_of == "2026-07-25"
    assert estimate.price_is_estimate is True
    assert estimate.price_label == "gpt-4.1-mini estimate as of 2026-07-25"
    assert estimate.input_usd == estimate.input_tokens * 0.40 / 1_000_000
    assert estimate.output_usd == estimate.output_tokens * 1.60 / 1_000_000
    assert estimate.usd == estimate.input_usd + estimate.output_usd


def test_serialized_cards_change_the_parent_estimate() -> None:
    tree = _two_leaf_tree()
    leaves = [node for node in tree.nodes.values() if node["kind"] == "leaf"]
    short = NodeCard(
        title="Bread",
        summary="Bake loaves.",
        scope=("loaves",),
        capabilities=("bake",),
        concepts=("dough",),
        uncertainty=(),
    )
    long = NodeCard(
        title="Bread",
        summary=(
            "Bake loaves in a hot oven until the crust browns and the crumb sets."
        ),
        scope=("home loaves", "hot oven"),
        capabilities=("bake until the crust browns",),
        concepts=("dough", "oven spring"),
        uncertainty=("final crumb temperature",),
    )
    short_cards = {node["node_id"]: short for node in leaves}
    long_cards = {node["node_id"]: long for node in leaves}
    short_estimate = estimate_labeling(
        tree, cards=short_cards, model=LEGACY_LABELING_MODEL
    )
    long_estimate = estimate_labeling(
        tree, cards=long_cards, model=LEGACY_LABELING_MODEL
    )
    assert long_estimate.input_characters > short_estimate.input_characters
    assert short_estimate.input_characters == _expected_bottom_up_characters(
        tree, short_cards
    )
    assert short_estimate.bottom_up_calls == long_estimate.bottom_up_calls == 3


def test_contrast_uses_neighbor_counts_and_includes_the_root() -> None:
    tree = _two_leaf_tree()
    root = tree.root_node_id
    leaf_id = next(
        node["node_id"] for node in tree.nodes.values() if node["kind"] == "leaf"
    )
    root_only = {root: 2, leaf_id: 0}
    with_root = estimate_labeling(
        tree,
        mode="contrast",
        neighbor_counts=root_only,
        model=LEGACY_LABELING_MODEL,
    )
    assert with_root.bottom_up_calls == 0
    assert with_root.contrast_calls == 1
    assert with_root.call_count == 1
    assert with_root.leaf_count == 2
    assert with_root.input_characters == _expected_contrast_characters(tree, root_only)

    leaf_only = {root: 0, leaf_id: 1}
    without_root = estimate_labeling(
        tree,
        mode="contrast",
        neighbor_counts=leaf_only,
        model=LEGACY_LABELING_MODEL,
    )
    assert without_root.contrast_calls == 1
    assert without_root.input_characters == _expected_contrast_characters(
        tree, leaf_only
    )

    both_nodes = {root: 1, leaf_id: 3}
    included = estimate_labeling(
        tree,
        mode="contrast",
        neighbor_counts=both_nodes,
        model=LEGACY_LABELING_MODEL,
    )
    assert included.contrast_calls == 2
    assert included.input_characters > with_root.input_characters
    one_neighbor = estimate_labeling(
        tree,
        mode="contrast",
        neighbor_counts={root: 1},
        model=LEGACY_LABELING_MODEL,
    )
    two_neighbors = estimate_labeling(
        tree,
        mode="contrast",
        neighbor_counts={root: 2},
        model=LEGACY_LABELING_MODEL,
    )
    assert one_neighbor.contrast_calls == two_neighbors.contrast_calls == 1
    assert two_neighbors.input_characters > one_neighbor.input_characters


def test_both_mode_adds_the_two_passes() -> None:
    tree = _two_leaf_tree()
    counts = {tree.root_node_id: 1}
    bottom = estimate_labeling(tree, mode="bottom_up", model=LEGACY_LABELING_MODEL)
    contrast = estimate_labeling(
        tree,
        mode="contrast",
        neighbor_counts=counts,
        model=LEGACY_LABELING_MODEL,
    )
    both = estimate_labeling(
        tree,
        mode="both",
        neighbor_counts=counts,
        model=LEGACY_LABELING_MODEL,
    )
    assert both.bottom_up_calls == bottom.bottom_up_calls == 3
    assert both.contrast_calls == contrast.contrast_calls == 1
    assert both.call_count == 4
    assert both.input_characters == (
        bottom.input_characters + contrast.input_characters
    )
    assert both.input_tokens == both.input_characters // 4
    assert both.output_tokens == both.call_count * OUTPUT_TOKENS_PER_CARD
    assert both.usd == both.input_usd + both.output_usd


def test_caller_can_pin_another_model_and_price() -> None:
    tree = _two_leaf_tree()
    price = ModelPrice(
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
        as_of="2026-01-01",
        label="test model rates",
        is_estimate=False,
    )
    estimate = estimate_labeling(
        tree,
        model="other-model-2026",
        price_table={"other-model-2026": price},
    )
    assert estimate.model == "other-model-2026"
    assert estimate.price_as_of == "2026-01-01"
    assert estimate.price_is_estimate is False
    assert estimate.input_usd == estimate.input_tokens * 1.0 / 1_000_000
    assert estimate.output_usd == estimate.output_tokens * 2.0 / 1_000_000


def test_fixture_tree_bottom_up_calls_every_node() -> None:
    tree = _fixture_tree()
    before = tree.linkage_fingerprint()
    estimate = estimate_labeling(tree, mode="bottom_up", model=LEGACY_LABELING_MODEL)
    assert tree.linkage_fingerprint() == before
    assert estimate.leaf_count == 16
    assert estimate.internal_node_count == 15
    assert estimate.node_count == 31
    assert estimate.bottom_up_calls == 31
    assert estimate.contrast_calls == 0
    assert estimate.input_characters == _expected_bottom_up_characters(tree)
    assert estimate.input_tokens == estimate.input_characters // 4

    everyone = {node_id: 1 for node_id in tree.nodes}
    contrast = estimate_labeling(
        tree,
        mode="contrast",
        neighbor_counts=everyone,
        model=LEGACY_LABELING_MODEL,
    )
    assert contrast.contrast_calls == 31
    assert tree.root_node_id in everyone

    everyone[tree.root_node_id] = 0
    without_root = estimate_labeling(
        tree,
        mode="contrast",
        neighbor_counts=everyone,
        model=LEGACY_LABELING_MODEL,
    )
    assert without_root.contrast_calls == 30


def test_single_leaf_is_one_bottom_up_call() -> None:
    document = Document(text="Empty the rain gauge.", item_id="rain", title="Gauge")
    tree = Tree.from_linkage(
        [],
        item_ids=["rain"],
        tree_id="labeling",
        documents=[document],
    )
    estimate = estimate_labeling(tree, model=LEGACY_LABELING_MODEL)
    assert estimate.leaf_count == 1
    assert estimate.internal_node_count == 0
    assert estimate.bottom_up_calls == 1
    assert estimate.input_characters == render_leaf_prompt(
        format_document_for_prompt(
            text=document.text,
            title=document.title,
            item_id=document.item_id,
        )
    ).character_count


def test_default_model_uses_the_shipped_luna_rate() -> None:
    tree = _two_leaf_tree()
    estimate = estimate_labeling(tree)
    assert estimate.model == DEFAULT_LABELING_MODEL
    price = DEFAULT_PRICE_TABLE[DEFAULT_LABELING_MODEL]
    assert price.input_usd_per_million == 0.10
    assert price.output_usd_per_million == 0.50
    assert price.as_of == "2026-10-23"
    assert price.is_estimate is False
    assert price.label == "gpt-6-luna standard short-context as of 2026-10-23"
    assert estimate.price_as_of == "2026-10-23"
    assert estimate.price_is_estimate is False
    assert estimate.price_label == price.label
    assert estimate.input_usd == estimate.input_tokens * 0.10 / 1_000_000
    assert estimate.output_usd == estimate.output_tokens * 0.50 / 1_000_000
    assert estimate.usd == estimate.input_usd + estimate.output_usd


def test_estimate_rejects_bad_mode_price_and_neighbors() -> None:
    tree = _two_leaf_tree()
    with pytest.raises(LabelingError, match="mode"):
        estimate_labeling(tree, mode="batch")
    with pytest.raises(LabelingError, match="no price"):
        estimate_labeling(tree, model="gpt-4.1")
    legacy = LEGACY_LABELING_MODEL
    with pytest.raises(LabelingError, match="neighbor_counts"):
        estimate_labeling(tree, mode="contrast", model=legacy)
    with pytest.raises(LabelingError, match="neighbor_counts"):
        estimate_labeling(tree, mode="both", model=legacy)
    with pytest.raises(LabelingError, match="unknown node"):
        estimate_labeling(
            tree,
            mode="contrast",
            neighbor_counts={"missing": 1},
            model=legacy,
        )
    with pytest.raises(LabelingError, match="integer"):
        estimate_labeling(
            tree,
            mode="contrast",
            neighbor_counts={tree.root_node_id: True},
            model=legacy,
        )
    with pytest.raises(LabelingError, match=">= 0"):
        estimate_labeling(
            tree,
            mode="contrast",
            neighbor_counts={tree.root_node_id: -1},
            model=legacy,
        )
