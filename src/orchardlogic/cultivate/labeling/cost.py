"""Pure labeling cost estimate.

No network, no model client, and no card store. Contrast neighbor selection
is not implemented: the caller passes ``neighbor_counts`` as the stand-in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from orchardlogic.cultivate.exceptions import LabelingError
from orchardlogic.cultivate.labeling.cards import NodeCard
from orchardlogic.cultivate.labeling.client import LabelingPrompt
from orchardlogic.cultivate.labeling.prompts import (
    format_document_for_prompt,
    render_contrast_prompt,
    render_leaf_prompt,
    render_parent_prompt,
)
from orchardlogic.cultivate.tree import Tree

DEFAULT_LABELING_MODEL = "gpt-6-luna"
LEGACY_LABELING_MODEL = "gpt-4.1-mini-2025-04-14"
GPT_6_LUNA_PRICE_AS_OF = "2026-10-23"
GPT_6_LUNA_INPUT_USD_PER_MILLION = 0.10
GPT_6_LUNA_OUTPUT_USD_PER_MILLION = 0.50
GPT_4_1_MINI_PRICE_AS_OF = "2026-07-25"
GPT_4_1_MINI_INPUT_USD_PER_MILLION = 0.40
GPT_4_1_MINI_OUTPUT_USD_PER_MILLION = 1.60

# Fixed completion budget per card. This is a reserve for the estimate,
# not a measured model response.
OUTPUT_TOKENS_PER_CARD = 256

_MODES = ("bottom_up", "contrast", "both")


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Per-million-token rates used by the estimate.

    ``is_estimate`` is false for the published gpt-6-luna list price and
    true for the shipped gpt-4.1-mini table. ``as_of`` is the date that
    rate was written down. There is no network price lookup.
    """

    input_usd_per_million: float
    output_usd_per_million: float
    as_of: str
    label: str
    is_estimate: bool = True


DEFAULT_PRICE_TABLE: Mapping[str, ModelPrice] = MappingProxyType(
    {
        DEFAULT_LABELING_MODEL: ModelPrice(
            input_usd_per_million=GPT_6_LUNA_INPUT_USD_PER_MILLION,
            output_usd_per_million=GPT_6_LUNA_OUTPUT_USD_PER_MILLION,
            as_of=GPT_6_LUNA_PRICE_AS_OF,
            label="gpt-6-luna standard short-context as of 2026-10-23",
            is_estimate=False,
        ),
        LEGACY_LABELING_MODEL: ModelPrice(
            input_usd_per_million=GPT_4_1_MINI_INPUT_USD_PER_MILLION,
            output_usd_per_million=GPT_4_1_MINI_OUTPUT_USD_PER_MILLION,
            as_of=GPT_4_1_MINI_PRICE_AS_OF,
            label="gpt-4.1-mini estimate as of 2026-07-25",
            is_estimate=True,
        ),
    }
)

# Used when a node has no card yet. Same object for every missing card so
# the character count is reproducible. Tests can pass real cards instead.
SIZING_STAND_IN_CARD = NodeCard(
    title="Sizing stand-in",
    summary="Placeholder card used only to estimate prompt length.",
    scope=("stand-in scope",),
    capabilities=("stand-in capability",),
    concepts=("stand-in concept",),
    uncertainty=(),
)


@dataclass(frozen=True, slots=True)
class LabelingEstimate:
    """Node counts, call counts, token estimate, and dollars."""

    model: str
    mode: str
    leaf_count: int
    internal_node_count: int
    node_count: int
    bottom_up_calls: int
    contrast_calls: int
    call_count: int
    input_characters: int
    input_tokens: int
    output_tokens_per_card: int
    output_tokens: int
    input_usd: float
    output_usd: float
    usd: float
    price_as_of: str
    price_is_estimate: bool
    price_label: str


def estimate_labeling(
    tree: Tree,
    mode: str = "bottom_up",
    model: str = DEFAULT_LABELING_MODEL,
    *,
    price_table: Mapping[str, ModelPrice] | None = None,
    neighbor_counts: Mapping[str, int] | None = None,
    cards: Mapping[str, NodeCard] | None = None,
) -> LabelingEstimate:
    """Estimate labeling cost for one tree.

    ``mode`` is ``bottom_up``, ``contrast``, or ``both``. Bottom-up calls
    equal leaves plus internal nodes. Contrast calls equal nodes whose
    ``neighbor_counts`` value is at least one. The root is included when
    its count is at least one. ``neighbor_counts`` is the stand-in until
    neighbor selection exists; contrast and both require it.

    Input tokens are ``len(rendered prompt) // 4``. Each rendered prompt
    includes the leaf document or the serialized cards. Output tokens are
    ``OUTPUT_TOKENS_PER_CARD`` times the call count. Dollars come from
    ``price_table``. The default model id is ``gpt-6-luna``. Its shipped
    rate is the published standard short-context price as of 2026-10-23
    ($0.10 input, $0.50 output per million tokens). Batch and fast rates
    are not used. Pass ``price_table`` or ``model=`` to price another id.
    """
    if mode not in _MODES:
        raise LabelingError(
            "labeling mode must be bottom_up, contrast, or both; "
            f"got {mode!r}"
        )
    if not isinstance(model, str) or not model.strip():
        raise LabelingError("model id must be a non-empty string")
    model_id = model.strip()
    price = _price_for(model_id, price_table)
    resolved_cards = _validate_cards(tree, cards)

    bottom_prompts: list[LabelingPrompt] = []
    contrast_prompts: list[LabelingPrompt] = []
    if mode in {"bottom_up", "both"}:
        bottom_prompts = _bottom_up_prompts(tree, resolved_cards)
    if mode in {"contrast", "both"}:
        if neighbor_counts is None:
            raise LabelingError(
                "contrast estimate needs neighbor_counts; "
                "neighbor selection is not part of this phase"
            )
        contrast_prompts = _contrast_prompts(tree, resolved_cards, neighbor_counts)

    ordered = _ordered_nodes(tree)
    leaf_count = sum(1 for node in ordered if node["kind"] == "leaf")
    internal_node_count = sum(1 for node in ordered if node["kind"] == "internal")
    bottom_up_calls = len(bottom_prompts)
    contrast_calls = len(contrast_prompts)
    call_count = bottom_up_calls + contrast_calls
    input_characters = sum(
        prompt.character_count for prompt in bottom_prompts + contrast_prompts
    )
    input_tokens = input_characters // 4
    output_tokens = call_count * OUTPUT_TOKENS_PER_CARD
    input_usd = input_tokens * price.input_usd_per_million / 1_000_000
    output_usd = output_tokens * price.output_usd_per_million / 1_000_000
    return LabelingEstimate(
        model=model_id,
        mode=mode,
        leaf_count=leaf_count,
        internal_node_count=internal_node_count,
        node_count=leaf_count + internal_node_count,
        bottom_up_calls=bottom_up_calls,
        contrast_calls=contrast_calls,
        call_count=call_count,
        input_characters=input_characters,
        input_tokens=input_tokens,
        output_tokens_per_card=OUTPUT_TOKENS_PER_CARD,
        output_tokens=output_tokens,
        input_usd=input_usd,
        output_usd=output_usd,
        usd=input_usd + output_usd,
        price_as_of=price.as_of,
        price_is_estimate=price.is_estimate,
        price_label=price.label,
    )


def _price_for(
    model: str,
    price_table: Mapping[str, ModelPrice] | None,
) -> ModelPrice:
    table = DEFAULT_PRICE_TABLE if price_table is None else price_table
    try:
        price = table[model]
    except (KeyError, TypeError) as exc:
        raise LabelingError(
            f"no price for model {model!r}; pass a price_table entry for that id"
        ) from exc
    if not isinstance(price, ModelPrice):
        raise LabelingError(f"price table entry for {model!r} must be a ModelPrice")
    if price.input_usd_per_million < 0 or price.output_usd_per_million < 0:
        raise LabelingError(f"price table entry for {model!r} has a negative rate")
    return price


def _validate_cards(
    tree: Tree,
    cards: Mapping[str, NodeCard] | None,
) -> Mapping[str, NodeCard] | None:
    if cards is None:
        return None
    if isinstance(cards, (str, bytes)) or not isinstance(cards, Mapping):
        raise LabelingError("cards must map node ids to NodeCard values")
    for node_id, card in cards.items():
        if node_id not in tree.nodes:
            raise LabelingError(f"cards names unknown node {node_id!r}")
        if not isinstance(card, NodeCard):
            raise LabelingError(f"cards[{node_id!r}] must be a NodeCard")
    return cards


def _ordered_nodes(tree: Tree) -> list[dict[str, Any]]:
    by_item: dict[str, dict[str, Any]] = {}
    internals: list[dict[str, Any]] = []
    for node in tree.nodes.values():
        kind = node["kind"]
        if kind == "leaf":
            by_item[node["item_id"]] = node
        elif kind == "internal":
            internals.append(node)
        else:
            raise LabelingError(f"unknown node kind {kind!r}")
    internals.sort(key=lambda node: int(node["linkage_row"]))
    return [by_item[item_id] for item_id in tree.item_ids] + internals


def _card_for(cards: Mapping[str, NodeCard] | None, node_id: str) -> NodeCard:
    if cards is not None and node_id in cards:
        return cards[node_id]
    return SIZING_STAND_IN_CARD


def _document_block(tree: Tree, item_id: str) -> str:
    document = tree.documents_by_id.get(item_id)
    if document is None:
        return format_document_for_prompt(text=item_id, item_id=item_id)
    return format_document_for_prompt(
        text=document.text,
        title=document.title,
        item_id=document.item_id,
    )


def _bottom_up_prompts(
    tree: Tree,
    cards: Mapping[str, NodeCard] | None,
) -> list[LabelingPrompt]:
    rendered: list[LabelingPrompt] = []
    for node in _ordered_nodes(tree):
        if node["kind"] == "leaf":
            rendered.append(render_leaf_prompt(_document_block(tree, node["item_id"])))
            continue
        children = node["children"]
        if len(children) != 2:
            raise LabelingError(
                f"parent {node['node_id']} must have two child cards"
            )
        left_id, right_id = children
        rendered.append(
            render_parent_prompt(_card_for(cards, left_id), _card_for(cards, right_id))
        )
    return rendered


def _validate_neighbor_counts(
    tree: Tree,
    neighbor_counts: Mapping[str, int],
) -> dict[str, int]:
    if isinstance(neighbor_counts, (str, bytes)) or not isinstance(
        neighbor_counts, Mapping
    ):
        raise LabelingError("neighbor_counts must map node ids to integer counts")
    cleaned: dict[str, int] = {}
    for node_id, count in neighbor_counts.items():
        if node_id not in tree.nodes:
            raise LabelingError(f"neighbor_counts names unknown node {node_id!r}")
        if type(count) is not int:
            raise LabelingError(
                f"neighbor count for {node_id!r} must be an integer"
            )
        if count < 0:
            raise LabelingError(f"neighbor count for {node_id!r} must be >= 0")
        cleaned[str(node_id)] = count
    return cleaned


def _contrast_prompts(
    tree: Tree,
    cards: Mapping[str, NodeCard] | None,
    neighbor_counts: Mapping[str, int],
) -> list[LabelingPrompt]:
    counts = _validate_neighbor_counts(tree, neighbor_counts)
    rendered: list[LabelingPrompt] = []
    for node in _ordered_nodes(tree):
        count = counts.get(node["node_id"], 0)
        if count < 1:
            continue
        neighbors = (SIZING_STAND_IN_CARD,) * count
        rendered.append(
            render_contrast_prompt(_card_for(cards, node["node_id"]), neighbors)
        )
    return rendered
