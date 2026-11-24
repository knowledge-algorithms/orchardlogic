"""Opt-in labeling: cards, prompts, and a cost estimate.

The bottom-up runner is ``orchard.label_bottom_up``. Importing this package
does not open a network connection.
"""

from orchardlogic.cultivate.labeling.cards import (
    CARD_SCHEMA_VERSION,
    NodeCard,
    parse_card,
    serialize_card,
    style_warnings_for,
)
from orchardlogic.cultivate.labeling.client import LabelingPrompt
from orchardlogic.cultivate.labeling.cost import (
    DEFAULT_LABELING_MODEL,
    DEFAULT_PRICE_TABLE,
    LEGACY_LABELING_MODEL,
    OUTPUT_TOKENS_PER_CARD,
    SIZING_STAND_IN_CARD,
    LabelingEstimate,
    ModelPrice,
    estimate_labeling,
)
from orchardlogic.cultivate.labeling.prompts import (
    BOTTOM_UP_LEAF_PROMPT_VERSION,
    BOTTOM_UP_LEAF_SYSTEM,
    BOTTOM_UP_PARENT_PROMPT_VERSION,
    BOTTOM_UP_PARENT_SYSTEM,
    CONTRAST_PROMPT_VERSION,
    CONTRAST_SYSTEM,
    format_document_for_prompt,
    render_contrast_prompt,
    render_leaf_prompt,
    render_parent_prompt,
)

__all__ = [
    "BOTTOM_UP_LEAF_PROMPT_VERSION",
    "BOTTOM_UP_LEAF_SYSTEM",
    "BOTTOM_UP_PARENT_PROMPT_VERSION",
    "BOTTOM_UP_PARENT_SYSTEM",
    "CARD_SCHEMA_VERSION",
    "CONTRAST_PROMPT_VERSION",
    "CONTRAST_SYSTEM",
    "LabelingPrompt",
    "DEFAULT_LABELING_MODEL",
    "LEGACY_LABELING_MODEL",
    "DEFAULT_PRICE_TABLE",
    "OUTPUT_TOKENS_PER_CARD",
    "SIZING_STAND_IN_CARD",
    "LabelingEstimate",
    "ModelPrice",
    "NodeCard",
    "estimate_labeling",
    "format_document_for_prompt",
    "parse_card",
    "render_contrast_prompt",
    "render_leaf_prompt",
    "render_parent_prompt",
    "serialize_card",
    "style_warnings_for",
]
