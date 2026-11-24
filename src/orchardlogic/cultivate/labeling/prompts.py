"""Versioned labeling prompts.

The strings are the product of this phase. Filling them is local string
work. Nothing here opens a network connection or calls a model.
"""

from __future__ import annotations

from collections.abc import Sequence

from orchardlogic.cultivate.exceptions import LabelingError
from orchardlogic.cultivate.labeling.cards import NodeCard, serialize_card
from orchardlogic.cultivate.labeling.client import LabelingPrompt

BOTTOM_UP_LEAF_PROMPT_VERSION = "bottom_up_leaf_v2"
BOTTOM_UP_PARENT_PROMPT_VERSION = "bottom_up_parent_v2"
CONTRAST_PROMPT_VERSION = "contrast_v2"

_BANNED_LIST = (
    "unified, consolidated, combined, integrated, comprehensive, various"
)

_TITLE_RULES = f"""\
Title: a few words. No articles. A word stays only if removing it would \
change the meaning. Ban consult-speak that only names the grouping: \
{_BANNED_LIST}.
Summary: one or two plain sentences. A word stays only if removing it would \
change the meaning. Ban the same grouping words: {_BANNED_LIST}.
"""

_JSON_RULES = """\
Return one JSON object and no other text. Use exactly these keys: title and \
summary. Do not add a style-warning field and do not ask for a retry.
"""

BOTTOM_UP_LEAF_SYSTEM = f"""\
Labeling prompt {BOTTOM_UP_LEAF_PROMPT_VERSION}.

Write one card from the document in the user message. Use only that document.

{_TITLE_RULES}
{_JSON_RULES}
"""

BOTTOM_UP_PARENT_SYSTEM = f"""\
Labeling prompt {BOTTOM_UP_PARENT_PROMPT_VERSION}.

Write one card from the two child cards in the user message only, not a \
descendant list. Those two cards are the only evidence. Cover both children. \
The parent card is more abstract than the children. Do not reuse one child's \
title as the parent title. Do not drop either child.

{_TITLE_RULES}
{_JSON_RULES}
"""

CONTRAST_SYSTEM = f"""\
Labeling prompt {CONTRAST_PROMPT_VERSION}.

Rewrite the title and the summary so the card is less like the neighbor \
cards in the user message. A word stays when it increases that distance. \
Title and summary are what must change the comparison.

{_TITLE_RULES}
{_JSON_RULES}
"""


def format_document_for_prompt(
    *,
    text: str,
    title: str = "",
    item_id: str = "",
) -> str:
    """Leaf text block inserted into the bottom-up leaf prompt."""
    body = text.strip() if isinstance(text, str) else str(text).strip()
    heading = title.strip() if isinstance(title, str) else ""
    if heading and body:
        return f"title: {heading}\ntext: {body}"
    if body:
        return body
    if heading:
        return f"title: {heading}"
    return item_id.strip()


def _require_card(card: NodeCard, role: str) -> NodeCard:
    if not isinstance(card, NodeCard):
        raise LabelingError(f"{role} must be a NodeCard")
    return card


def render_leaf_prompt(document_text: str) -> LabelingPrompt:
    """Bottom-up leaf prompt: one card from that document."""
    if not isinstance(document_text, str):
        raise LabelingError("leaf document text must be a string")
    return LabelingPrompt(
        system=BOTTOM_UP_LEAF_SYSTEM,
        user=f"Document:\n{document_text}",
    )


def render_parent_prompt(child_a: NodeCard, child_b: NodeCard) -> LabelingPrompt:
    """Bottom-up parent prompt: one card from the two child cards only."""
    _require_card(child_a, "child card A")
    _require_card(child_b, "child card B")
    user = (
        "Child card A:\n"
        f"{serialize_card(child_a)}\n\n"
        "Child card B:\n"
        f"{serialize_card(child_b)}"
    )
    return LabelingPrompt(system=BOTTOM_UP_PARENT_SYSTEM, user=user)


def render_contrast_prompt(card: NodeCard, neighbors: Sequence[NodeCard]) -> LabelingPrompt:
    """Contrast prompt text. Does not choose neighbors or call a model."""
    _require_card(card, "card")
    if isinstance(neighbors, (str, bytes)) or not isinstance(neighbors, Sequence):
        raise LabelingError("contrast neighbors must be a sequence of cards")
    blocks = [serialize_card(_require_card(item, "neighbor card")) for item in neighbors]
    user = (
        "Card:\n"
        f"{serialize_card(card)}\n\n"
        "Neighbor cards:\n"
        f"{chr(10).join(blocks)}"
    )
    return LabelingPrompt(system=CONTRAST_SYSTEM, user=user)
