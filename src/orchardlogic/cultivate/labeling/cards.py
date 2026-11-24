"""Label cards for one Orchard node.

A card is the contract a later labeling pass will store. This module checks
the schema and records style warnings. It does not call a model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from orchardlogic.cultivate.exceptions import LabelingError
from orchardlogic.cultivate.identity import canonical_json

CARD_SCHEMA_VERSION = "orchard_node_card_v1"

_REQUIRED_FIELDS = (
    "schema_version",
    "title",
    "summary",
    "scope",
    "capabilities",
    "concepts",
    "uncertainty",
)
_DESCRIPTOR_FIELDS = ("scope", "capabilities", "concepts", "uncertainty")

# Consult-speak that only names the grouping. Recorded, not rejected.
_BANNED_WORDS = (
    "unified",
    "consolidated",
    "combined",
    "integrated",
    "comprehensive",
    "various",
)
_ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b", re.IGNORECASE)
_BANNED_RE = re.compile(
    r"\b(?:" + "|".join(_BANNED_WORDS) + r")\b",
    re.IGNORECASE,
)


def style_warnings_for(title: str, summary: str) -> tuple[str, ...]:
    """Record banned words and a title article. These do not fail the card."""
    warnings: list[str] = []
    for match in _ARTICLE_RE.finditer(title):
        warnings.append(f"title contains article {match.group(0).lower()!r}")
    for field_name, text in (("title", title), ("summary", summary)):
        for match in _BANNED_RE.finditer(text):
            warnings.append(
                f"{field_name} contains banned word {match.group(0).lower()!r}"
            )
    return tuple(warnings)


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise LabelingError(
            f"label card schema mismatch: {field_name} must be a string"
        )
    text = value.strip()
    if not text:
        raise LabelingError(f"label card {field_name} is empty")
    return text


def _require_phrases(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise LabelingError(
            f"label card schema mismatch: {field_name} must be a list of strings"
        )
    phrases: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise LabelingError(
                f"label card schema mismatch: {field_name} must be a list of strings"
            )
        phrase = item.strip()
        if not phrase:
            raise LabelingError(
                f"label card schema mismatch: {field_name} contains an empty phrase"
            )
        phrases.append(phrase)
    return tuple(phrases)


@dataclass(frozen=True, slots=True)
class NodeCard:
    """One node's title, summary, and descriptor fields.

    ``style_warnings`` is filled by the constructor from the title and
    summary. Passing warnings in, or asking for a retry, is not part of
    the card. Empty title, empty summary, and a schema mismatch raise
    ``LabelingError``.
    """

    title: str
    summary: str
    scope: tuple[str, ...]
    capabilities: tuple[str, ...]
    concepts: tuple[str, ...]
    uncertainty: tuple[str, ...]
    schema_version: str = CARD_SCHEMA_VERSION
    style_warnings: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CARD_SCHEMA_VERSION:
            raise LabelingError(
                "label card schema mismatch: "
                f"schema_version {self.schema_version!r}"
            )
        title = _require_text(self.title, "title")
        summary = _require_text(self.summary, "summary")
        scope = _require_phrases(self.scope, "scope")
        capabilities = _require_phrases(self.capabilities, "capabilities")
        concepts = _require_phrases(self.concepts, "concepts")
        uncertainty = _require_phrases(self.uncertainty, "uncertainty")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "concepts", concepts)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "schema_version", CARD_SCHEMA_VERSION)
        object.__setattr__(
            self,
            "style_warnings",
            style_warnings_for(title, summary),
        )

    def model_payload(self) -> dict[str, Any]:
        """Fields a prompt sends. Style warnings stay on the card."""
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "summary": self.summary,
            "scope": list(self.scope),
            "capabilities": list(self.capabilities),
            "concepts": list(self.concepts),
            "uncertainty": list(self.uncertainty),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_payload()
        payload["style_warnings"] = list(self.style_warnings)
        return payload


def _optional_phrases(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return _require_phrases(value, field_name)


def parse_model_card(payload: Mapping[str, Any]) -> NodeCard:
    """Parse a live model response. Only title and summary are required."""
    if not isinstance(payload, Mapping):
        raise LabelingError("label card schema mismatch: expected an object")
    if "title" not in payload or "summary" not in payload:
        raise LabelingError(
            "label card schema mismatch: missing title or summary"
        )
    return NodeCard(
        title=payload["title"],
        summary=payload["summary"],
        scope=_optional_phrases(payload.get("scope"), "scope"),
        capabilities=_optional_phrases(payload.get("capabilities"), "capabilities"),
        concepts=_optional_phrases(payload.get("concepts"), "concepts"),
        uncertainty=_optional_phrases(payload.get("uncertainty"), "uncertainty"),
        schema_version=CARD_SCHEMA_VERSION,
    )


def parse_card(payload: Mapping[str, Any]) -> NodeCard:
    """Validate a card mapping and record style warnings.

    A banned word or an article in the title is stored on
    ``style_warnings`` and still returns a card. Empty title, empty
    summary, and a schema mismatch raise ``LabelingError``.
    """
    if not isinstance(payload, Mapping):
        raise LabelingError("label card schema mismatch: expected an object")
    missing = [key for key in _REQUIRED_FIELDS if key not in payload]
    if missing:
        raise LabelingError(
            "label card schema mismatch: missing " + ", ".join(missing)
        )
    return NodeCard(
        title=payload["title"],
        summary=payload["summary"],
        scope=payload["scope"],
        capabilities=payload["capabilities"],
        concepts=payload["concepts"],
        uncertainty=payload["uncertainty"],
        schema_version=payload["schema_version"],
    )


def serialize_card(card: NodeCard) -> str:
    """Stable JSON of the model-facing card fields."""
    if not isinstance(card, NodeCard):
        raise LabelingError("expected a NodeCard")
    return canonical_json(card.model_payload())
