"""Concurrent labeling waves with one retry, a 4 percent gate, and checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from orchardlogic.cultivate.exceptions import LabelingError, MissingOptionalDependencyError
from orchardlogic.cultivate.labeling.cards import NodeCard, parse_card
from orchardlogic.cultivate.labeling.client import (
    DEFAULT_MAX_WORKERS,
    LabelingClient,
    LabelingPrompt,
    complete_many,
)

CHECKPOINT_SCHEMA_VERSION = "orchard_labeling_checkpoint_v1"
# 4 percent. ``count * 25 >= denominator`` is the same cutoff without a float edge.
FAILURE_RATE_LIMIT = 0.04
_FAILURE_RATE_DENOMINATOR = 25


@dataclass(frozen=True, slots=True)
class WaveItem:
    """One node in a labeling wave."""

    request_index: int
    node_id: str
    prompt: LabelingPrompt


@dataclass(frozen=True, slots=True)
class LabelingErrorRecord:
    node_id: str
    request_index: int
    message: str
    fallback: bool


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Cards, wave indexes, and error records loaded from ``{name}.json``."""

    cards: dict[str, NodeCard]
    request_indexes: dict[str, int]
    errors: list[LabelingErrorRecord]


@dataclass(frozen=True, slots=True)
class WaveResult:
    """Successes from one wave, including fallbacks when the rate stays under 4%."""

    cards: dict[str, NodeCard]
    errors: list[LabelingErrorRecord]
    request_indexes: dict[str, int]


class WaveStopped(LabelingError):
    """The wave crossed the 4 percent limit. Successes are kept; no fallbacks."""

    def __init__(
        self,
        message: str,
        *,
        cards: dict[str, NodeCard],
        errors: list[LabelingErrorRecord],
        request_indexes: dict[str, int],
    ) -> None:
        super().__init__(message)
        self.cards = cards
        self.errors = errors
        self.request_indexes = request_indexes


def failure_rate_reached(failure_count: int, denominator: int) -> bool:
    """True when remaining failures are at least 4 percent of ``denominator``."""
    if denominator < 1:
        raise LabelingError("failure denominator must be at least 1")
    return failure_count * _FAILURE_RATE_DENOMINATOR >= denominator


def load_checkpoint(path: Path, *, set_name: str) -> CheckpointState:
    """Load cards, request indexes, and errors. A missing file is empty."""
    if not path.is_file():
        return CheckpointState(cards={}, request_indexes={}, errors=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LabelingError("labeling checkpoint must be a JSON object")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise LabelingError(
            "unsupported labeling checkpoint schema: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("name") != set_name:
        raise LabelingError(
            f"checkpoint name {payload.get('name')!r} does not match {set_name!r}"
        )
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, dict):
        raise LabelingError("labeling checkpoint is missing cards")
    raw_indexes = payload.get("request_indexes", {})
    if not isinstance(raw_indexes, dict):
        raise LabelingError("labeling checkpoint request_indexes must be an object")
    indexes: dict[str, int] = {}
    for node_id, request_index in raw_indexes.items():
        if type(request_index) is not int:
            raise LabelingError(
                f"checkpoint request index for {node_id!r} must be an integer"
            )
        indexes[str(node_id)] = request_index
    raw_errors = payload.get("errors", [])
    if not isinstance(raw_errors, list):
        raise LabelingError("labeling checkpoint errors must be a list")
    errors: list[LabelingErrorRecord] = []
    for record in raw_errors:
        if not isinstance(record, dict):
            raise LabelingError("labeling checkpoint error must be an object")
        node_id = record.get("node_id")
        request_index = record.get("request_index")
        message = record.get("message")
        fallback = record.get("fallback")
        if not isinstance(node_id, str) or not node_id:
            raise LabelingError("labeling checkpoint error is missing node_id")
        if type(request_index) is not int:
            raise LabelingError(
                f"checkpoint error request index for {node_id!r} must be an integer"
            )
        if not isinstance(message, str):
            raise LabelingError(f"checkpoint error message for {node_id!r} must be a string")
        if type(fallback) is not bool:
            raise LabelingError(f"checkpoint error fallback for {node_id!r} must be a bool")
        errors.append(
            LabelingErrorRecord(
                node_id=node_id,
                request_index=request_index,
                message=message,
                fallback=fallback,
            )
        )
    return CheckpointState(
        cards={
            str(node_id): parse_card(card_payload)
            for node_id, card_payload in raw_cards.items()
        },
        request_indexes=indexes,
        errors=errors,
    )


def write_checkpoint(
    path: Path,
    *,
    set_name: str,
    cards: Mapping[str, NodeCard],
    request_indexes: Mapping[str, int],
    errors: Sequence[LabelingErrorRecord],
) -> None:
    """Write ``{name}.json`` with cards so far, wave indexes, and error records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "name": set_name,
        "cards": {node_id: card.to_dict() for node_id, card in cards.items()},
        "request_indexes": {
            node_id: request_indexes[node_id]
            for node_id in request_indexes
        },
        "errors": [
            {
                "node_id": record.node_id,
                "request_index": record.request_index,
                "message": record.message,
                "fallback": record.fallback,
            }
            for record in errors
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_wave(
    cards: dict[str, NodeCard],
    errors: list[LabelingErrorRecord],
    request_indexes: dict[str, int],
    wave: WaveResult | WaveStopped,
) -> None:
    """Fold one wave into the run state. A new card drops that node's old error."""
    cards.update(wave.cards)
    request_indexes.update(wave.request_indexes)
    touched = set(wave.cards) | {record.node_id for record in wave.errors}
    errors[:] = [record for record in errors if record.node_id not in touched]
    errors.extend(wave.errors)


def run_wave(
    client: LabelingClient,
    *,
    model: str,
    items: Sequence[WaveItem],
    parse_response: Callable[[str], NodeCard],
    fallback_for: Callable[[WaveItem], NodeCard],
    failure_denominator: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> WaveResult:
    """Label one wave, retry each failed id once, then apply the 4 percent rule.

    Successes are keyed by node id and applied in request-index order. Under
    the limit, each remaining failure gets ``fallback_for``. At or above the
    limit, successes are returned on ``WaveStopped`` and no fallback is stored.
    """
    if failure_denominator < 1:
        raise LabelingError("failure denominator must be at least 1")
    request_indexes = {item.node_id: item.request_index for item in items}
    if not items:
        return WaveResult(cards={}, errors=[], request_indexes={})

    by_index = {item.request_index: item for item in items}
    try:
        responses = complete_many(
            client,
            model=model,
            prompts=[(item.request_index, item.prompt) for item in items],
            max_workers=max_workers,
        )
        batch_error: str | None = None
    except MissingOptionalDependencyError:
        raise
    except Exception as exc:
        responses = {}
        batch_error = str(exc)

    produced: dict[str, NodeCard] = {}
    failed: dict[int, str] = {}
    for index in sorted(by_index):
        item = by_index[index]
        if index not in responses:
            failed[index] = batch_error or "missing model response"
            continue
        try:
            produced[item.node_id] = parse_response(responses[index])
        except MissingOptionalDependencyError:
            raise
        except Exception as exc:
            failed[index] = str(exc)

    for index in sorted(failed):
        item = by_index[index]
        try:
            raw = client.complete(model=model, prompt=item.prompt)
            produced[item.node_id] = parse_response(raw)
        except MissingOptionalDependencyError:
            raise
        except Exception as exc:
            failed[index] = str(exc)
            continue
        del failed[index]

    if not failed:
        return WaveResult(
            cards=produced,
            errors=[],
            request_indexes=request_indexes,
        )

    errors = [
        LabelingErrorRecord(
            node_id=by_index[index].node_id,
            request_index=index,
            message=failed[index],
            fallback=False,
        )
        for index in sorted(failed)
    ]
    if failure_rate_reached(len(failed), failure_denominator):
        percent = FAILURE_RATE_LIMIT * 100
        raise WaveStopped(
            f"labeling failure rate {len(failed)}/{failure_denominator} "
            f"reached the {percent:.0f}% limit",
            cards=produced,
            errors=errors,
            request_indexes=request_indexes,
        )

    fallback_errors: list[LabelingErrorRecord] = []
    for index in sorted(failed):
        item = by_index[index]
        produced[item.node_id] = fallback_for(item)
        fallback_errors.append(
            LabelingErrorRecord(
                node_id=item.node_id,
                request_index=index,
                message=failed[index],
                fallback=True,
            )
        )
    return WaveResult(
        cards=produced,
        errors=fallback_errors,
        request_indexes=request_indexes,
    )
