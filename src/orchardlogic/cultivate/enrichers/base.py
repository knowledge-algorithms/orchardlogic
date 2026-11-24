"""Enricher contract (spec section 3): protocols, TreeContext, records, registry."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from orchardlogic.cultivate.document import Document
from orchardlogic.cultivate.exceptions import OrchardError
from orchardlogic.cultivate.perspectives import sha256_json

ENRICHER_SCHEMA_VERSION = "orchard_enricher_v1"


class UnknownEnricherError(OrchardError, KeyError):
    """Raised when a handle does not name a registered enricher."""


@dataclass(frozen=True)
class TreeContext:
    """Everything a tree enricher may read (spec section 3).

    ``cut`` is the full ``orchard_cultivar_cut_v1`` document (``cut["nodes"]`` holds the
    rows; ``display`` is not filled yet). ``tree`` is an optional, additive field: the
    ``orchard.tree.Tree`` the cut came from, needed only by LLM enrichers.
    """

    tree_id: str
    cut: dict
    docs: Mapping[str, Document]
    item_outputs: Mapping[str, Mapping[str, dict]]
    node_outputs: Mapping[str, Mapping[str, dict]]
    vectors: Mapping[str, np.ndarray] | None
    only: frozenset[str] | None = None
    tree: Any = None

    @property
    def nodes(self) -> list[dict]:
        cut = self.cut
        return list(cut["nodes"]) if isinstance(cut, Mapping) else list(cut)


@runtime_checkable
class ItemEnricher(Protocol):
    record: dict

    def enrich(self, docs: Sequence[Document]) -> dict[str, dict]: ...


@runtime_checkable
class TreeEnricher(Protocol):
    record: dict

    def enrich(self, ctx: TreeContext) -> dict[str, dict]: ...


def handle_of(record: Mapping[str, Any]) -> str:
    return f"{record['name']}@{record['version']}"


def handle_filename(handle: str) -> str:
    """``tree.x@1`` -> ``tree.x__1`` (used in cultivar file names)."""
    return handle.replace("@", "__")


def enricher_id(record: Mapping[str, Any]) -> str:
    return sha256_json(record)


def module_sha256(module_file: str | Path) -> str:
    """sha256 of the module source file bytes, with CRLF normalised to LF (checkout-stable)."""
    data = Path(module_file).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def make_record(
    *,
    name: str,
    version: str,
    scope: str,
    method: str,
    params: Mapping[str, Any],
    title: str,
    description: str,
    module: str,
    module_file: str | Path,
    link: str | None = None,
) -> dict[str, Any]:
    assert scope in ("item", "tree") and method in ("extractive", "statistical", "llm")
    return {
        "schema_version": ENRICHER_SCHEMA_VERSION,
        "name": name,
        "version": version,
        "scope": scope,
        "method": method,
        "params": dict(params),
        "code": {
            "title": title,
            "description": description,
            "module": module,
            "sha256": module_sha256(module_file),
            "link": link,
        },
    }


_FACTORIES: dict[str, Callable[[], ItemEnricher | TreeEnricher]] = {}
_INSTANCES: dict[str, ItemEnricher | TreeEnricher] = {}


def register_enricher(handle: str, factory: Callable[[], ItemEnricher | TreeEnricher]) -> None:
    _FACTORIES[handle] = factory
    _INSTANCES.pop(handle, None)


def available_enrichers() -> list[str]:
    return sorted(_FACTORIES)


def get_enricher(handle: str) -> ItemEnricher | TreeEnricher:
    try:
        factory = _FACTORIES[handle]
    except KeyError as exc:
        raise UnknownEnricherError(
            f"unknown enricher {handle!r}; available: {', '.join(sorted(_FACTORIES))}"
        ) from exc
    if handle not in _INSTANCES:
        _INSTANCES[handle] = factory()
    return _INSTANCES[handle]
