"""Perspective records (``orchard_perspective_v1``): how items become a tree.

Spec: DATA_ORCHARD_SPEC.md section 2 (+ A10 for the vector space name).
A perspective is plain JSON; its id is the sha256 of its canonical JSON.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from typing import Any

from orchardlogic.cultivate.backends.minilm import MINILM_MODEL_ID, MINILM_REVISION
from orchardlogic.cultivate.exceptions import OrchardError
from orchardlogic.cultivate.identity import canonical_json

PERSPECTIVE_SCHEMA_VERSION = "orchard_perspective_v1"
HANDLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]*@[A-Za-z0-9_.\-]+$")
LINKAGE_METHODS = ("ward", "average", "complete", "single")


class InvalidPerspectiveError(OrchardError, ValueError):
    """Raised when a perspective record does not satisfy orchard_perspective_v1."""


def minilm_space_name(revision: str = MINILM_REVISION, template_version: str = "tt1") -> str:
    """Vector space name: model, revision, pooling and text config (A10)."""
    return f"minilm-l6-v2@{revision[:8]}/mean/{template_version}"


def sha256_json(value: Any) -> str:
    """sha256 of canonical JSON (spec section 1)."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def perspective_id(record: Mapping[str, Any]) -> str:
    """64-hex id of a perspective record."""
    return sha256_json(record)


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def topic_perspective(**overrides: Any) -> dict[str, Any]:
    """The default topic perspective; ``overrides`` deep-merge dict values.

    Lists (``cuts``, enricher lists, ``features``) are replaced wholesale.
    """
    record: dict[str, Any] = {
        "schema_version": PERSPECTIVE_SCHEMA_VERSION,
        "name": "topic",
        "version": "2",
        "description": (
            "Topic tree over captured pages: MiniLM embeddings, Ward linkage, "
            "3-level packed cut."
        ),
        "embedding": {
            "space": minilm_space_name(),
            "model_id": MINILM_MODEL_ID,
            "text_template": "{title}\n\n{text}",
            "max_chars": 2000,
            "normalize": True,
        },
        "features": [{"name": "embedding", "weight": 1.0}],
        "taxonomies": [],
        "linkage": {"method": "ward", "metric": "euclidean", "on": "normalized_embedding"},
        "cuts": [
            {
                "name": "topics",
                "top_criterion": "cluster_count",
                "cluster_count": 8,
                "min_width": 3,
                "max_width": 8,
                "target_width": 5,
                "max_depth": 3,
            }
        ],
        "item_enrichers": ["item.extract@1"],
        "tree_enrichers": [
            "tree.bottom_up_extractive@2",
            "tree.top_down_tags@2",
            "tree.side_contrast@2",
        ],
        "display": {
            "title": ["tree.side_contrast@2", "tree.bottom_up_extractive@2"],
            "summary": ["tree.bottom_up_extractive@2"],
            "tags": ["tree.top_down_tags@2"],
        },
    }
    _deep_merge(record, overrides)
    validate_perspective(record)
    return record


def validate_perspective(record: Mapping[str, Any]) -> None:
    """Raise InvalidPerspectiveError unless ``record`` is a valid orchard_perspective_v1."""

    def need(cond: bool, message: str) -> None:
        if not cond:
            raise InvalidPerspectiveError(message)

    need(isinstance(record, Mapping), "perspective must be a mapping")
    need(
        record.get("schema_version") == PERSPECTIVE_SCHEMA_VERSION,
        f"schema_version must be {PERSPECTIVE_SCHEMA_VERSION!r}",
    )
    for key in ("name", "version"):
        need(
            isinstance(record.get(key), str) and bool(record[key].strip()),
            f"{key} must be a non-empty string",
        )
    emb = record.get("embedding")
    need(isinstance(emb, Mapping), "embedding must be an object")
    for key in ("space", "model_id", "text_template"):
        need(isinstance(emb.get(key), str) and bool(emb[key]), f"embedding.{key} is required")
    need(
        isinstance(emb.get("max_chars"), int) and emb["max_chars"] > 0,
        "embedding.max_chars must be a positive integer",
    )
    need(isinstance(emb.get("normalize"), bool), "embedding.normalize must be a boolean")
    features = record.get("features")
    need(isinstance(features, list) and features, "features must be a non-empty list")
    for feature in features:
        need(
            isinstance(feature, Mapping)
            and isinstance(feature.get("name"), str)
            and isinstance(feature.get("weight"), (int, float))
            and feature["weight"] > 0,
            "each feature needs a name and a positive weight",
        )
    linkage = record.get("linkage")
    need(isinstance(linkage, Mapping), "linkage must be an object")
    method = linkage.get("method")
    need(method in LINKAGE_METHODS, f"linkage.method must be one of {LINKAGE_METHODS}")
    if method == "ward":
        need(len(features) == 1, "ward linkage supports a single feature only (A5)")
        need(linkage.get("metric", "euclidean") == "euclidean", "ward requires euclidean")
    cuts = record.get("cuts")
    need(isinstance(cuts, list) and cuts, "cuts must be a non-empty list")
    names = [cut.get("name") if isinstance(cut, Mapping) else None for cut in cuts]
    need(
        all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names),
        "cuts need unique non-empty names",
    )
    for cut in cuts:
        need("top_criterion" in cut, f"cut {cut['name']!r} needs top_criterion")
    for key in ("item_enrichers", "tree_enrichers"):
        handles = record.get(key)
        need(isinstance(handles, list), f"{key} must be a list")
        for handle in handles:
            need(
                isinstance(handle, str) and HANDLE_RE.match(handle) is not None,
                f"{key}: {handle!r} is not a name@version handle",
            )
    if "placement" in record:
        need(record["placement"] in ("knn", "centroid"), "placement must be knn or centroid")
    display = record.get("display")
    need(isinstance(display, Mapping), "display must be an object")
    allowed = set(record["tree_enrichers"])
    for field_name, order in display.items():
        need(field_name in ("title", "summary", "tags"), f"unknown display field {field_name!r}")
        need(isinstance(order, list), f"display.{field_name} must be a list")
        for handle in order:
            need(
                handle in allowed,
                f"display.{field_name} references {handle!r}, not in tree_enrichers",
            )
