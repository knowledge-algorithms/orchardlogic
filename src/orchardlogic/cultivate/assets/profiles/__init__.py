"""Packaged similarity-profile weight dicts."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

PACKAGED_MINILM_LAYER = "description_minilm_centered_cosine"
def load_profile_payload(name: str) -> dict[str, Any]:
    """Load a packaged profile JSON payload by tree id."""
    resource = resources.files("orchardlogic.cultivate.assets.profiles").joinpath(f"{name}.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_semantic_profile() -> dict[str, Any]:
    """Load the packaged no-app semantic profile (0.66 MiniLM / 0.34 TF-IDF)."""
    return load_profile_payload("semantic")


def _rename_minilm_layer(weights: dict[str, float], minilm_layer: str) -> dict[str, float]:
    if minilm_layer != PACKAGED_MINILM_LAYER and PACKAGED_MINILM_LAYER in weights:
        weights[minilm_layer] = weights.pop(PACKAGED_MINILM_LAYER)
    return {name: float(value) for name, value in weights.items()}


def load_semantic_weights(*, minilm_layer: str) -> dict[str, float]:
    """Return semantic weights with the MiniLM layer name for the active transform."""
    payload = load_semantic_profile()
    return _rename_minilm_layer(dict(payload["weights"]), minilm_layer)


