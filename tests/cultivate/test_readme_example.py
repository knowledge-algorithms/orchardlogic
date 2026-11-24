"""Semantic-only build with an explicit TF-IDF backend."""

from __future__ import annotations

import pytest

from orchardlogic.cultivate import OrchardBuilder
from orchardlogic.cultivate.backends.tfidf import TfidfEmbeddingBackend


def test_readme_semantic_corner() -> None:
    documents = [
        "Schedule a calendar reminder for tomorrow morning.",
        "Send an email summary of the weekly status notes.",
        "Create a task to review the draft proposal.",
        "Find documents mentioning quarterly budget planning.",
    ]
    orchard = OrchardBuilder(
        taxonomies=[],
        embedding_backend=TfidfEmbeddingBackend(),
    ).build(documents)
    semantic = orchard.tree("semantic")
    assert semantic.tree_id == "semantic"
    assert semantic.leaf_count == 4
