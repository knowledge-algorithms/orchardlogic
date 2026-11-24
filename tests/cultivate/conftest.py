"""Shared fixtures for Orchard tests."""

from __future__ import annotations

import os
import sys

import pytest

_TREEDB_SRC = os.environ.get("TREEDB_SRC")
if _TREEDB_SRC and _TREEDB_SRC not in sys.path:
    sys.path.insert(0, _TREEDB_SRC)

from orchardlogic.cultivate import Document, Orchard, Tree
from orchardlogic.cultivate.fixtures import load_documents, load_sample_linkage


@pytest.fixture
def documents() -> list[Document]:
    return load_documents()


@pytest.fixture
def sample_tree(documents: list[Document]) -> Tree:
    payload = load_sample_linkage()
    return Tree.from_linkage(
        payload["linkage"],
        item_ids=payload["leaf_ids"],
        tree_id="semantic",
        documents=documents,
    )


@pytest.fixture
def alt_tree(documents: list[Document]) -> Tree:
    """Second linkage on the same corpus (different merge order/distances)."""
    linkage = [
        [1.0, 2.0, 0.15, 2.0],
        [0.0, 3.0, 0.25, 2.0],
        [4.0, 5.0, 0.60, 4.0],
    ]
    item_ids = [doc.item_id for doc in documents]
    return Tree.from_linkage(
        linkage,
        item_ids=item_ids,
        tree_id="custom",
        documents=documents,
    )


@pytest.fixture
def multi_orchard(
    documents: list[Document],
    sample_tree: Tree,
    alt_tree: Tree,
) -> Orchard:
    return Orchard.from_trees(
        documents=documents,
        trees={"semantic": sample_tree, "custom": alt_tree},
    )


# ---- synthetic topic corpus + deterministic fake encoder (cultivate tests) ----

import hashlib as _hashlib  # noqa: E402
import random as _random  # noqa: E402
import re as _re  # noqa: E402

import numpy as np  # noqa: E402

TOPIC_VOCAB = {
    "cooking": (
        "recipe simmer sauce garlic onion butter oven roast flour dough kitchen skillet "
        "seasoning broth pasta whisk marinade"
    ).split(),
    "astronomy": (
        "telescope galaxy nebula orbit planet comet asteroid supernova constellation eclipse "
        "stellar cosmic lunar solar meteor"
    ).split(),
    "gardening": (
        "compost seedling tomato soil pruning mulch greenhouse perennial trellis fertilizer "
        "watering harvest raised-bed herbs weeding"
    ).split(),
    "football": (
        "striker midfield goalkeeper penalty league offside tackle dribble referee stadium "
        "transfer coach defender corner fixture"
    ).split(),
}
FILLER = "people often consider careful steady practice because results improve over several seasons".split()


def make_topic_docs(per_topic: int = 10, seed: int = 7) -> list[Document]:
    rng = _random.Random(seed)
    docs: list[Document] = []
    for topic, vocab in TOPIC_VOCAB.items():
        for n in range(per_topic):
            sentences = []
            for _ in range(4):
                words = rng.sample(vocab, 6) + rng.sample(FILLER, 4)
                rng.shuffle(words)
                sentences.append(" ".join(words).capitalize() + " matters for " + topic + " fans.")
            title = f"{rng.choice(vocab).title()} and {rng.choice(vocab)} notes {n}"
            docs.append(
                Document(
                    text=" ".join(sentences),
                    item_id=f"t:{topic}-{n:02d}",
                    title=title,
                    metadata={"captured_at": f"2026-0{1 + n % 9}-1{n % 9}T10:00:00Z", "topic": topic},
                )
            )
    return docs


def fake_encode(texts, dim: int = 96) -> np.ndarray:
    """Deterministic hashed bag-of-words encoder (no model)."""
    out = np.zeros((len(texts), dim))
    for row, text in enumerate(texts):
        for token in _re.findall(r"[a-z][a-z\-]+", text.lower()):
            digest = _hashlib.sha256(token.encode()).digest()
            out[row, int.from_bytes(digest[:4], "little") % dim] += 1.0
    return out


@pytest.fixture
def topic_docs() -> list[Document]:
    return make_topic_docs()


@pytest.fixture
def encoder():
    from orchardlogic.cultivate.backends.minilm import MiniLMEmbeddingBackend

    return MiniLMEmbeddingBackend(encode_fn=fake_encode)


V1_PERSPECTIVE = {
    "version": "1",
    "tree_enrichers": ["tree.bottom_up_extractive@1", "tree.top_down_tags@1", "tree.side_contrast@1"],
    "display": {
        "title": ["tree.side_contrast@1", "tree.bottom_up_extractive@1"],
        "summary": ["tree.bottom_up_extractive@1"],
        "tags": ["tree.top_down_tags@1"],
    },
}


def topic_perspective_v1(**overrides):
    """topic_perspective with the @1 enrichers (the tests below pin their behaviour)."""
    from orchardlogic.cultivate.perspectives import topic_perspective

    return topic_perspective(**{**V1_PERSPECTIVE, **overrides})
