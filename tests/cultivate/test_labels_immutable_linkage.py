"""Labels and contrast cards never mutate linkage or an existing card set."""

from __future__ import annotations

import json

import orchardlogic.cultivate as orchard
from orchardlogic.cultivate import import_labels, label_bottom_up, label_contrast, label_intrinsic
from orchardlogic.cultivate.labeling.cards import CARD_SCHEMA_VERSION
from orchardlogic.cultivate.labeling.client import LabelingPrompt


def test_intrinsic_and_imported_labels_do_not_mutate_linkage(sample_tree) -> None:
    before = sample_tree.linkage_fingerprint()
    intrinsic = label_intrinsic(sample_tree, name="intrinsic_v1")
    assert sample_tree.linkage_fingerprint() == before
    assert sample_tree.root_node_id in intrinsic
    assert sample_tree.active_label_set == "intrinsic_v1"

    import_labels(
        sample_tree,
        {sample_tree.root_node_id: "User root"},
        name="imported",
    )
    assert sample_tree.linkage_fingerprint() == before
    assert sample_tree.labels["imported"][sample_tree.root_node_id] == "User root"

    sample_tree.use_labels("intrinsic_v1")
    assert sample_tree.linkage_fingerprint() == before
    sample_tree.use_labels("imported")
    assert sample_tree.linkage_fingerprint() == before


class _FakeLabelClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, model: str, prompt: LabelingPrompt) -> str:
        self.calls += 1
        return json.dumps(
            {
                "schema_version": CARD_SCHEMA_VERSION,
                "title": f"Card {self.calls}",
                "summary": f"Summary {self.calls}.",
                "scope": ["scope"],
                "capabilities": ["act"],
                "concepts": ["idea"],
                "uncertainty": ["none"],
            }
        )


def test_contrast_does_not_mutate_linkage_or_the_source_card_set(sample_tree) -> None:
    assert "contrastive" not in orchard.__all__
    assert not hasattr(orchard, "label_contrastive")
    assert not hasattr(orchard, "ContrastiveLabeler")
    assert hasattr(orchard, "label_contrast")

    before = sample_tree.linkage_fingerprint()
    canonical = sample_tree.to_canonical_dict()
    label_bottom_up(sample_tree, client=_FakeLabelClient())
    source_cards = {
        node_id: card.to_dict()
        for node_id, card in sample_tree.cards["llm_bottom_up"].items()
    }
    source_labels = dict(sample_tree.labels["llm_bottom_up"])
    source_map = sample_tree.cards["llm_bottom_up"]

    cards = label_contrast(sample_tree, client=_FakeLabelClient())

    assert sample_tree.linkage_fingerprint() == before
    assert sample_tree.to_canonical_dict() == canonical
    assert sample_tree.cards["llm_bottom_up"] is source_map
    assert {
        node_id: card.to_dict()
        for node_id, card in sample_tree.cards["llm_bottom_up"].items()
    } == source_cards
    assert sample_tree.labels["llm_bottom_up"] == source_labels
    assert sample_tree.active_label_set == "llm_contrast"
    assert set(cards) == set(sample_tree.nodes)
    assert "llm_contrast" in sample_tree.labels
