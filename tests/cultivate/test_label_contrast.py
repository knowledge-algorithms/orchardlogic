"""Contrast labeling with a fake client. No AppWorld corpus and no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from orchardlogic.cultivate import Document, Orchard, Tree, label_bottom_up, label_contrast
from orchardlogic.cultivate.exceptions import InvalidIdentityError, LabelingError, MissingOptionalDependencyError
from orchardlogic.cultivate.labeling.cards import CARD_SCHEMA_VERSION, NodeCard
from orchardlogic.cultivate.labeling.client import LabelingPrompt
from orchardlogic.cultivate.labeling.contrast import select_neighbors
from orchardlogic.cultivate.labeling.cost import DEFAULT_LABELING_MODEL
from orchardlogic.cultivate.labeling.prompts import render_contrast_prompt

FIXTURE = Path(__file__).parent / "fixtures" / "labeling_16.json"


class RecordingClient:
    """Returns one valid card per call and records the prompt."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        title_prefix: str = "Card",
    ) -> None:
        self._responses = list(responses) if responses is not None else None
        self._title_prefix = title_prefix
        self._bad_prompts: set[tuple[str, str]] = set()
        self.prompts: list[LabelingPrompt] = []
        self.models: list[str] = []

    def complete(self, *, model: str, prompt: LabelingPrompt) -> str:
        self.models.append(model)
        self.prompts.append(prompt)
        key = (prompt.system, prompt.user)
        if key in self._bad_prompts:
            return "not-json"
        if self._responses is not None:
            if self._responses:
                item = self._responses.pop(0)
                if not item.strip():
                    self._bad_prompts.add(key)
                return item
            return "not-json"
        index = len(self.prompts)
        prefix = self._title_prefix
        return _card_json(
            title=f"{prefix} {index:02d}",
            summary=f"{prefix} summary {index:02d}.",
            scope=[f"{prefix} scope {index}"],
        )

    def complete_many(
        self,
        *,
        model: str,
        prompts: list[tuple[int, LabelingPrompt]],
        max_workers: int,
    ) -> dict[int, str]:
        return {
            index: self.complete(model=model, prompt=prompt)
            for index, prompt in sorted(prompts, key=lambda item: item[0])
        }


def _card_json(**overrides: object) -> str:
    payload: dict[str, object] = {
        "schema_version": CARD_SCHEMA_VERSION,
        "title": "Mail routing",
        "summary": "Routes letters.",
        "scope": ["letters"],
        "capabilities": ["route"],
        "concepts": ["mail"],
        "uncertainty": ["none"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _card(**overrides: object) -> NodeCard:
    payload = json.loads(_card_json(**overrides))
    return NodeCard(
        title=payload["title"],
        summary=payload["summary"],
        scope=payload["scope"],
        capabilities=payload["capabilities"],
        concepts=payload["concepts"],
        uncertainty=payload["uncertainty"],
        schema_version=payload["schema_version"],
    )


def _documents_from_fixture() -> list[Document]:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [
        Document(
            text=row["text"],
            item_id=row["item_id"],
            title=row["title"],
            source=row["source"],
        )
        for row in rows
    ]


def _balanced_linkage(n: int) -> list[list[float]]:
    clusters = [(index, 1) for index in range(n)]
    rows: list[list[float]] = []
    next_id = n
    level = 1
    while len(clusters) > 1:
        nxt: list[tuple[int, int]] = []
        for index in range(0, len(clusters), 2):
            left_id, left_size = clusters[index]
            right_id, right_size = clusters[index + 1]
            rows.append(
                [
                    float(left_id),
                    float(right_id),
                    round(0.1 * level, 10),
                    float(left_size + right_size),
                ]
            )
            nxt.append((next_id, left_size + right_size))
            next_id += 1
        clusters = nxt
        level += 1
    return rows


def _tree_from_documents(
    documents: list[Document],
    linkage: list[list[float]],
    *,
    tree_id: str = "labeling",
) -> Tree:
    return Tree.from_linkage(
        linkage,
        item_ids=[document.item_id for document in documents],
        tree_id=tree_id,
        documents=documents,
    )


def _fixture_tree() -> tuple[Tree, list[Document]]:
    documents = _documents_from_fixture()
    return _tree_from_documents(documents, _balanced_linkage(len(documents))), documents


def _two_leaf_tree() -> Tree:
    documents = [
        Document(text="Knead dough until smooth.", item_id="bread", title="Dough"),
        Document(text="Patch the inner tube.", item_id="bike", title="Tube"),
    ]
    return _tree_from_documents(documents, [[0, 1, 0.2, 2]])


def _parent_of(tree: Tree, node_id: str) -> dict:
    return next(
        node for node in tree.nodes.values() if node_id in node["children"]
    )


def _sibling_id(tree: Tree, node_id: str) -> str:
    children = _parent_of(tree, node_id)["children"]
    return next(child for child in children if child != node_id)


def _snapshot_cards(tree: Tree, name: str) -> dict[str, dict]:
    return {
        node_id: card.to_dict()
        for node_id, card in tree.cards[name].items()
    }


def _prepare_bottom_up(tree: Tree) -> None:
    label_bottom_up(tree, client=RecordingClient(title_prefix="Card"))


def test_root_keeps_both_children_and_leaves_skip_ancestors(sample_tree: Tree) -> None:
    neighbors = select_neighbors(sample_tree)
    root_id = sample_tree.root_node_id
    children = sample_tree.nodes[root_id]["children"]
    assert set(neighbors[root_id]) == set(children)
    assert len(neighbors[root_id]) == 2

    for node in sample_tree.nodes.values():
        if node["kind"] != "leaf":
            continue
        node_id = node["node_id"]
        parent = _parent_of(sample_tree, node_id)
        assert parent["node_id"] not in neighbors[node_id]
        assert root_id not in neighbors[node_id]
        assert neighbors[node_id][0] == _sibling_id(sample_tree, node_id)

    low_pair = next(
        node
        for node in sample_tree.nodes.values()
        if node["kind"] == "internal" and node["linkage_distance"] == pytest.approx(0.10)
    )
    high_pair = next(
        node
        for node in sample_tree.nodes.values()
        if node["kind"] == "internal" and node["linkage_distance"] == pytest.approx(0.20)
    )
    assert high_pair["node_id"] == neighbors[low_pair["node_id"]][0]
    assert all(child not in neighbors[low_pair["node_id"]] for child in low_pair["children"])
    assert set(neighbors[high_pair["node_id"]]) == {low_pair["node_id"]}


def test_height_cap_and_top_k_drop_distant_nodes(sample_tree: Tree) -> None:
    exact = select_neighbors(sample_tree, lambda_value=0, height_cap=0.20)
    leaf = next(node for node in sample_tree.nodes.values() if node["kind"] == "leaf")
    leaf_id = leaf["node_id"]
    assert exact[leaf_id][0] == _sibling_id(sample_tree, leaf_id)
    assert all(
        sample_tree.nodes[node_id]["kind"] == "leaf" for node_id in exact[leaf_id]
    )
    assert set(exact[sample_tree.root_node_id]) == set(
        sample_tree.nodes[sample_tree.root_node_id]["children"]
    )

    capped = select_neighbors(sample_tree, top_k=1)
    assert capped[leaf_id] == (_sibling_id(sample_tree, leaf_id),)
    assert len(capped[sample_tree.root_node_id]) == 1
    assert capped[sample_tree.root_node_id][0] in (
        sample_tree.nodes[sample_tree.root_node_id]["children"]
    )


def test_root_band_includes_a_high_grandchild_and_both_children() -> None:
    documents = [
        Document(text=f"Note {index}.", item_id=f"n{index}", title=f"Note {index}")
        for index in range(4)
    ]
    linkage = [
        [0, 1, 0.92, 2],
        [4, 2, 0.96, 3],
        [3, 5, 1.00, 4],
    ]
    tree = _tree_from_documents(documents, linkage)
    neighbors = select_neighbors(tree)
    root = tree.nodes[tree.root_node_id]
    high_child = next(
        tree.nodes[child]
        for child in root["children"]
        if tree.nodes[child]["kind"] == "internal"
    )
    low_child = next(child for child in root["children"] if child != high_child["node_id"])
    grandchild = high_child["children"][0]
    assert tree.nodes[grandchild]["kind"] == "internal"
    ranked = neighbors[tree.root_node_id]
    assert ranked[:2] == (high_child["node_id"], low_child)
    assert grandchild in ranked
    leaf_under_grandchild = high_child["children"][0]
    buried = tree.nodes[leaf_under_grandchild]["children"][0]
    assert buried not in ranked


def test_bad_selection_parameters_are_rejected(sample_tree: Tree) -> None:
    with pytest.raises(LabelingError, match="lambda"):
        select_neighbors(sample_tree, lambda_value=-0.1)
    with pytest.raises(LabelingError, match="height cap"):
        select_neighbors(sample_tree, height_cap=float("nan"))
    with pytest.raises(LabelingError, match="top_k"):
        select_neighbors(sample_tree, top_k=0)
    with pytest.raises(LabelingError, match="top_k"):
        select_neighbors(sample_tree, top_k=True)  # type: ignore[arg-type]


def test_contrast_on_sixteen_documents_rewrites_a_second_set() -> None:
    tree, _documents = _fixture_tree()
    before = tree.linkage_fingerprint()
    canonical = tree.to_canonical_dict()
    _prepare_bottom_up(tree)
    source_cards = _snapshot_cards(tree, "llm_bottom_up")
    source_labels = dict(tree.labels["llm_bottom_up"])
    source_map = tree.cards["llm_bottom_up"]
    neighbors = select_neighbors(tree)
    client = RecordingClient(title_prefix="Apart")

    cards = label_contrast(tree, client=client)

    assert tree.linkage_fingerprint() == before
    assert tree.to_canonical_dict() == canonical
    assert tree.cards["llm_bottom_up"] is source_map
    assert _snapshot_cards(tree, "llm_bottom_up") == source_cards
    assert tree.labels["llm_bottom_up"] == source_labels
    assert tree.active_label_set == "llm_contrast"
    assert set(tree.cards) == {"llm_bottom_up", "llm_contrast"}
    assert set(cards) == set(tree.nodes)

    root_children = set(tree.nodes[tree.root_node_id]["children"])
    assert set(neighbors[tree.root_node_id]) == root_children
    called = [node_id for node_id, picked in neighbors.items() if picked]
    assert len(client.prompts) == len(called)
    assert client.models == [DEFAULT_LABELING_MODEL] * len(called)
    for node_id, prompt in zip(called, client.prompts, strict=True):
        expected = render_contrast_prompt(
            source_map[node_id],
            [source_map[neighbor_id] for neighbor_id in neighbors[node_id]],
        )
        assert prompt == expected
        assert cards[node_id].title == tree.labels["llm_contrast"][node_id]
        assert cards[node_id] != source_map[node_id]
    for node_id, picked in neighbors.items():
        if picked:
            continue
        assert cards[node_id] == source_map[node_id]
        assert tree.labels["llm_contrast"][node_id] == source_labels[node_id]
    for node in tree.nodes.values():
        if node["kind"] != "leaf":
            continue
        assert _parent_of(tree, node["node_id"])["node_id"] not in neighbors[node["node_id"]]
        assert neighbors[node["node_id"]][0] == _sibling_id(tree, node["node_id"])


def test_top_k_limits_the_prompt_and_descriptors_can_be_replaced(sample_tree: Tree) -> None:
    _prepare_bottom_up(sample_tree)
    source = {
        node_id: card.to_dict()
        for node_id, card in sample_tree.cards["llm_bottom_up"].items()
    }
    client = RecordingClient(title_prefix="Apart")
    cards = label_contrast(sample_tree, client=client, top_k=1, model="other-model")
    neighbors = select_neighbors(sample_tree, top_k=1)
    called = [node_id for node_id, picked in neighbors.items() if picked]
    assert client.models == ["other-model"] * len(called)
    assert all(len(picked) == 1 for picked in neighbors.values() if picked)
    for node_id, prompt in zip(called, client.prompts, strict=True):
        assert prompt == render_contrast_prompt(
            sample_tree.cards["llm_bottom_up"][node_id],
            [sample_tree.cards["llm_bottom_up"][neighbors[node_id][0]]],
        )
    rewritten = cards[called[0]]
    assert rewritten.scope == ("Apart scope 1",)
    assert source[called[0]]["scope"] == ["Card scope 1"]
    assert sample_tree.cards["llm_bottom_up"][called[0]].to_dict() == source[called[0]]


def test_contrast_cards_round_trip_beside_the_source_set(tmp_path: Path) -> None:
    tree, documents = _fixture_tree()
    label_bottom_up(tree, client=RecordingClient())
    contrast_client = RecordingClient()
    cards = label_contrast(tree, client=contrast_client)
    orchard = Orchard.from_trees(documents=documents, trees={"labeling": tree})
    loaded = Orchard.load(orchard.save(tmp_path / "art")).tree("labeling")
    assert loaded.linkage_fingerprint() == tree.linkage_fingerprint()
    assert loaded.cards["llm_bottom_up"] == tree.cards["llm_bottom_up"]
    assert loaded.cards["llm_contrast"] == cards
    assert loaded.labels["llm_contrast"] == {
        node_id: card.title for node_id, card in cards.items()
    }
    assert loaded.active_label_set == "llm_contrast"


def test_style_warning_does_not_retry() -> None:
    tree = _two_leaf_tree()
    _prepare_bottom_up(tree)
    warning = _card_json(title="The unified desk", summary="Routes letters.")
    client = RecordingClient(responses=[warning, warning, warning])
    cards = label_contrast(tree, client=client)
    assert len(client.prompts) == 3
    assert client._responses == []
    assert all(card.style_warnings for card in cards.values())


def test_contrast_over_the_failure_limit_keeps_successes_and_the_source() -> None:
    tree = _two_leaf_tree()
    _prepare_bottom_up(tree)
    before = tree.linkage_fingerprint()
    source_map = tree.cards["llm_bottom_up"]
    source_cards = _snapshot_cards(tree, "llm_bottom_up")
    source_labels = dict(tree.labels["llm_bottom_up"])
    good = _card_json(title="Apart routing", summary="Keeps letters moving.")
    client = RecordingClient(responses=[good, "", good])
    neighbors = select_neighbors(tree)
    sent = [node_id for node_id, picked in neighbors.items() if picked]
    with pytest.raises(LabelingError, match="failure rate"):
        label_contrast(tree, client=client)
    failed = sent[1]
    assert len(client.prompts) == len(sent) + 1
    assert failed not in tree.cards["llm_contrast"]
    assert sent[0] in tree.cards["llm_contrast"]
    assert sent[2] in tree.cards["llm_contrast"]
    assert len(tree.cards["llm_contrast"]) == len(tree.nodes) - 1
    assert tree.active_label_set == "llm_bottom_up"
    assert tree.cards["llm_bottom_up"] is source_map
    assert _snapshot_cards(tree, "llm_bottom_up") == source_cards
    assert tree.labels["llm_bottom_up"] == source_labels
    assert tree.linkage_fingerprint() == before


def test_contrast_single_failure_keeps_the_source_card() -> None:
    tree, _documents = _fixture_tree()
    _prepare_bottom_up(tree)
    source = tree.cards["llm_bottom_up"]
    neighbors = select_neighbors(tree)
    sent = [node_id for node_id, picked in neighbors.items() if picked]
    assert len(sent) > 25
    before = tree.linkage_fingerprint()

    class FailFirst:
        def __init__(self) -> None:
            self.target: str | None = None
            self.prompts: list[LabelingPrompt] = []
            self.attempts = 0

        def complete(self, *, model: str, prompt: LabelingPrompt) -> str:
            self.prompts.append(prompt)
            if self.target is None:
                self.target = prompt.user
            if prompt.user == self.target:
                self.attempts += 1
                return "not-json"
            return _card_json(
                title=f"Apart {len(self.prompts)}",
                summary="Moved away from the neighbor.",
            )

        def complete_many(
            self,
            *,
            model: str,
            prompts: list[tuple[int, LabelingPrompt]],
            max_workers: int,
        ) -> dict[int, str]:
            ordered = sorted(prompts, key=lambda item: item[0])
            return {
                index: self.complete(model=model, prompt=prompt)
                for index, prompt in ordered
            }

    client = FailFirst()
    cards = label_contrast(tree, client=client)
    assert client.attempts == 2
    assert cards[sent[0]] is source[sent[0]]
    assert cards[sent[1]] != source[sent[1]]
    assert len(cards) == len(tree.nodes)
    assert tree.cards["llm_bottom_up"] is source
    assert tree.linkage_fingerprint() == before


def test_missing_source_set_refuses_before_any_call() -> None:
    tree = _two_leaf_tree()
    client = RecordingClient()
    with pytest.raises(LabelingError, match="source set 'llm_bottom_up' is missing"):
        label_contrast(tree, client=client)
    leaf = next(node for node in tree.nodes.values() if node["kind"] == "leaf")
    tree.set_cards("llm_bottom_up", {leaf["node_id"]: _card()})
    with pytest.raises(LabelingError, match="missing a card"):
        label_contrast(tree, client=client)
    with pytest.raises(LabelingError, match="must differ from the source set"):
        label_contrast(tree, client=client, source_set="partial", name="partial")
    assert client.prompts == []
    assert "llm_contrast" not in tree.cards


def test_single_node_copies_the_card_without_a_client() -> None:
    document = Document(text="One note.", item_id="solo", title="Solo")
    tree = Tree.from_linkage(
        [],
        item_ids=["solo"],
        documents=[document],
    )
    card = _card(title="Solo note", summary="One note sits alone.")
    tree.set_cards("llm_bottom_up", {tree.root_node_id: card})
    tree.set_labels("llm_bottom_up", {tree.root_node_id: card.title})
    before = tree.linkage_fingerprint()
    cards = label_contrast(tree)
    assert cards == {tree.root_node_id: card}
    assert tree.labels["llm_contrast"][tree.root_node_id] == "Solo note"
    assert tree.cards["llm_bottom_up"][tree.root_node_id] is card
    assert tree.linkage_fingerprint() == before


def test_empty_model_and_name_are_rejected_before_any_call() -> None:
    tree = _two_leaf_tree()
    _prepare_bottom_up(tree)
    client = RecordingClient()
    with pytest.raises(LabelingError, match="model id"):
        label_contrast(tree, client=client, model="  ")
    with pytest.raises(InvalidIdentityError, match="label set name"):
        label_contrast(tree, client=client, name=" ")
    with pytest.raises(LabelingError, match="complete"):
        label_contrast(tree, client=object())  # type: ignore[arg-type]
    assert client.prompts == []
    assert "llm_contrast" not in tree.cards


def test_missing_openai_names_the_labeling_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    tree = _two_leaf_tree()
    _prepare_bottom_up(tree)
    before = tree.linkage_fingerprint()
    source = _snapshot_cards(tree, "llm_bottom_up")
    with pytest.raises(MissingOptionalDependencyError, match=r"orchardlogic\[labeling\]") as caught:
        label_contrast(tree)
    assert caught.value.extra == "labeling"
    assert "llm_contrast" not in tree.cards
    assert _snapshot_cards(tree, "llm_bottom_up") == source
    assert tree.linkage_fingerprint() == before


@pytest.mark.live_labeling
def test_live_contrast_on_sixteen_documents() -> None:
    import os

    if os.environ.get("ORCHARD_LIVE_LABELING") != "1" or not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("live labeling skipped")
    tree, _documents = _fixture_tree()
    label_bottom_up(tree, client=RecordingClient())
    source = _snapshot_cards(tree, "llm_bottom_up")
    before = tree.linkage_fingerprint()
    cards = label_contrast(tree)
    assert tree.linkage_fingerprint() == before
    assert _snapshot_cards(tree, "llm_bottom_up") == source
    assert set(cards) == set(tree.nodes)
    assert tree.active_label_set == "llm_contrast"
    neighbors = select_neighbors(tree)
    assert neighbors[tree.root_node_id]
    for node_id, card in cards.items():
        assert tree.labels["llm_contrast"][node_id] == card.title
        assert card.schema_version == CARD_SCHEMA_VERSION
        if neighbors[node_id]:
            assert card.title
            assert card.summary
