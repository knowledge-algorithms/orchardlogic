"""Bottom-up labeling with a fake client. No AppWorld corpus and no network."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

from orchardlogic.cultivate import Document, Orchard, Tree, estimate_labeling, label_bottom_up, label_intrinsic
from orchardlogic.cultivate.exceptions import InvalidIdentityError, LabelingError, MissingOptionalDependencyError
from orchardlogic.cultivate.labeling.cards import CARD_SCHEMA_VERSION, NodeCard
from orchardlogic.cultivate.labeling.client import DEFAULT_MAX_WORKERS, LabelingPrompt, OpenAILabelingClient
from orchardlogic.cultivate.labeling.cost import DEFAULT_LABELING_MODEL, LEGACY_LABELING_MODEL
from orchardlogic.cultivate.labeling.dispatch import CHECKPOINT_SCHEMA_VERSION
from orchardlogic.cultivate.labeling.prompts import (
    BOTTOM_UP_LEAF_PROMPT_VERSION,
    BOTTOM_UP_PARENT_PROMPT_VERSION,
    format_document_for_prompt,
    render_leaf_prompt,
    render_parent_prompt,
)
from orchardlogic.cultivate.schemas import CARD_SET_SCHEMA_VERSION, LABEL_SET_SCHEMA_VERSION

FIXTURE = Path(__file__).parent / "fixtures" / "labeling_16.json"


class RecordingClient:
    """Returns one valid card per call and records the prompt."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        fail_indexes: set[int] | None = None,
    ) -> None:
        self._responses = list(responses) if responses is not None else None
        self._fail_indexes = fail_indexes
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
                return self._responses.pop(0)
            return "not-json"
        index = len(self.prompts)
        return _card_json(title=f"Card {index:02d}", summary=f"Summary {index:02d}.")

    def complete_many(
        self,
        *,
        model: str,
        prompts: list[tuple[int, LabelingPrompt]],
        max_workers: int,
    ) -> dict[int, str]:
        results: dict[int, str] = {}
        for index, prompt in prompts:
            if self._fail_indexes is not None and index in self._fail_indexes:
                self.models.append(model)
                self.prompts.append(prompt)
                self._bad_prompts.add((prompt.system, prompt.user))
                results[index] = "not-json"
                continue
            results[index] = self.complete(model=model, prompt=prompt)
        return results


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


def _generations(tree: Tree) -> list[list[dict]]:
    leaves = {
        node["item_id"]: node for node in tree.nodes.values() if node["kind"] == "leaf"
    }
    layers: list[list[dict]] = [[leaves[item_id] for item_id in tree.item_ids]]
    pending = {
        node["node_id"]: node
        for node in tree.nodes.values()
        if node["kind"] == "internal"
    }
    done = {node["node_id"] for node in layers[0]}
    while pending:
        layer = [
            node
            for node in pending.values()
            if all(child in done for child in node["children"])
        ]
        layer.sort(key=lambda node: int(node["linkage_row"]))
        assert layer
        layers.append(layer)
        for node in layer:
            done.add(node["node_id"])
            del pending[node["node_id"]]
    return layers


def _assert_prompts_match_generations(
    tree: Tree,
    client: RecordingClient,
    cards: dict[str, NodeCard],
    *,
    name: str,
    model: str = DEFAULT_LABELING_MODEL,
) -> None:
    order = [node for layer in _generations(tree) for node in layer]
    assert len(client.prompts) == len(order) == len(tree.nodes)
    assert client.models == [model] * len(order)
    produced: dict[str, NodeCard] = {}
    for node, prompt in zip(order, client.prompts, strict=True):
        if node["kind"] == "leaf":
            document = tree.documents_by_id[node["item_id"]]
            block = format_document_for_prompt(
                text=document.text,
                title=document.title,
                item_id=document.item_id,
            )
            assert prompt == render_leaf_prompt(block)
        else:
            left_id, right_id = node["children"]
            assert prompt == render_parent_prompt(produced[left_id], produced[right_id])
        card = cards[node["node_id"]]
        assert isinstance(card, NodeCard)
        assert tree.labels[name][node["node_id"]] == card.title
        assert tree.cards[name][node["node_id"]] == card
        produced[node["node_id"]] = card
    assert all(
        BOTTOM_UP_LEAF_PROMPT_VERSION in prompt.system
        for prompt in client.prompts[: tree.leaf_count]
    )
    assert all(
        BOTTOM_UP_LEAF_PROMPT_VERSION not in prompt.system
        for prompt in client.prompts[tree.leaf_count :]
    )


def test_bottom_up_on_sixteen_documents_labels_layers_without_touching_linkage() -> None:
    tree, _documents = _fixture_tree()
    before = tree.linkage_fingerprint()
    canonical = tree.to_canonical_dict()
    label_intrinsic(tree, name="intrinsic_v1")
    client = RecordingClient()
    cards = label_bottom_up(tree, client=client)

    assert tree.linkage_fingerprint() == before
    assert tree.to_canonical_dict() == canonical
    assert (
        estimate_labeling(tree, model=LEGACY_LABELING_MODEL).bottom_up_calls
        == len(client.prompts)
        == 31
    )
    assert tree.active_label_set == "llm_bottom_up"
    assert "intrinsic_v1" in tree.labels
    _assert_prompts_match_generations(tree, client, cards, name="llm_bottom_up")

    parent_prompts = client.prompts[tree.leaf_count :]
    assert "flour-and-water" in client.prompts[0].user
    assert all("flour-and-water" not in prompt.user for prompt in parent_prompts)
    assert tree.root_node_id in cards
    assert tree.label_for(tree.root_node_id) == cards[tree.root_node_id].title


def test_shallower_merge_runs_before_an_earlier_deeper_row() -> None:
    """Row 2 merges two leaves. Row 1 is already a deeper branch. Layers flip that order."""
    documents = [
        Document(text=f"Note {index}.", item_id=f"s{index}", title=f"Note {index}")
        for index in range(5)
    ]
    linkage = [
        [0, 1, 0.1, 2],
        [5, 2, 0.2, 3],
        [3, 4, 0.15, 2],
        [6, 7, 0.4, 5],
    ]
    tree = _tree_from_documents(documents, linkage)
    before = tree.linkage_fingerprint()
    client = RecordingClient()
    cards = label_bottom_up(tree, client=client, model="other-model", name="field_notes")

    assert tree.linkage_fingerprint() == before
    assert client.models == ["other-model"] * len(tree.nodes)
    assert tree.active_label_set == "field_notes"
    _assert_prompts_match_generations(
        tree,
        client,
        cards,
        name="field_notes",
        model="other-model",
    )

    def node_for(members: set[str]) -> dict:
        return next(
            node
            for node in tree.nodes.values()
            if set(node["descendant_item_ids"]) == members
        )

    late = node_for({"s3", "s4"})
    deep = node_for({"s0", "s1", "s2"})
    late_prompt = render_parent_prompt(
        cards[late["children"][0]],
        cards[late["children"][1]],
    )
    deep_prompt = render_parent_prompt(
        cards[deep["children"][0]],
        cards[deep["children"][1]],
    )
    assert client.prompts.index(late_prompt) < client.prompts.index(deep_prompt)


def test_cards_round_trip_through_save_and_old_manifests_keep_labels(tmp_path: Path) -> None:
    tree, documents = _fixture_tree()
    client = RecordingClient()
    cards = label_bottom_up(tree, client=client)
    orchard = Orchard.from_trees(documents=documents, trees={"labeling": tree})
    saved = orchard.save(tmp_path / "art")

    label_path = saved / "trees" / "labeling" / "labels" / "llm_bottom_up.json"
    card_path = saved / "trees" / "labeling" / "cards" / "llm_bottom_up.json"
    label_payload = json.loads(label_path.read_text(encoding="utf-8"))
    card_payload = json.loads(card_path.read_text(encoding="utf-8"))
    assert label_payload["schema_version"] == LABEL_SET_SCHEMA_VERSION == "orchard_label_set_v1"
    assert set(label_payload) == {"schema_version", "name", "labels"}
    assert all(isinstance(value, str) for value in label_payload["labels"].values())
    assert card_payload["schema_version"] == CARD_SET_SCHEMA_VERSION == "orchard_card_set_v1"
    assert set(card_payload) == {"schema_version", "name", "cards"}

    loaded = Orchard.load(saved).tree("labeling")
    assert loaded.linkage_fingerprint() == tree.linkage_fingerprint()
    assert loaded.active_label_set == "llm_bottom_up"
    assert loaded.cards["llm_bottom_up"] == cards
    assert loaded.labels["llm_bottom_up"] == {
        node_id: card.title for node_id, card in cards.items()
    }

    manifest_path = saved / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["trees"]:
        entry.pop("card_sets")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    restored = Orchard.load(saved).tree("labeling")
    assert restored.cards == {}
    assert restored.labels["llm_bottom_up"][tree.root_node_id] == cards[tree.root_node_id].title


def test_subtree_copies_overlapping_cards() -> None:
    tree, _documents = _fixture_tree()
    label_bottom_up(tree, client=RecordingClient())
    bread = {"bread-starter", "bread-shape", "bread-score", "bread-oven"}
    quad = next(
        node for node in tree.nodes.values() if set(node["descendant_item_ids"]) == bread
    )
    bike = next(node for node in tree.nodes.values() if node.get("item_id") == "bike-flat")
    before = tree.linkage_fingerprint()

    sub = tree.subtree(quad["node_id"])
    assert sub.root_node_id == quad["node_id"]
    assert set(sub.cards["llm_bottom_up"]) <= set(sub.nodes)
    assert quad["node_id"] in sub.cards["llm_bottom_up"]
    assert sub.cards["llm_bottom_up"][quad["node_id"]] == (
        tree.cards["llm_bottom_up"][quad["node_id"]]
    )
    assert bike["node_id"] not in sub.cards["llm_bottom_up"]
    assert bike["node_id"] in tree.cards["llm_bottom_up"]

    leaf = tree.subtree(bike["node_id"])
    assert set(leaf.cards["llm_bottom_up"]) == {bike["node_id"]}
    assert leaf.cards["llm_bottom_up"][bike["node_id"]] == (
        tree.cards["llm_bottom_up"][bike["node_id"]]
    )
    assert tree.linkage_fingerprint() == before


def test_title_and_summary_only_response_fills_empty_descriptors() -> None:
    tree = _two_leaf_tree()
    raw = json.dumps({"title": "Desk route", "summary": "Routes letters."})
    kept = _card_json(title="Brake pads", summary="Aligns the pads.", scope=["pads"])
    client = RecordingClient(responses=[raw, kept, raw])
    cards = label_bottom_up(tree, client=client)
    titles = {card.title: card for card in cards.values()}
    assert titles["Desk route"].scope == ()
    assert titles["Desk route"].schema_version == CARD_SCHEMA_VERSION
    assert titles["Brake pads"].scope == ("pads",)


def test_style_warnings_do_not_retry() -> None:
    tree = _two_leaf_tree()
    warning = _card_json(title="unified desk", summary="Routes letters.")
    client = RecordingClient(responses=[warning, warning, warning])
    cards = label_bottom_up(tree, client=client)
    assert len(client.prompts) == 3
    assert client._responses == []
    assert all(card.style_warnings for card in cards.values())
    assert all(tree.labels["llm_bottom_up"][node_id] == "unified desk" for node_id in cards)


@pytest.mark.parametrize(
    "bad",
    ["not-json", "", _card_json(title="  ")],
)
def test_hard_failure_keeps_partial_cards_when_rate_exceeds_limit(bad: str) -> None:
    tree = _two_leaf_tree()
    before = tree.linkage_fingerprint()
    good = _card_json(title="Mail routing", summary="Routes letters.")
    client = RecordingClient(responses=[good, bad, bad])
    with pytest.raises(LabelingError, match="failure rate|reached the limit"):
        label_bottom_up(tree, client=client)
    assert len(client.prompts) >= 2
    assert len(tree.cards.get("llm_bottom_up", {})) == 1
    assert tree.linkage_fingerprint() == before


def test_parent_before_children_does_not_call_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _two_leaf_tree()
    parent = next(node for node in tree.nodes.values() if node["kind"] == "internal")

    def parents_first(_tree: Tree) -> list[list[dict]]:
        return [[parent]]

    monkeypatch.setattr("orchardlogic.cultivate.labeling.bottom_up._bottom_up_layers", parents_first)
    client = RecordingClient()
    with pytest.raises(LabelingError, match="before both child cards exist"):
        label_bottom_up(tree, client=client)
    assert client.prompts == []
    assert tree.cards == {}


def test_empty_model_and_name_are_rejected_before_any_call() -> None:
    tree = _two_leaf_tree()
    client = RecordingClient()
    with pytest.raises(LabelingError, match="model id"):
        label_bottom_up(tree, client=client, model="  ")
    with pytest.raises(InvalidIdentityError, match="label set name"):
        label_bottom_up(tree, client=client, name=" ")
    with pytest.raises(LabelingError, match="complete"):
        label_bottom_up(tree, client=object())  # type: ignore[arg-type]
    assert client.prompts == []


def test_missing_openai_names_the_labeling_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    tree = _two_leaf_tree()
    before = tree.linkage_fingerprint()
    with pytest.raises(MissingOptionalDependencyError, match=r"orchardlogic\[labeling\]") as caught:
        label_bottom_up(tree)
    assert caught.value.extra == "labeling"
    assert tree.cards == {}
    assert tree.linkage_fingerprint() == before


def test_openai_adapter_sends_one_json_completion() -> None:
    content = _card_json(title="Desk", summary="Routes mail.")
    prompt = LabelingPrompt(system="Label rules.", user="Document:\nhello")

    class _Completions:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def create(self, **kwargs: object) -> object:
            self.kwargs = kwargs
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = _Completions()
    chat = type("Chat", (), {"completions": completions})()
    sdk = type("SDK", (), {"chat": chat})()
    client = OpenAILabelingClient(sdk_client=sdk)
    assert client.complete(model=DEFAULT_LABELING_MODEL, prompt=prompt) == content
    assert completions.kwargs is not None
    assert completions.kwargs["model"] == DEFAULT_LABELING_MODEL
    assert completions.kwargs["messages"] == [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["reasoning_effort"] == "none"
    assert "reasoning" not in completions.kwargs
    assert "temperature" not in completions.kwargs
    assert DEFAULT_LABELING_MODEL == "gpt-6-luna"

    many_calls: list[dict[str, object]] = []

    def create_many(**kwargs: object) -> object:
        many_calls.append(kwargs)
        user = kwargs["messages"][1]["content"]  # type: ignore[index]
        if user == "bad":
            return type("Response", (), {"choices": []})()
        message = type("Message", (), {"content": str(user)})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()

    completions_type = type(
        "Completions",
        (),
        {"create": staticmethod(create_many)},
    )
    chat_type = type("Chat", (), {"completions": completions_type()})
    pooled = OpenAILabelingClient(sdk_client=type("SDK", (), {"chat": chat_type()})())
    good = LabelingPrompt(system="Label rules.", user="good-document")
    bad = LabelingPrompt(system="Label rules.", user="bad")
    pooled_result = pooled.complete_many(
        model=DEFAULT_LABELING_MODEL,
        prompts=[(2, bad), (0, good)],
        max_workers=4,
    )
    assert pooled_result == {0: "good-document"}
    assert len(many_calls) == 2
    assert all(call["model"] == "gpt-6-luna" for call in many_calls)
    assert all(call["reasoning_effort"] == "none" for call in many_calls)
    assert all("reasoning" not in call for call in many_calls)
    assert all("temperature" not in call for call in many_calls)
    assert all(
        call["messages"][0]["role"] == "system" and call["messages"][1]["role"] == "user"  # type: ignore[index]
        for call in many_calls
    )

    empty = OpenAILabelingClient(
        sdk_client=type(
            "SDK",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {
                        "completions": type(
                            "Completions",
                            (),
                            {
                                "create": staticmethod(
                                    lambda **_kwargs: type(
                                        "Response",
                                        (),
                                        {"choices": []},
                                    )()
                                )
                            },
                        )()
                    },
                )()
            },
        )()
    )
    with pytest.raises(LabelingError, match="empty model response"):
        empty.complete(model=DEFAULT_LABELING_MODEL, prompt=prompt)


def test_openai_import_stays_inside_the_live_call() -> None:
    module = ast.parse(inspect.getsource(sys.modules["orchardlogic.cultivate.labeling.client"]))
    for node in module.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "openai" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "openai"


class _SelectiveFailureClient:
    """Fails leaf prompts whose user text contains a marker."""

    def __init__(self, markers: set[str], *, fail_limit: int | None = None) -> None:
        self.markers = markers
        self.fail_limit = fail_limit
        self.failures = 0
        self.prompts: list[LabelingPrompt] = []
        self.models: list[str] = []
        self.wave_sizes: list[int] = []
        self.max_workers_seen: list[int] = []

    def complete(self, *, model: str, prompt: LabelingPrompt) -> str:
        self.models.append(model)
        self.prompts.append(prompt)
        leaf = BOTTOM_UP_LEAF_PROMPT_VERSION in prompt.system
        marked = leaf and any(marker in prompt.user for marker in self.markers)
        if marked and (self.fail_limit is None or self.failures < self.fail_limit):
            self.failures += 1
            return "not-json"
        index = len(self.prompts)
        return _card_json(title=f"Card {index:02d}", summary=f"Summary {index:02d}.")

    def complete_many(
        self,
        *,
        model: str,
        prompts: list[tuple[int, LabelingPrompt]],
        max_workers: int,
    ) -> dict[int, str]:
        self.wave_sizes.append(len(prompts))
        self.max_workers_seen.append(max_workers)
        return {
            index: self.complete(model=model, prompt=prompt)
            for index, prompt in sorted(prompts)
        }


def test_one_generation_is_concurrent_and_the_next_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree, _documents = _fixture_tree()
    parent_renders = {"count": 0}
    real_render = render_parent_prompt

    def counting(child_a: NodeCard, child_b: NodeCard) -> LabelingPrompt:
        parent_renders["count"] += 1
        return real_render(child_a, child_b)

    monkeypatch.setattr("orchardlogic.cultivate.labeling.bottom_up.render_parent_prompt", counting)

    class BarrierClient(RecordingClient):
        def __init__(self) -> None:
            super().__init__()
            self.wave_sizes: list[int] = []
            self.max_workers_seen: list[int] = []

        def complete_many(
            self,
            *,
            model: str,
            prompts: list[tuple[int, LabelingPrompt]],
            max_workers: int,
        ) -> dict[int, str]:
            systems = {prompt.system for _, prompt in prompts}
            assert len(systems) == 1
            system = next(iter(systems))
            if BOTTOM_UP_LEAF_PROMPT_VERSION in system:
                assert parent_renders["count"] == 0
            else:
                assert BOTTOM_UP_PARENT_PROMPT_VERSION in system
                assert parent_renders["count"] > 0
            self.wave_sizes.append(len(prompts))
            self.max_workers_seen.append(max_workers)
            return super().complete_many(
                model=model,
                prompts=prompts,
                max_workers=max_workers,
            )

    client = BarrierClient()
    cards = label_bottom_up(tree, client=client)
    generations = _generations(tree)
    assert client.wave_sizes == [len(layer) for layer in generations]
    assert client.max_workers_seen == [DEFAULT_MAX_WORKERS] * len(generations)
    _assert_prompts_match_generations(tree, client, cards, name="llm_bottom_up")


def test_single_failure_on_fixture_tree_uses_fallback_and_parent_sees_it() -> None:
    tree, _documents = _fixture_tree()
    before = tree.linkage_fingerprint()
    client = _SelectiveFailureClient({"Sourdough starter"}, fail_limit=2)
    cards = label_bottom_up(tree, client=client)

    assert tree.linkage_fingerprint() == before
    assert len(cards) == len(tree.nodes)
    assert client.failures == 2
    assert client.wave_sizes[0] == tree.leaf_count
    assert len(client.wave_sizes) == len(_generations(tree))
    leaf_id = next(
        node["node_id"]
        for node in tree.nodes.values()
        if node.get("item_id") == "bread-starter"
    )
    fallback = cards[leaf_id]
    assert fallback.title == "Sourdough starter"
    assert fallback.summary.startswith("Feed a flour-and-water")
    assert fallback.scope == ()
    assert fallback.capabilities == ()
    assert fallback.concepts == ()
    assert fallback.uncertainty == ()
    parents = [
        prompt
        for prompt in client.prompts
        if BOTTOM_UP_PARENT_PROMPT_VERSION in prompt.system
    ]
    assert any(
        "Sourdough starter" in prompt.user and "Feed a flour-and-water" in prompt.user
        for prompt in parents
    )


def test_high_failure_rate_keeps_checkpoint_and_resume_skips_done_nodes(
    tmp_path: Path,
) -> None:
    tree, _documents = _fixture_tree()
    before = tree.linkage_fingerprint()
    client = _SelectiveFailureClient({"Sourdough starter", "Loaf shaping"})
    with pytest.raises(LabelingError, match="failure rate"):
        label_bottom_up(tree, client=client, checkpoint_dir=tmp_path)

    assert tree.linkage_fingerprint() == before
    assert client.wave_sizes == [tree.leaf_count]
    assert all(
        BOTTOM_UP_PARENT_PROMPT_VERSION not in prompt.system for prompt in client.prompts
    )
    saved = dict(tree.cards["llm_bottom_up"])
    assert len(saved) == tree.leaf_count - 2
    failed_ids = {
        node["node_id"]
        for node in tree.nodes.values()
        if node.get("item_id") in {"bread-starter", "bread-shape"}
    }
    assert failed_ids.isdisjoint(saved)
    checkpoint = json.loads((tmp_path / "llm_bottom_up.json").read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["name"] == "llm_bottom_up"
    assert set(checkpoint["cards"]) == set(saved)
    assert set(checkpoint["request_indexes"]) == set(saved) | failed_ids
    assert all(type(index) is int for index in checkpoint["request_indexes"].values())
    assert {record["node_id"] for record in checkpoint["errors"]} == failed_ids
    assert all(
        set(record) == {"node_id", "request_index", "message", "fallback"}
        and record["fallback"] is False
        and type(record["request_index"]) is int
        for record in checkpoint["errors"]
    )

    tree.cards.clear()
    tree.labels.clear()
    tree.active_label_set = None
    second = RecordingClient()
    cards = label_bottom_up(tree, client=second, checkpoint_dir=tmp_path)
    assert len(cards) == len(tree.nodes)
    assert len(second.prompts) == len(tree.nodes) - len(saved)
    assert tree.linkage_fingerprint() == before
    leaf_prompts = [
        prompt
        for prompt in second.prompts
        if BOTTOM_UP_LEAF_PROMPT_VERSION in prompt.system
    ]
    assert len(leaf_prompts) == 2
    assert any("Sourdough starter" in prompt.user for prompt in leaf_prompts)
    assert any("Loaf shaping" in prompt.user for prompt in leaf_prompts)


@pytest.mark.live_labeling
def test_live_bottom_up_on_sixteen_documents() -> None:
    import os

    if os.environ.get("ORCHARD_LIVE_LABELING") != "1" or not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("live labeling skipped")
    tree, _documents = _fixture_tree()
    before = tree.linkage_fingerprint()
    cards = label_bottom_up(tree)
    assert tree.linkage_fingerprint() == before
    assert len(cards) == len(tree.nodes)
    assert set(tree.labels["llm_bottom_up"]) == set(tree.nodes)
    for node_id, card in cards.items():
        assert tree.labels["llm_bottom_up"][node_id] == card.title
        assert card.schema_version == CARD_SCHEMA_VERSION
