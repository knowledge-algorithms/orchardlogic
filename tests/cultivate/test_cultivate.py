from __future__ import annotations

import json
import re
import shutil
from collections import Counter

import numpy as np
import pytest
from conftest import topic_perspective_v1, TOPIC_VOCAB, fake_encode, make_topic_docs

from orchardlogic.cultivate import Orchard
from orchardlogic.cultivate.backends.minilm import MiniLMEmbeddingBackend
from orchardlogic.cultivate.cultivate import (
    CultivarIntegrityError,
    cultivate,
    embed_texts,
    embedding_texts,
    place_items,
    read_cultivar,
    recompute_display,
)
from orchardlogic.cultivate.document import Document

FIXED = "2026-11-17T03:00:00Z"
PERSPECTIVE = topic_perspective_v1(
    cuts=[
        {
            "name": "topics",
            "top_criterion": "cluster_count",
            "cluster_count": 4,
            "min_width": 3,
            "max_width": 6,
            "target_width": 5,
            "max_depth": 3,
        }
    ]
)


def _encoder():
    return MiniLMEmbeddingBackend(encode_fn=fake_encode)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("cv") / "cultivar"
    docs = make_topic_docs()
    cultivate(
        docs, PERSPECTIVE, name="test-topic", out_dir=out, encoder=_encoder(), created_at=FIXED
    )
    return out, docs


def _topic(item_id: str) -> str:
    return item_id.split(":")[1].rsplit("-", 1)[0]


def test_layout_and_manifest(built):
    out, docs = built
    cv = read_cultivar(out)
    m = cv.manifest
    assert m["schema_version"] == "orchard_cultivar_v1" and m["created_at"] == FIXED
    assert m["items"]["count"] == len(docs) == 40
    assert "cultivar.json" not in m["files"]
    assert {
        "manifest.json",
        "documents.json",
        "trees/topic/canonical_tree.json",
        "trees/topic/cuts/topics.json",
    } <= set(m["files"])
    space = m["vectors"][0]["space"]
    assert m["vectors"][0]["dim"] == 96 and cv.vectors[space].dtype == np.float32
    sha = cv.vector_sha256[space]
    assert len(sha) == 40 and all(len(s) == 64 for s in sha)
    assert Orchard.load(out).tree("topic").leaf_count == 40  # still a valid orchard artifact
    assert len(m["enrichers"]) == 4
    assert all(len(e["enricher_id"]) == 64 for e in m["enrichers"])


def test_cut_has_four_pure_top_clusters(built):
    out, _ = built
    nodes = read_cultivar(out).cuts["topic"]["topics"]["nodes"]
    by_key = {n["node_key"]: n for n in nodes}
    root = next(n for n in nodes if n["kind"] == "root")
    assert root["descendant_count"] == 40 and root["path"] == []
    top = [n for n in nodes if n["kind"] == "cluster" and n["depth"] == 1]
    assert len(top) == 4
    members: dict[str, list[str]] = {}
    for n in nodes:
        if n["kind"] == "item":
            members.setdefault(n["path"][1], []).append(_topic(n["item_id"]))
    assert len(members) == 4
    assert all(max(Counter(v).values()) == len(v) for v in members.values())
    assert sum(1 for n in nodes if n["kind"] == "item") == 40
    for n in nodes:
        assert n["node_key"].startswith("member_")
        if n["parent_key"] is not None:
            parent = by_key[n["parent_key"]]
            assert n["path"] == [*parent["path"], parent["node_key"]]
            assert n["depth"] == parent["depth"] + 1


def test_display_titles_non_empty_unique_and_readable(built):
    out, _ = built
    cut = read_cultivar(out).cuts["topic"]["topics"]
    nodes = {n["node_key"]: n for n in cut["nodes"]}
    display = cut["display"]
    assert set(display) == {k for k, n in nodes.items() if n["kind"] != "item"}
    siblings: dict[str, list[str]] = {}
    for key, entry in display.items():
        assert entry["title"].strip(), key
        words = [w for w in re.split(r"\s+", entry["title"].replace("·", " ")) if w]
        assert len(words) <= 6, entry["title"]
        assert not re.search(r"https?|\d{3}", entry["title"])
        siblings.setdefault(nodes[key]["parent_key"], []).append(entry["title"].casefold())
    for titles in siblings.values():
        assert len(titles) == len(set(titles))
    vocab_all = {w for v in TOPIC_VOCAB.values() for w in v}
    for key, node in nodes.items():
        if node["kind"] == "cluster" and node["depth"] == 1:
            words = {w.lower() for w in re.split(r"[^\w-]+", display[key]["title"]) if w}
            assert words & vocab_all, display[key]["title"]
            assert display[key]["summary"]


def test_deterministic_artifact_hash_and_input_order(built, tmp_path):
    out, docs = built
    again = tmp_path / "again"
    cultivate(
        list(reversed(docs)),
        PERSPECTIVE,
        name="test-topic",
        out_dir=again,
        encoder=_encoder(),
        created_at=FIXED,
    )
    assert read_cultivar(out).artifact_sha256 == read_cultivar(again).artifact_sha256
    assert (out / "cultivar.json").read_bytes() == (again / "cultivar.json").read_bytes()


def test_plain_callable_encoder_and_non_empty_dir_refused(tmp_path):
    docs = make_topic_docs(per_topic=4)
    out = tmp_path / "c"
    cultivate(docs, PERSPECTIVE, name="x", out_dir=out, encoder=fake_encode, created_at=FIXED)
    with pytest.raises(FileExistsError):
        cultivate(docs, PERSPECTIVE, name="x", out_dir=out, encoder=fake_encode)


def test_integrity_detects_tampering(built, tmp_path):
    out, _ = built
    enrichment = "trees/topic/enrichment/tree.bottom_up_extractive__1.json"
    copy = tmp_path / "enr"
    shutil.copytree(out, copy)
    (copy / enrichment).write_bytes((copy / enrichment).read_bytes().replace(b"a", b"b", 1))
    with pytest.raises(CultivarIntegrityError, match="hash mismatch"):
        read_cultivar(copy)
    copy = tmp_path / "vec"
    shutil.copytree(out, copy)
    target = next((copy / "vectors").iterdir())
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(CultivarIntegrityError, match="hash mismatch"):
        read_cultivar(copy)
    copy = tmp_path / "extra"
    shutil.copytree(out, copy)
    (copy / "stray.txt").write_text("x")
    with pytest.raises(CultivarIntegrityError, match="not listed"):
        read_cultivar(copy)
    copy = tmp_path / "missing"
    shutil.copytree(out, copy)
    (copy / "documents.json").unlink()
    with pytest.raises(CultivarIntegrityError, match="missing"):
        read_cultivar(copy)


def test_place_items_puts_new_doc_under_its_topic_cluster(built):
    out, _ = built
    cv = read_cultivar(out)
    space = cv.manifest["vectors"][0]["space"]
    by_id = dict(zip(cv.vector_ids[space], cv.vectors[space], strict=True))
    nodes = cv.cuts["topic"]["topics"]["nodes"]
    new_docs = [
        Document(
            text="Telescope galaxy orbit comet planet nebula observation tonight under clear "
            "skies and cosmic dust.",
            item_id="n:astro",
            title="Telescope notes",
        ),
        Document(
            text="Skillet garlic butter sauce simmer roast onion recipe in the kitchen oven.",
            item_id="n:cook",
            title="Sauce recipe",
        ),
    ]
    vecs = embed_texts(embedding_texts(new_docs, cv.perspective), fake_encode)
    placements = place_items(nodes, by_id, {d.item_id: vecs[i] for i, d in enumerate(new_docs)})
    rows = {n["node_key"]: n for n in nodes}
    items = [n for n in nodes if n["kind"] == "item"]
    for placement, expect in zip(placements, ("astronomy", "cooking"), strict=True):
        assert placement["node_key"] == f"item:{placement['item_id']}"
        assert placement["path"][-1] == placement["parent_key"]
        assert placement["path"][0] == next(n["node_key"] for n in nodes if n["kind"] == "root")
        assert 0.0 < placement["similarity"] <= 1.0
        under = [n for n in items if placement["parent_key"] in (n["parent_key"], *n["path"])]
        assert under and {_topic(n["item_id"]) for n in under} == {expect}
        assert rows[placement["parent_key"]]["kind"] == "cluster"


def test_place_items_directly_under_root_when_no_clusters():
    nodes = [
        {"node_key": "r", "parent_key": None, "kind": "root", "depth": 0, "position": 0,
         "item_id": None, "descendant_count": 2, "path": []},
        {"node_key": "a", "parent_key": "r", "kind": "item", "depth": 1, "position": 0,
         "item_id": "a", "descendant_count": 1, "path": ["r"]},
    ]  # fmt: skip
    (placement,) = place_items(nodes, {"a": np.array([1.0, 0.0])}, {"new": np.array([0.0, 1.0])})
    assert placement["parent_key"] == "r" and placement["path"] == ["r"]
    assert placement["similarity"] is None


def test_recompute_display_only_subset_keeps_siblings_unique(built):
    out, _ = built
    cv = read_cultivar(out)
    cut = cv.cuts["topic"]["topics"]
    nodes = cut["nodes"]
    spec = cv.perspective["display"]
    assert recompute_display(nodes, cv.node_enrichment["topic"], spec) == cut["display"]
    top = [n["node_key"] for n in nodes if n["kind"] == "cluster" and n["depth"] == 1]
    outputs = json.loads(json.dumps(cv.node_enrichment["topic"]))
    taken = cut["display"][top[1]]["title"]
    for handle in ("tree.side_contrast@1", "tree.bottom_up_extractive@1"):
        outputs[handle][top[0]]["title"] = taken  # force a collision with an untouched sibling
    part = recompute_display(nodes, outputs, spec, only={top[0]}, existing=cut["display"])
    assert set(part) == {top[0]}
    assert part[top[0]]["title"].casefold() != taken.casefold()
    assert part[top[0]]["title"].startswith(taken)


@pytest.mark.optional_model
def test_real_minilm_cultivation(tmp_path):
    from orchardlogic.cultivate.backends.minilm import embeddings_extra_available, minilm_snapshot_available

    if not (embeddings_extra_available() and minilm_snapshot_available()):
        pytest.skip("MiniLM snapshot not available locally")
    out = tmp_path / "real"
    cultivate(make_topic_docs(per_topic=6), PERSPECTIVE, name="real", out_dir=out)
    cut = read_cultivar(out).cuts["topic"]["topics"]
    assert len([n for n in cut["nodes"] if n["kind"] == "cluster" and n["depth"] == 1]) == 4
    assert all(d["title"] for d in cut["display"].values())


def test_knn_placement_stays_in_centroid_top_cluster_and_validates(built):
    out, _ = built
    cv = read_cultivar(out)
    space = cv.manifest["vectors"][0]["space"]
    by_id = dict(zip(cv.vector_ids[space], cv.vectors[space], strict=True))
    nodes = cv.cuts["topic"]["topics"]["nodes"]
    doc = Document(text="Striker midfield goalkeeper penalty league referee stadium tackle and "
                   "dribble in the fixture.", item_id="n:fb", title="Match report")  # fmt: skip
    vec = {"n:fb": embed_texts(embedding_texts([doc], cv.perspective), fake_encode)[0]}
    (knn,) = place_items(nodes, by_id, vec, strategy="knn")
    (cen,) = place_items(nodes, by_id, vec, strategy="centroid")
    assert knn["method"] == "knn" and knn["path"][:2] == cen["path"][:2]
    peers = [n for n in nodes if n["kind"] == "item" and knn["parent_key"] == n["parent_key"]]
    assert {_topic(n["item_id"]) for n in peers} == {"football"}
    with pytest.raises(ValueError):
        place_items(nodes, by_id, vec, strategy="nope")
