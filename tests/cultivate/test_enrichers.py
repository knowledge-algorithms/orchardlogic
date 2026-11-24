from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path

import pytest
from conftest import topic_perspective_v1, fake_encode, make_topic_docs

from orchardlogic.cultivate.backends.minilm import MiniLMEmbeddingBackend
from orchardlogic.cultivate.cultivate import cultivate, read_cultivar
from orchardlogic.cultivate.document import Document
from orchardlogic.cultivate.enrichers import (
    TreeContext,
    UnknownEnricherError,
    available_enrichers,
    enricher_id,
    get_enricher,
)
from orchardlogic.cultivate.exceptions import LabelingError, MissingOptionalDependencyError

BUILTINS = [
    "item.extract@1",
    "tree.bottom_up_extractive@1",
    "tree.bottom_up_extractive@2",
    "tree.llm_bottom_up@1",
    "tree.side_contrast@1",
    "tree.side_contrast@2",
    "tree.top_down_tags@1",
    "tree.top_down_tags@2",
]
CUT = [
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


def _ctx(docs, cut_doc, item_outputs=None, node_outputs=None, vectors=None, only=None):
    return TreeContext(
        tree_id="topic",
        cut=cut_doc,
        docs={d.item_id: d for d in docs},
        item_outputs=item_outputs or {},
        node_outputs=node_outputs or {},
        vectors=vectors,
        only=only,
    )


@pytest.fixture(scope="module")
def cultivated(tmp_path_factory):
    out = tmp_path_factory.mktemp("enr") / "cv"
    docs = make_topic_docs()
    cultivate(
        docs,
        topic_perspective_v1(cuts=CUT),
        name="enr",
        out_dir=out,
        encoder=MiniLMEmbeddingBackend(encode_fn=fake_encode),
        created_at="2026-11-17T03:00:00Z",
    )
    return docs, read_cultivar(out)


def test_registry_and_records():
    assert available_enrichers() == BUILTINS
    with pytest.raises(UnknownEnricherError):
        get_enricher("tree.nope@1")
    for handle in BUILTINS:
        record = get_enricher(handle).record
        assert record["schema_version"] == "orchard_enricher_v1"
        assert f"{record['name']}@{record['version']}" == handle
        assert record["scope"] == ("item" if handle.startswith("item.") else "tree")
        assert record["method"] in ("extractive", "statistical", "llm")
        module = importlib.import_module(record["code"]["module"])
        assert record["code"]["sha256"] == hashlib.sha256(Path(module.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert record["code"]["link"] is None and record["code"]["title"]
        assert re.fullmatch(r"[0-9a-f]{64}", enricher_id(record))
    assert get_enricher("tree.llm_bottom_up@1").record["method"] == "llm"


def test_item_extract_shape_and_content():
    text = (
        "# Quarterly planning notes\n\n"
        "We met on 2026-03-04 and again on March 9, 2026 to plan the quarterly roadmap in detail. "
        "The roadmap prioritises onboarding improvements and the new reporting dashboard for teams. "
        "Nothing else was decided at that meeting beyond the roadmap and the dashboard scope. "
        "Click here to subscribe to our newsletter and accept the cookie policy today friends."
    )
    docs = [
        Document(text=text, item_id="a", metadata={"captured_at": "2026-05-01T12:00:00Z"}),
        Document(text="short", item_id="b", title="Plain &amp; simple"),
    ]
    out = get_enricher("item.extract@1").enrich(docs)
    a, b = out["a"], out["b"]
    assert a["title"] == "Quarterly planning notes"
    assert a["dates"] == ["2026-03-04", "2026-03-09", "2026-05-01"]
    assert a["word_count"] == len(re.findall(r"\w+", text))
    assert 1 <= len(a["summary"].split(". ")) <= 2 and "subscribe" not in a["summary"].lower()
    assert "roadmap" in a["key_terms"] and len(a["key_terms"]) <= 8
    assert b["title"] == "Plain & simple" and b["summary"] == "" and b["dates"] == []
    assert set(a) == {"title", "summary", "dates", "key_terms", "word_count"}


def test_item_extract_title_fallback_first_sentence():
    doc = Document(text="Plain prose without any heading but with a decent length sentence here.")
    (out,) = get_enricher("item.extract@1").enrich([doc]).values()
    assert out["title"].startswith("Plain prose")


def test_tree_enrichers_output_shapes(cultivated):
    docs, cv = cultivated
    cut = cv.cuts["topic"]["topics"]
    keys = {n["node_key"] for n in cut["nodes"] if n["kind"] != "item"}
    nodes = cv.node_enrichment["topic"]
    bu = nodes["tree.bottom_up_extractive@1"]
    assert set(bu) == keys
    for out in bu.values():
        assert set(out) == {"title", "summary", "representative_item_ids", "key_terms"}
        assert len(out["representative_item_ids"]) <= 3 and len(out["key_terms"]) <= 8
    tags = nodes["tree.top_down_tags@1"]
    assert all(set(o) == {"tags", "context_path"} for o in tags.values())
    contrast = nodes["tree.side_contrast@1"]
    assert all(set(o) == {"title", "distinct_terms"} and len(o["distinct_terms"]) <= 5
               for o in contrast.values())
    root = next(n["node_key"] for n in cut["nodes"] if n["kind"] == "root")
    assert contrast[root] == {"title": "", "distinct_terms": []}
    assert tags[root]["context_path"] == []
    deep = next(n for n in cut["nodes"] if n["kind"] == "cluster" and n["depth"] == 2)
    parent = deep["parent_key"]
    assert tags[deep["node_key"]]["context_path"] == [bu[parent]["title"]] or \
        tags[deep["node_key"]]["context_path"] == [contrast[parent]["title"]]


def test_enrichers_standalone_and_only_subset(cultivated):
    docs, cv = cultivated
    cut = cv.cuts["topic"]["topics"]
    space = cv.manifest["vectors"][0]["space"]
    vectors = dict(zip(cv.vector_ids[space], cv.vectors[space], strict=True))
    top = next(n["node_key"] for n in cut["nodes"] if n["kind"] == "cluster" and n["depth"] == 1)
    full_ctx = _ctx(docs, cut, cv.item_enrichment, vectors=vectors)
    part_ctx = _ctx(docs, cut, cv.item_enrichment, vectors=vectors, only=frozenset({top}))
    for handle in ("tree.bottom_up_extractive@1", "tree.side_contrast@1"):
        enricher = get_enricher(handle)
        full, part = enricher.enrich(full_ctx), enricher.enrich(part_ctx)
        assert set(part) == {top}
        assert part[top] == full[top] == cv.node_enrichment["topic"][handle][top]


def test_labels_avoid_boilerplate_urls_numbers(tmp_path):
    junk = " Click here to subscribe https://example.com/page/12345 accept cookie policy 2024 99887."
    docs = [Document(text=d.text + junk, item_id=d.item_id, title=d.title) for d in make_topic_docs()]
    out = tmp_path / "junk"
    cultivate(docs, topic_perspective_v1(cuts=CUT), name="junk", out_dir=out,
              encoder=fake_encode, created_at="2026-11-17T03:00:00Z")
    cv = read_cultivar(out)
    bad = re.compile(r"click|subscribe|cookie|https?|example|\d", re.IGNORECASE)
    for display in cv.cuts["topic"]["topics"]["display"].values():
        assert not bad.search(display["title"]), display["title"]
        assert not bad.search(" ".join(display["tags"]))


def test_single_item_cluster_uses_item_title_and_siblings_deduplicate():
    docs = [Document(text=f"alpha beta gamma delta epsilon item {n} here", item_id=f"i{n}",
                     title="One very long document title that keeps going and going on") for n in range(3)]
    nodes = [
        {"node_key": "r", "parent_key": None, "kind": "root", "depth": 0, "position": 0,
         "item_id": None, "descendant_count": 3, "path": []},
        {"node_key": "c0", "parent_key": "r", "kind": "cluster", "depth": 1, "position": 0,
         "item_id": None, "descendant_count": 1, "path": ["r"]},
        {"node_key": "c1", "parent_key": "r", "kind": "cluster", "depth": 1, "position": 1,
         "item_id": None, "descendant_count": 1, "path": ["r"]},
        {"node_key": "c2", "parent_key": "r", "kind": "cluster", "depth": 1, "position": 2,
         "item_id": None, "descendant_count": 1, "path": ["r"]},
    ]
    for n, doc in enumerate(docs):
        nodes.append({"node_key": f"l{n}", "parent_key": f"c{n}", "kind": "item", "depth": 2,
                      "position": 0, "item_id": doc.item_id, "descendant_count": 1,
                      "path": ["r", f"c{n}"]})
    ctx = _ctx(docs, {"nodes": nodes})
    out = get_enricher("tree.bottom_up_extractive@1").enrich(ctx)
    titles = [out[k]["title"] for k in ("c0", "c1", "c2")]
    assert all(len(t.replace("·", " ").split()) <= 6 for t in titles) and len(set(titles)) == 3
    # identical siblings: no term is distinctive, so side_contrast leaves the title empty
    contrast = get_enricher("tree.side_contrast@1").enrich(ctx)
    assert contrast["c0"] == {"title": "", "distinct_terms": []}


def test_llm_enricher_registered_and_fails_clearly_without_key(monkeypatch, cultivated):
    docs, cv = cultivated
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    enricher = get_enricher("tree.llm_bottom_up@1")
    assert enricher.record["scope"] == "tree"
    ctx = _ctx(docs, cv.cuts["topic"]["topics"])
    with pytest.raises((MissingOptionalDependencyError, LabelingError), match="labeling|OPENAI"):
        enricher.enrich(ctx)


def test_v2_titles_are_clearer_phrases_and_item_titles_for_small_clusters(tmp_path):
    docs = []
    for d in make_topic_docs():
        extra = " Progressive summarization helps." if d.item_id.startswith("t:cooking") else ""
        docs.append(Document(text=d.text + extra, item_id=d.item_id, title=d.title, metadata=d.metadata))
    out = tmp_path / "v2"
    cultivate(docs, topic_perspective_v1(
        version="2", cuts=CUT,
        tree_enrichers=["tree.bottom_up_extractive@2", "tree.top_down_tags@2", "tree.side_contrast@2"],
        display={"title": ["tree.side_contrast@2", "tree.bottom_up_extractive@2"],
                 "summary": ["tree.bottom_up_extractive@2"], "tags": ["tree.top_down_tags@2"]},
    ), name="v2", out_dir=out, encoder=fake_encode, created_at="2026-11-17T03:00:00Z")  # fmt: skip
    cv = read_cultivar(out)
    cut = cv.cuts["topic"]["topics"]
    nodes = {n["node_key"]: n for n in cut["nodes"]}
    bu = cv.node_enrichment["topic"]["tree.bottom_up_extractive@2"]
    assert all(d["title"].strip() for d in cut["display"].values())
    sibs: dict = {}
    for key, d in cut["display"].items():
        sibs.setdefault(nodes[key]["parent_key"], []).append(d["title"].casefold())
    assert all(len(v) == len(set(v)) for v in sibs.values())
    small = [k for k, n in nodes.items() if n["kind"] == "cluster" and n["descendant_count"] <= 3]
    for key in small:
        rep = bu[key]["representative_item_ids"][0]
        assert cut["display"][key]["title"].startswith(cv.item_enrichment["item.extract@1"][rep]["title"][:12])
    assert any("progressive summarization" in o["key_terms"] for o in bu.values())  # phrases are candidates
    assert any(" " in t for o in bu.values() for t in o["key_terms"])


def test_v2_helpers():
    from orchardlogic.cultivate.enrichers import terms2 as P

    assert P.short_item_title("Podcast: Deep Work | Forte Labs") == "Podcast: Deep Work"
    assert P.short_item_title("A Complete Guide to Tagging for Personal Knowledge Bases") == "A Complete Guide to Tagging"
    assert P.short_item_title("Short & sweet") == "Short & sweet"
    assert "progressive summarization" in P.analyze("Progressive summarization works on Monday quickly")
    assert "monday" not in P.analyze("Progressive summarization works on Monday quickly")
    assert "quickly" not in P.analyze("quickly")
    assert P.compose_title(["book launch", "cohort", "course", "extra"]) == "Book Launch · Cohort · Course"
