from __future__ import annotations

import pytest

from orchardlogic.cultivate.backends.minilm import MINILM_REVISION
from orchardlogic.cultivate.perspectives import (
    InvalidPerspectiveError,
    perspective_id,
    topic_perspective,
    validate_perspective,
)


def test_topic_perspective_defaults_and_space_name():
    p = topic_perspective()
    assert p["schema_version"] == "orchard_perspective_v1"
    assert p["embedding"]["space"] == f"minilm-l6-v2@{MINILM_REVISION[:8]}/mean/tt1"
    assert p["cuts"][0]["name"] == "topics"
    assert p["tree_enrichers"][0] == "tree.bottom_up_extractive@2"


def test_overrides_deep_merge_and_leave_default_untouched():
    p = topic_perspective(embedding={"max_chars": 500}, cuts=[{"name": "c", "top_criterion": "optimal"}])
    assert p["embedding"]["max_chars"] == 500
    assert p["embedding"]["model_id"].startswith("sentence-transformers/")
    assert [c["name"] for c in p["cuts"]] == ["c"]
    assert topic_perspective()["embedding"]["max_chars"] == 2000


def test_perspective_id_is_stable_sha256_and_content_sensitive():
    a, b = topic_perspective(), topic_perspective()
    assert perspective_id(a) == perspective_id(b)
    assert len(perspective_id(a)) == 64
    assert perspective_id(topic_perspective(version="3")) != perspective_id(a)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version="nope"),
        lambda p: p.pop("cuts"),
        lambda p: p["embedding"].pop("space"),
        lambda p: p["item_enrichers"].append("bad handle"),
        lambda p: p["display"].update(title=["tree.unlisted@1"]),
        lambda p: p["features"].append({"name": "x", "weight": 1.0}),  # ward + multi-feature
        lambda p: p["cuts"].append(dict(p["cuts"][0])),  # duplicate cut name
    ],
)
def test_validation_rejects_bad_records(mutate):
    p = topic_perspective()
    mutate(p)
    with pytest.raises(InvalidPerspectiveError):
        validate_perspective(p)
