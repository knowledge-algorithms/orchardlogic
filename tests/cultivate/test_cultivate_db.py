from __future__ import annotations

import pytest
from conftest import topic_perspective_v1, fake_encode, make_topic_docs

pytest.importorskip("orchardlogic.treedb")

from orchardlogic.treedb import cutstore, enrichment, ledger, registry, store  # noqa: E402

from orchardlogic.cultivate.cultivate import cultivate_into_db, insert_items, read_cultivar  # noqa: E402
from orchardlogic.cultivate.document import Document  # noqa: E402
from orchardlogic.cultivate.perspectives import topic_perspective  # noqa: E402

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


def _topic(item_id: str) -> str:
    return item_id.split(":")[1].rsplit("-", 1)[0]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "orchard.duckdb"
    docs = make_topic_docs()
    imp = cultivate_into_db(
        docs, PERSPECTIVE, name="vb-topic", db=path, alias="vb/topic",
        encoder=fake_encode, created_at=FIXED, out_dir=tmp_path / "cv",
    )  # fmt: skip
    return path, imp, docs, tmp_path / "cv"


def test_round_trip_matches_cultivar_and_is_idempotent(db):
    path, imp, docs, cv_dir = db
    cv = read_cultivar(cv_dir)
    assert imp.artifact_sha256 == cv.artifact_sha256 and not imp.already_imported
    con = store.connect_with_retry(path)
    try:
        tid = registry.resolve(con, "vb/topic")
        rows = cutstore.read_cut(con, tid, "topics")
        cut = cv.cuts["topic"]["topics"]
        want = {n["node_key"]: n for n in cut["nodes"]}
        assert {r["node_key"] for r in rows} == set(want)
        for r in rows:
            n = want[r["node_key"]]
            assert (r["parent_key"], r["kind"], r["depth"], r["item_id"], r["descendant_count"]) == (
                n["parent_key"], n["kind"], n["depth"], n["item_id"], n["descendant_count"])  # fmt: skip
            assert list(r["path"]) == n["path"]
            if n["kind"] != "item":
                d = cut["display"][n["node_key"]]
                assert (r["title"], r["summary"], list(r["tags"] or [])) == (
                    d["title"], d["summary"], d["tags"])  # fmt: skip
    finally:
        con.close()
    again = cultivate_into_db(
        docs, PERSPECTIVE, name="vb-topic", db=path, alias="vb/topic",
        encoder=fake_encode, created_at=FIXED,
    )  # fmt: skip
    assert again.already_imported and again.artifact_sha256 == imp.artifact_sha256


def _new_docs():
    return [
        Document(
            text="Telescope galaxy orbit comet planet nebula observation tonight under clear "
            "skies with cosmic dust and a lunar eclipse.",
            item_id="n:astro",
            title="Telescope notes",
            metadata={"captured_at": "2026-11-01T09:00:00Z"},
        ),
        Document(
            text="Skillet garlic butter sauce simmer roast onion recipe in the kitchen oven "
            "with fresh broth and pasta.",
            item_id="n:cook",
            title="Sauce recipe",
            metadata={"captured_at": "2026-11-02T09:00:00Z"},
        ),
    ]


def test_insert_places_under_topic_cluster_and_refreshes_only_ancestors(db):
    path, _, docs, _ = db
    con = store.connect_with_retry(path)
    tid = registry.resolve(con, "vb/topic")
    before = {r["node_key"]: r for r in cutstore.read_cut(con, tid, "topics")}
    rev = cutstore.cut_revision(con, tid, "topics")
    con.close()

    res = insert_items(path, "vb/topic", _new_docs(), encoder=fake_encode)
    assert res.new_item_ids == ["n:astro", "n:cook"] and not res.recaptured_item_ids
    assert res.cut_revision > rev and len(res.placements) == 2
    assert res.attach == {"n:astro": None, "n:cook": None} and res.tree_attached

    con = store.connect_with_retry(path)
    try:
        after = {r["node_key"]: r for r in cutstore.read_cut(con, tid, "topics")}
        assert len(after) == len(before) + 2
        for p, expect in zip(res.placements, ("astronomy", "cooking"), strict=True):
            row = after[p["node_key"]]
            assert row["kind"] == "item" and row["parent_key"] == p["parent_key"]
            assert list(row["path"]) == p["path"] and row["depth"] == len(p["path"])
            peers = [
                r for r in before.values()
                if r["kind"] == "item" and p["parent_key"] in (r["parent_key"], *r["path"])
            ]  # fmt: skip
            assert {_topic(r["item_id"]) for r in peers} == {expect}
            for anc in p["path"]:
                assert after[anc]["descendant_count"] == before[anc]["descendant_count"] + (
                    1 + (anc in res.placements[1 - res.placements.index(p)]["path"]))
        touched = {a for p in res.placements for a in p["path"]}
        assert set(res.refreshed_nodes) == touched
        for key, row in before.items():
            if row["kind"] != "item" and key not in touched:
                assert (after[key]["title"], after[key]["summary"]) == (row["title"], row["summary"])
                assert after[key]["descendant_count"] == row["descendant_count"]
        assert all(after[k]["title"] for k in touched)
        assert enrichment.get_item_outputs(con, "item.extract@1", ["n:astro"])["n:astro"]["title"]
        events = ledger.events(con, action="insert")
        assert len(events) == 1 and events[0]["tree_id"] is not None
    finally:
        con.close()


def test_recapture_is_idempotent_and_refreshes_ancestors(db):
    path, _, _, _ = db
    insert_items(path, "vb/topic", _new_docs(), encoder=fake_encode, attach=False)
    con = store.connect_with_retry(path)
    tid = registry.resolve(con, "vb/topic")
    rows = cutstore.read_cut(con, tid, "topics")
    rev, count = cutstore.cut_revision(con, tid, "topics"), len(rows)
    con.close()
    changed = Document(text=_new_docs()[0].text + " Further notes on the telescope mirror.",
                       item_id="n:astro", title="Telescope notes")  # fmt: skip
    res = insert_items(path, "vb/topic", [changed], encoder=fake_encode)
    assert res.recaptured_item_ids == ["n:astro"] and not res.new_item_ids and not res.placements
    assert res.refreshed_nodes and res.attach == {}
    con = store.connect_with_retry(path)
    try:
        assert len(cutstore.read_cut(con, tid, "topics")) == count
        assert cutstore.cut_revision(con, tid, "topics") <= rev + 1  # display refresh only
    finally:
        con.close()


def test_orchard_imports_without_treedb():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['orchardlogic.treedb'] = None; import orchardlogic.cultivate.cultivate; "
        "import orchardlogic.cultivate.cultivate_db"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_write_phase_redoes_placement_when_cut_changed_meanwhile(db, monkeypatch):
    from orchardlogic.cultivate import cultivate_db

    path, _, _, _ = db
    real, state = cultivate_db._compute, {"calls": 0}

    def racing(snap, documents, encoder, placement):
        state["calls"] += 1
        plan = real(snap, documents, encoder, placement)
        if state["calls"] == 1:  # another writer lands between the read and the write phase
            other = Document(text="Compost seedling tomato soil mulch greenhouse harvest herbs.",
                             item_id="n:garden", title="Garden")  # fmt: skip
            monkeypatch.setattr(cultivate_db, "_compute", real)
            insert_items(path, "vb/topic", [other], encoder=fake_encode, attach=False)
            monkeypatch.setattr(cultivate_db, "_compute", racing)
        return plan

    monkeypatch.setattr(cultivate_db, "_compute", racing)
    res = insert_items(path, "vb/topic", _new_docs()[:1], encoder=fake_encode)
    assert state["calls"] == 2  # first attempt, then the redo
    con = store.connect_with_retry(path)
    try:
        ids = {r["item_id"] for r in cutstore.read_cut(con, registry.resolve(con, "vb/topic"), "topics")}
    finally:
        con.close()
    assert {"n:astro", "n:garden"} <= ids and res.write_lock_seconds is not None


def test_write_phase_retries_when_alias_repointed_meanwhile(db, monkeypatch):
    from orchardlogic.cultivate import cultivate_db

    path, imp, docs, _ = db
    real, state = cultivate_db._compute, {"calls": 0}

    def repoint(snap, documents, encoder, placement):
        state["calls"] += 1
        plan = real(snap, documents, encoder, placement)
        if state["calls"] == 1:  # re-point the alias at another tree between read and write
            con = store.connect_with_retry(path)
            try:
                other = cultivate_db.cultivate_into_db(
                    docs[:20], PERSPECTIVE, name="vb-other", db=con, alias="vb/other",
                    encoder=fake_encode, created_at=FIXED)  # fmt: skip
                registry.set_alias(con, "vb/topic", other.trees["topic"])
            finally:
                con.close()
        return plan

    monkeypatch.setattr(cultivate_db, "_compute", repoint)
    res = insert_items(path, "vb/topic", _new_docs()[:1], encoder=fake_encode, attach=False)
    assert state["calls"] == 2 and res.new_item_ids == ["n:astro"]
    con = store.connect_with_retry(path)
    try:
        new_tid = registry.resolve(con, "vb/topic")
        ids = {r["item_id"] for r in cutstore.read_cut(con, new_tid, "topics")}
        old_tid = registry.resolve(con, "vb/other")
        assert new_tid == old_tid and "n:astro" in ids
    finally:
        con.close()
