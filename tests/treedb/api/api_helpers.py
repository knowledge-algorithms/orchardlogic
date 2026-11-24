"""Helpers for tests/api (P6A). Not a conftest."""

from __future__ import annotations

import warnings
from typing import Any

from conftest import load_named
from orchardlogic.treedb import TreeDB
from orchardlogic.treedb.fixtures import item_uuid


def load_into(db: TreeDB, stem: str) -> dict:
    """Import every cultivar of a committed fixture through the facade."""
    fx = load_named(stem)
    for cv in fx["cultivars"]:
        db.import_cultivar(cv["z"], list(cv.get("item_ids", fx["item_ids"])), cv["name"],
                           labels=cv.get("label_sets") or None, cards=cv.get("cards") or None,
                           source={"fixture": fx["name"]})
    return fx


def parent_branch(db: TreeDB, cultivar: str, item: str):
    return db.node(cultivar, item_uuid(item)).parent_branch_id


def orchard_row(db: TreeDB, cultivar: str, item: str):
    return db.node(cultivar, item_uuid(item)).orchard_id


def quietly(fn, *args: Any, **kw: Any):
    """Call ``fn`` and return ``(result, [warning categories])`` without letting warnings escape."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, sorted(type(w.message).__name__ for w in caught)
