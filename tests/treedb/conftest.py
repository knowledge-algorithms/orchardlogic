"""Shared fixture-loading helpers for the TreeDB tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent
FIXTURE_DIR = REPO / "fixtures"
# Only P1 owns these; other phases (e.g. query_cases/) use other formats.
FIXTURE_SUBDIRS = ("synthetic",)

from orchardlogic.treedb.fixtures import load_fixture  # noqa: E402


def fixture_paths() -> list[Path]:
    """Every committed ``treedb_fixture_v1`` file, in a stable order."""
    return sorted(p for d in FIXTURE_SUBDIRS for p in (FIXTURE_DIR / d).glob("*.json"))


def load_named(stem: str) -> dict:
    """Load a committed fixture by file stem, e.g. ``F5_twocult8``."""
    matches = [p for p in fixture_paths() if p.stem == stem]
    if len(matches) != 1:
        raise FileNotFoundError(stem)
    return load_fixture(matches[0])


