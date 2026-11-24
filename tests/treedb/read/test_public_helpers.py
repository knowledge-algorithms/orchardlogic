"""PX-G4 (P4B review note 6): public read helpers, and no module outside read.py calls ``read._*``."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest

from orchardlogic.treedb import read, store

SRC = Path(read.__file__).resolve().parent


def test_no_module_outside_read_calls_a_private_read_name():
    """Scan src/treedb for ``read._name`` (other modules' private names are out of scope)."""
    pattern = re.compile(r"\bread\._\w+")
    hits = []
    for path in SRC.rglob("*.py"):
        if path == Path(read.__file__).resolve():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(SRC)}:{n}: {line.strip()}")
    assert hits == []


def test_public_helpers_behave_like_the_private_ones():
    con = store.connect()
    read.ensure_installed(con)
    assert con.execute("SELECT treedb.read_sql_version()").fetchone()[0] == read.read_sql_version()
    assert read.in_transaction(con) is False
    with read.snapshot(con):
        assert read.in_transaction(con) is True
    assert read.in_transaction(con) is False
    with pytest.raises(read.ReadError):
        read.cultivar_version(con, "nope")


def test_cultivar_version_by_name_and_id():
    con = store.connect()
    tid = uuid.uuid4()
    con.execute("INSERT INTO treedb.tree_registry (id, name) VALUES (?, 'x')", [tid])
    got = read.cultivar_version(con, "x")
    assert got[0] == tid and got[1] == "x" and isinstance(got[2], int)
    assert read.cultivar_version(con, tid) == got
