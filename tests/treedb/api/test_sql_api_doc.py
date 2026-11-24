"""P6A: the Python examples in docs/SQL_API.md run, and the doc names every macro the SQL file defines."""

from __future__ import annotations

import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[3] / "docs" / "treedb_sql_api.md"
MACROS = Path(__file__).resolve().parents[3] / "src" / "orchardlogic" / "treedb" / "sql" / "api_macros.sql"


def test_sql_api_examples_run():
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)
    assert blocks
    for code in blocks:
        exec(compile(code, "docs/SQL_API.md", "exec"), {"__name__": "sql_api_doc"})


def test_every_macro_is_documented():
    defined = re.findall(r"CREATE OR REPLACE MACRO (treedb_\w+)\(", MACROS.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")
    assert defined and all(f"`{m}(" in doc for m in defined)
    assert "OD-10" in doc and "D-020" in doc
