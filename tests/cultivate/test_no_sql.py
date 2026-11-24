"""orchard owns no SQL: all database access goes through treedb's API."""

from __future__ import annotations

import ast
import re
from pathlib import Path

SQL = re.compile(r"(SELECT\s.+\sFROM|INSERT\s+(OR\s+\w+\s+)?INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM)", re.I | re.S)


def _docstring_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                yield body[0].value


def test_orchard_source_contains_no_sql_strings():
    root = Path(__file__).resolve().parents[1] / "src" / "orchard"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = {id(n) for n in _docstring_nodes(tree)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                if SQL.search(node.value):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders
