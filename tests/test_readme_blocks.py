"""Every ```python block in README.md runs (in a temporary working directory)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


@pytest.mark.parametrize("code", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_readme_block_runs(code: str, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "README.md", "exec"), {"__name__": "readme"})
