"""P6A: the README quickstart runs. The first ```python block under "## Quickstart" is executed."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[3] / "docs" / "treedb.md"


def _quickstart() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    m = re.search(r"```python\n(.*?)```", section, re.S)
    assert m, "no python block under ## Quickstart"
    return m.group(1)


def test_readme_quickstart_runs():
    code = _quickstart()
    exec(compile(code, "README.md quickstart", "exec"), {"__name__": "readme_quickstart"})
