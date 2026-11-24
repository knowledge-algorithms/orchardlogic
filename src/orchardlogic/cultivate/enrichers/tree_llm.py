"""tree.llm_bottom_up@1: thin wrapper over ``orchard.labeling.bottom_up`` (optional).

Not run in the overnight pipeline (D5) and never invoked in tests. It needs the ``labeling``
extra (``openai``) and an API key, or an injected ``client``; otherwise it raises a clear
error before any network call.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from orchardlogic.cultivate.enrichers import text as T
from orchardlogic.cultivate.enrichers.base import TreeContext, make_record, register_enricher
from orchardlogic.cultivate.exceptions import LabelingError, MissingOptionalDependencyError

RECORD = make_record(
    name="tree.llm_bottom_up",
    version="1",
    scope="tree",
    method="llm",
    params={"label_set": "llm_bottom_up", "model": "gpt-6-luna"},
    title="LLM bottom-up labels",
    description=(
        "Wraps orchard.labeling.bottom_up.label_bottom_up; title and summary come from the "
        "node cards. Requires the labeling extra and an API key. Not run overnight."
    ),
    module="orchardlogic.cultivate.enrichers.tree_llm",
    module_file=__file__,
)


class LLMBottomUp:
    record = RECORD

    def __init__(self, client: Any = None, model: str | None = None) -> None:
        self.client = client
        self.model = model

    def _require_runtime(self) -> None:
        if self.client is not None:
            return
        if importlib.util.find_spec("openai") is None:
            raise MissingOptionalDependencyError(
                "labeling",
                "tree.llm_bottom_up@1 needs orchard[labeling] (openai) or an injected client.",
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise LabelingError(
                "tree.llm_bottom_up@1 needs OPENAI_API_KEY (or an injected client); "
                "it is not run by default."
            )

    def enrich(self, ctx: TreeContext) -> dict[str, dict]:
        self._require_runtime()
        if ctx.tree is None:
            raise LabelingError("tree.llm_bottom_up@1 needs TreeContext.tree (the Orchard Tree)")
        from orchardlogic.cultivate.labeling.bottom_up import label_bottom_up

        kwargs: dict[str, Any] = {"client": self.client}
        if self.model:
            kwargs["model"] = self.model
        cards = label_bottom_up(ctx.tree, **kwargs)
        nodes = {n["node_key"]: n for n in ctx.nodes}
        return {
            key: {"title": cards[key].title, "summary": cards[key].summary}
            for key in T.target_nodes(ctx, nodes)
            if key in cards
        }


register_enricher("tree.llm_bottom_up@1", LLMBottomUp)
