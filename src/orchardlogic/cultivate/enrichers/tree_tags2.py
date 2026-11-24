"""tree.top_down_tags@2: tags from phrase-aware term statistics and @2 titles."""

from __future__ import annotations

from orchardlogic.cultivate.enrichers import terms2 as P
from orchardlogic.cultivate.enrichers import text as T
from orchardlogic.cultivate.enrichers.base import TreeContext, make_record, register_enricher

ANCESTOR_TERMS = 2
OWN_TERMS = 3
MAX_TAGS = 8

RECORD = make_record(
    name="tree.top_down_tags",
    version="2",
    scope="tree",
    method="statistical",
    params={"ancestor_terms": ANCESTOR_TERMS, "own_terms": OWN_TERMS, "max_tags": MAX_TAGS},
    title="Top-down tags (phrases)",
    description="As @1 but over phrase-aware terms; context_path uses the @2 node titles.",
    module="orchardlogic.cultivate.enrichers.tree_tags2",
    module_file=__file__,
)

_TITLE_SOURCES = ("tree.side_contrast@2", "tree.bottom_up_extractive@2")


class TopDownTags2:
    record = RECORD

    def enrich(self, ctx: TreeContext) -> dict[str, dict]:
        ts = P.node_terms2(ctx)
        cache: dict[str, list[str]] = {}

        def terms(key: str) -> list[str]:
            if key not in cache:
                cache[key] = ts.ranked_terms(key, 6)
            return cache[key]

        def title(key: str) -> str:
            for handle in _TITLE_SOURCES:
                value = ctx.node_outputs.get(handle, {}).get(key, {}).get("title")
                if value:
                    return value
            return P.compose_title(terms(key), 3)

        outputs: dict[str, dict] = {}
        for key in T.target_nodes(ctx, ts.nodes):
            node = ts.nodes[key]
            ancestors = [a for a in node["path"] if ts.nodes[a]["kind"] != "root"]
            chosen: list[str] = []
            for anc in ancestors:
                chosen += P.pick_phrases(terms(anc), ANCESTOR_TERMS, exclude=chosen)
            chosen += P.pick_phrases(terms(key), OWN_TERMS, exclude=chosen)
            outputs[key] = {"tags": chosen[:MAX_TAGS], "context_path": [title(a) for a in ancestors]}
        return outputs


register_enricher("tree.top_down_tags@2", TopDownTags2)
