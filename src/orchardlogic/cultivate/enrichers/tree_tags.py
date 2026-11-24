"""tree.top_down_tags@1: tags from the ancestors' and the node's own top terms."""

from __future__ import annotations

from orchardlogic.cultivate.enrichers import text as T
from orchardlogic.cultivate.enrichers.base import TreeContext, make_record, register_enricher

ANCESTOR_TERMS = 2
OWN_TERMS = 3
MAX_TAGS = 8

RECORD = make_record(
    name="tree.top_down_tags",
    version="1",
    scope="tree",
    method="statistical",
    params={"ancestor_terms": ANCESTOR_TERMS, "own_terms": OWN_TERMS, "max_tags": MAX_TAGS},
    title="Top-down tags",
    description=(
        "Tags are the non-root ancestors' top terms plus the node's own, each kept once; "
        "context_path lists the ancestors' titles."
    ),
    module="orchardlogic.cultivate.enrichers.tree_tags",
    module_file=__file__,
)

_TITLE_SOURCES = ("tree.side_contrast@1", "tree.bottom_up_extractive@1")


class TopDownTags:
    record = RECORD

    def enrich(self, ctx: TreeContext) -> dict[str, dict]:
        ts = T.node_terms(ctx)
        terms_cache: dict[str, list[str]] = {}

        def terms(key: str) -> list[str]:
            if key not in terms_cache:
                terms_cache[key] = ts.ranked_terms(key, 6)
            return terms_cache[key]

        def title(key: str) -> str:
            for handle in _TITLE_SOURCES:
                value = ctx.node_outputs.get(handle, {}).get(key, {}).get("title")
                if value:
                    return value
            return T.title_from_terms(terms(key)[:3])

        outputs: dict[str, dict] = {}
        for key in T.target_nodes(ctx, ts.nodes):
            node = ts.nodes[key]
            ancestors = [a for a in node["path"] if ts.nodes[a]["kind"] != "root"]
            chosen: list[str] = []
            for anc in ancestors:
                chosen += T.pick_distinct(terms(anc), ANCESTOR_TERMS, exclude=chosen)
            chosen += T.pick_distinct(terms(key), OWN_TERMS, exclude=chosen)
            outputs[key] = {
                "tags": chosen[:MAX_TAGS],
                "context_path": [title(a) for a in ancestors],
            }
        return outputs


register_enricher("tree.top_down_tags@1", TopDownTags)
