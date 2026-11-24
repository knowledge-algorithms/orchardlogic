"""tree.bottom_up_extractive@2: clearer cluster titles (phrases; item titles for small clusters)."""

from __future__ import annotations

from orchardlogic.cultivate.enrichers import terms2 as P
from orchardlogic.cultivate.enrichers import text as T
from orchardlogic.cultivate.enrichers.base import TreeContext, make_record, register_enricher
from orchardlogic.cultivate.enrichers.tree_extractive import (
    KEY_TERMS,
    REPRESENTATIVES,
    SUMMARY_SENTENCES,
    _item_summary,
    _item_title,
    _representatives,
)

SMALL_CLUSTER = 3
TITLE_TERMS = 3
MAX_ITEM_TITLE_WORDS = 7

RECORD = make_record(
    name="tree.bottom_up_extractive",
    version="2",
    scope="tree",
    method="extractive",
    params={
        "summary_sentences": SUMMARY_SENTENCES,
        "representatives": REPRESENTATIVES,
        "key_terms": KEY_TERMS,
        "title_terms": TITLE_TERMS,
        "small_cluster_items": SMALL_CLUSTER,
        "item_title_words": MAX_ITEM_TITLE_WORDS,
        "ngram": "1-2 (adjacent content words)",
        "max_df": P.MAX_DF,
    },
    title="Bottom-up extractive summary (phrases)",
    description=(
        "Like @1, but terms are phrase-aware (frequent bigrams preferred; generic words, "
        "weekdays, months and corpus-common words dropped) and clusters of at most 3 items are "
        "titled with the shortened title of their most representative item."
    ),
    module="orchardlogic.cultivate.enrichers.tree_extractive2",
    module_file=__file__,
)


class BottomUpExtractive2:
    record = RECORD

    def enrich(self, ctx: TreeContext) -> dict[str, dict]:
        ts = P.node_terms2(ctx)
        outputs: dict[str, dict] = {}
        titles: dict[str, str] = {}
        candidates: dict[str, list[str]] = {}
        for key in T.target_nodes(ctx, ts.nodes):
            items = ts.members.get(key, [])
            reps = _representatives(ctx, ts, items) if items else []
            terms = ts.ranked_terms(key, 12)
            node = ts.nodes[key]
            if reps and node["kind"] != "root" and len(items) <= SMALL_CLUSTER:
                title = P.short_item_title(_item_title(ctx, reps[0]), MAX_ITEM_TITLE_WORDS)
            else:
                title = P.compose_title(terms, TITLE_TERMS)
            sentences: list[str] = []
            seen: set[str] = set()
            for item_id in reps:
                for sentence in T.split_sentences(_item_summary(ctx, item_id)):
                    norm = sentence.casefold()
                    if norm not in seen:
                        seen.add(norm)
                        sentences.append(sentence)
                        break
                if len(sentences) == SUMMARY_SENTENCES:
                    break
            outputs[key] = {
                "title": title,
                "summary": " ".join(sentences),
                "representative_item_ids": reps,
                "key_terms": terms[:KEY_TERMS],
            }
            titles[key] = title
            candidates[key] = terms
        groups = [[k for k in kids if k in titles] for kids in ts.children.values()]
        for key, title in T.dedupe_sibling_titles(titles, groups, candidates).items():
            outputs[key]["title"] = title
        return outputs


register_enricher("tree.bottom_up_extractive@2", BottomUpExtractive2)
