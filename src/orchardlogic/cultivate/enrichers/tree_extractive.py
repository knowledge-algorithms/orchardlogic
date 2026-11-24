"""tree.bottom_up_extractive@1: cluster title, summary, representatives and key terms."""

from __future__ import annotations

import numpy as np

from orchardlogic.cultivate.enrichers import text as T
from orchardlogic.cultivate.enrichers.base import TreeContext, make_record, register_enricher
from orchardlogic.cultivate.enrichers.item_extract import central_sentences

SUMMARY_SENTENCES = 2
REPRESENTATIVES = 3
KEY_TERMS = 8
TITLE_TERMS = 3

RECORD = make_record(
    name="tree.bottom_up_extractive",
    version="1",
    scope="tree",
    method="extractive",
    params={
        "summary_sentences": SUMMARY_SENTENCES,
        "representatives": REPRESENTATIVES,
        "key_terms": KEY_TERMS,
        "title_terms": TITLE_TERMS,
    },
    title="Bottom-up extractive summary",
    description=(
        "Per cluster: representatives nearest the centroid, distinctive TF-IDF key terms, a "
        "title from the top terms and a summary from the representatives' item summaries."
    ),
    module="orchardlogic.cultivate.enrichers.tree_extractive",
    module_file=__file__,
)


def _representatives(ctx: TreeContext, ts: T.NodeTerms, items: list[str]) -> list[str]:
    items = sorted(items)
    if ctx.vectors is not None and all(i in ctx.vectors for i in items):
        mat = np.vstack([np.asarray(ctx.vectors[i], dtype=np.float64) for i in items])
    else:
        mat = ts.matrix[[ts.item_row[i] for i in items]].toarray()
        if mat.shape[1] == 0:
            return items[:REPRESENTATIVES]
    centroid = mat.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return items[:REPRESENTATIVES]
    row_norms = np.linalg.norm(mat, axis=1)
    row_norms[row_norms == 0] = 1.0
    sims = (mat / row_norms[:, None]) @ (centroid / norm)
    order = sorted(range(len(items)), key=lambda i: (-round(float(sims[i]), 12), items[i]))
    return [items[i] for i in order[:REPRESENTATIVES]]


def _item_summary(ctx: TreeContext, item_id: str) -> str:
    out = ctx.item_outputs.get("item.extract@1", {}).get(item_id)
    if out is not None:
        return out.get("summary", "")
    doc = ctx.docs.get(item_id)
    return " ".join(central_sentences(doc.text)) if doc is not None else ""


def _item_title(ctx: TreeContext, item_id: str) -> str:
    out = ctx.item_outputs.get("item.extract@1", {}).get(item_id)
    if out is not None and out.get("title"):
        return out["title"]
    doc = ctx.docs.get(item_id)
    return T.clean_title(doc.title) if doc is not None else ""


class BottomUpExtractive:
    record = RECORD

    def enrich(self, ctx: TreeContext) -> dict[str, dict]:
        ts = T.node_terms(ctx)
        keys = T.target_nodes(ctx, ts.nodes)
        outputs: dict[str, dict] = {}
        titles: dict[str, str] = {}
        candidates: dict[str, list[str]] = {}
        for key in keys:
            items = ts.members.get(key, [])
            reps = _representatives(ctx, ts, items) if items else []
            terms = ts.ranked_terms(key, 12)
            if len(items) == 1:
                title = T.limit_title_words(_item_title(ctx, items[0]), T.MAX_TITLE_WORDS - 1)
            else:
                title = T.title_from_terms(terms[:TITLE_TERMS])
            sentences: list[str] = []
            seen: set[str] = set()
            for item_id in reps:
                for sentence in T.split_sentences(_item_summary(ctx, item_id)):
                    norm = sentence.casefold()
                    if norm not in seen:
                        seen.add(norm)
                        sentences.append(sentence)
                        break  # one sentence per representative keeps the mix varied
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


register_enricher("tree.bottom_up_extractive@1", BottomUpExtractive)
