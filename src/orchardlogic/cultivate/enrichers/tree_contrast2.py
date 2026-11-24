"""tree.side_contrast@2: sibling-relative phrases; no term title for clusters of <= 3 items."""

from __future__ import annotations

import numpy as np

from orchardlogic.cultivate.enrichers import terms2 as P
from orchardlogic.cultivate.enrichers import text as T
from orchardlogic.cultivate.enrichers.base import TreeContext, make_record, register_enricher

LIFT_THRESHOLD = 1.5
EPSILON = 0.01
DISTINCT_TERMS = 5
MIN_COVERAGE = 0.3
SMALL_CLUSTER = 3

RECORD = make_record(
    name="tree.side_contrast",
    version="2",
    scope="tree",
    method="statistical",
    params={
        "lift_threshold": LIFT_THRESHOLD,
        "epsilon": EPSILON,
        "distinct_terms": DISTINCT_TERMS,
        "min_coverage": MIN_COVERAGE,
        "small_cluster_items": SMALL_CLUSTER,
        "ngram": "1-2 (adjacent content words)",
    },
    title="Side contrast (phrases)",
    description=(
        "Sibling-relative TF-IDF lift over phrase-aware terms. The title is empty below the lift "
        "threshold and for clusters of at most 3 items (display falls back to the item-based "
        "bottom-up title)."
    ),
    module="orchardlogic.cultivate.enrichers.tree_contrast2",
    module_file=__file__,
)


class SideContrast2:
    record = RECORD

    def enrich(self, ctx: TreeContext) -> dict[str, dict]:
        ts = P.node_terms2(ctx)
        mult = ts.multipliers() if ts.vocab.size else None
        bigram = np.fromiter((" " in t for t in ts.vocab), dtype=bool, count=len(ts.vocab))
        outputs: dict[str, dict] = {}
        titles: dict[str, str] = {}
        candidates: dict[str, list[str]] = {}
        for key in T.target_nodes(ctx, ts.nodes):
            parent = ts.nodes[key].get("parent_key")
            terms: list[str] = []
            n = len(ts.members.get(key, ()))
            if parent is not None and ts.vocab.size:
                mine = set(ts.members.get(key, ()))
                rest = [i for i in ts.members.get(parent, ()) if i not in mine]
                if rest and mine:
                    node_rows = sorted(ts.item_row[i] for i in mine)
                    rest_rows = sorted(ts.item_row[i] for i in rest)
                    p_node = np.asarray(ts.matrix[node_rows].mean(axis=0)).ravel()
                    p_rest = np.asarray(ts.matrix[rest_rows].mean(axis=0)).ravel()
                    cov = np.asarray(ts.binary[node_rows].mean(axis=0)).ravel()
                    lift = (p_node + EPSILON) / (p_rest + EPSILON)
                    ok = (cov >= MIN_COVERAGE) & (lift >= LIFT_THRESHOLD)
                    if n >= 2:
                        ok &= cov * n >= 2 - 1e-9
                    boost = np.where(bigram & (cov * n >= 3 - 1e-9), 1.5, np.where(bigram, 0.7, 1.0))
                    score = np.where(
                        ok, p_node * np.log(np.maximum(lift, 1e-12)) * mult * boost, 0.0
                    )
                    order = sorted(np.flatnonzero(score > 0), key=lambda i: (-score[i], ts.vocab[i]))
                    terms = P.pick_phrases((str(ts.vocab[i]) for i in order), 10)
            title = ""
            if n > SMALL_CLUSTER and (len(terms) >= 2 or (terms and " " in terms[0])):
                title = P.compose_title(terms, 3)
            outputs[key] = {"title": title, "distinct_terms": terms[:DISTINCT_TERMS]}
            titles[key] = title
            candidates[key] = terms
        groups = [[k for k in kids if k in titles] for kids in ts.children.values()]
        for key, title in T.dedupe_sibling_titles(titles, groups, candidates).items():
            outputs[key]["title"] = title
        return outputs


register_enricher("tree.side_contrast@2", SideContrast2)
