"""tree.side_contrast@1: terms high in a node and low among the rest of its parent."""

from __future__ import annotations

import numpy as np

from orchardlogic.cultivate.enrichers import text as T
from orchardlogic.cultivate.enrichers.base import TreeContext, make_record, register_enricher

LIFT_THRESHOLD = 1.5
EPSILON = 0.01
DISTINCT_TERMS = 5
MIN_COVERAGE = 0.3

RECORD = make_record(
    name="tree.side_contrast",
    version="1",
    scope="tree",
    method="statistical",
    params={
        "lift_threshold": LIFT_THRESHOLD,
        "epsilon": EPSILON,
        "distinct_terms": DISTINCT_TERMS,
        "min_coverage": MIN_COVERAGE,
    },
    title="Side contrast",
    description=(
        "Sibling-relative TF-IDF lift: terms strong in a node and weak in the other items "
        "under its parent. The title is empty when no term clears the lift threshold."
    ),
    module="orchardlogic.cultivate.enrichers.tree_contrast",
    module_file=__file__,
)


class SideContrast:
    record = RECORD

    def enrich(self, ctx: TreeContext) -> dict[str, dict]:
        ts = T.node_terms(ctx)
        outputs: dict[str, dict] = {}
        titles: dict[str, str] = {}
        candidates: dict[str, list[str]] = {}
        for key in T.target_nodes(ctx, ts.nodes):
            parent = ts.nodes[key].get("parent_key")
            terms: list[str] = []
            if parent is not None and ts.vocab.size:
                mine = set(ts.members.get(key, ()))
                rest = [i for i in ts.members.get(parent, ()) if i not in mine]
                n = len(mine)
                if rest and n:
                    node_rows = sorted(ts.item_row[i] for i in mine)
                    rest_rows = sorted(ts.item_row[i] for i in rest)
                    p_node = np.asarray(ts.matrix[node_rows].mean(axis=0)).ravel()
                    p_rest = np.asarray(ts.matrix[rest_rows].mean(axis=0)).ravel()
                    cov = np.asarray(ts.binary[node_rows].mean(axis=0)).ravel()
                    lift = (p_node + EPSILON) / (p_rest + EPSILON)
                    ok = (cov >= MIN_COVERAGE) & (lift >= LIFT_THRESHOLD)
                    if n >= 2:
                        ok &= cov * n >= 2 - 1e-9
                    score = np.where(ok, p_node * np.log(np.maximum(lift, 1e-12)), 0.0)
                    order = sorted(
                        np.flatnonzero(score > 0), key=lambda i: (-score[i], ts.vocab[i])
                    )
                    terms = T.pick_distinct((str(ts.vocab[i]) for i in order), 10)
            title = T.title_from_terms(terms[:3]) if len(terms) >= 2 else ""
            outputs[key] = {"title": title, "distinct_terms": terms[:DISTINCT_TERMS]}
            titles[key] = title
            candidates[key] = terms
        groups = [[k for k in kids if k in titles] for kids in ts.children.values()]
        for key, title in T.dedupe_sibling_titles(titles, groups, candidates).items():
            outputs[key]["title"] = title
        return outputs


register_enricher("tree.side_contrast@1", SideContrast)
