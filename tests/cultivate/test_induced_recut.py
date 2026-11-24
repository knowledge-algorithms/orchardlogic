"""Induced-view recutting: coverage, ancestry, exhaustive small-case checks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage

from orchardlogic.cultivate import Tree, build_dynamic_cut, cut_partition_item_ids
from orchardlogic.cultivate.induced import (
    CUT_INDIVIDUALS,
    CUT_OPTIMAL,
    CUT_UNDEFINED,
    CutSettings,
    induce_tree,
    recut,
)


def make_tree(count: int = 40, seed: int = 3, dims: int = 5) -> tuple[Tree, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(5, dims)) * 5
    features = np.vstack([centers[i % 5] + rng.normal(size=dims) for i in range(count)])
    item_ids = [f"item_{i:03d}" for i in range(count)]
    tree = Tree.from_linkage(
        scipy_linkage(features, method="average"), item_ids=item_ids, tree_id="t"
    )
    return tree, features


def expected_induced_memberships(tree: Tree, survivors: set[str]) -> set[frozenset[str]]:
    """Independent definition: branch points of the original tree among survivors."""
    expected = {frozenset([item]) for item in survivors}
    for node in tree.nodes.values():
        if node["kind"] != "internal":
            continue
        sides = [
            survivors.intersection(tree.nodes[child]["descendant_item_ids"])
            for child in node["children"]
        ]
        if all(sides):
            expected.add(frozenset(sides[0] | sides[1]))
    return expected


def calinski_harabasz(features: np.ndarray, clusters: list[list[int]]) -> float:
    overall = features.mean(axis=0)
    between = within = 0.0
    for rows in clusters:
        block = features[rows]
        center = block.mean(axis=0)
        between += len(rows) * float(((center - overall) ** 2).sum())
        within += float(((block - center) ** 2).sum())
    count, k = features.shape[0], len(clusters)
    return (between / (k - 1)) / (within / (count - k))


def brute_force_top_cut(induced: Tree, rows: np.ndarray, k_lo: int, k_hi: int) -> set[frozenset[str]]:
    """Score every admissible height-ordered cut by hand; smallest k wins ties."""
    index = {item_id: i for i, item_id in enumerate(induced.item_ids)}
    internals = sorted(
        (node for node in induced.nodes.values() if node["kind"] == "internal"),
        key=lambda node: node["linkage_row"],
        reverse=True,
    )
    best_score, best = float("-inf"), None
    for k in range(k_lo, min(k_hi, induced.leaf_count - 1) + 1):
        clusters = {induced.root_node_id}
        for node in internals[: k - 1]:
            clusters.remove(node["node_id"])
            clusters.update(node["children"])
        memberships = [induced.nodes[node_id]["descendant_item_ids"] for node_id in clusters]
        score = calinski_harabasz(rows, [[index[i] for i in m] for m in memberships])
        if score > best_score:
            best_score, best = score, {frozenset(m) for m in memberships}
    assert best is not None
    return best


def test_full_cohort_recut_matches_direct_cut() -> None:
    tree, features = make_tree()
    settings = CutSettings()
    view = recut(tree, tree.item_ids, features, settings)
    direct = build_dynamic_cut(
        tree,
        top_criterion="optimal",
        min_width=settings.min_width,
        max_width=settings.max_width,
        target_width=None,
        max_depth=settings.max_depth,
        cut_optimizer="calinski_harabasz_score",
        feature_matrix=features,
    )
    assert view.cut_status == CUT_OPTIMAL
    assert view.cut["root"] == direct["root"]
    assert set(view.tree.nodes) == set(tree.nodes)
    assert all(not p.restricted for p in view.provenance.values())


@pytest.mark.parametrize("seed", range(8))
def test_arbitrary_subset_preserves_coverage_partition_and_ancestry(seed: int) -> None:
    tree, features = make_tree(seed=seed)
    rng = np.random.default_rng(100 + seed)
    survivors = set(rng.choice(tree.item_ids, size=int(rng.integers(9, 35)), replace=False))
    before = (tree.linkage.copy(), dict(tree.nodes), features.copy())

    view = recut(tree, survivors, features)
    induced = view.tree

    # Exact induced hierarchy, in original leaf order.
    assert list(induced.item_ids) == [i for i in tree.item_ids if i in survivors]
    memberships = {
        frozenset(node["descendant_item_ids"]) for node in induced.nodes.values()
    }
    assert memberships == expected_induced_memberships(tree, survivors)

    # Provenance: the source is the lowest original node holding the survivors,
    # and merge heights/order are the originals.
    for node_id, node in induced.nodes.items():
        provenance = view.provenance[node_id]
        source = tree.nodes[provenance.source_node_id]
        assert survivors.intersection(source["descendant_item_ids"]) == set(
            node["descendant_item_ids"]
        )
        assert provenance.surviving_count == node["descendant_count"]
        assert provenance.source_count == source["descendant_count"]
        assert node["linkage_distance"] == source["linkage_distance"]
        for ancestor_id in provenance.compressed_source_node_ids:
            ancestor = tree.nodes[ancestor_id]
            assert survivors.intersection(ancestor["descendant_item_ids"]) == set(
                node["descendant_item_ids"]
            )
            assert ancestor["descendant_count"] > source["descendant_count"]
    assert np.all(np.diff(induced.linkage[:, 2]) >= 0)

    # The recut is a disjoint cover of the cohort.
    assert view.cut_status == CUT_OPTIMAL
    parts = cut_partition_item_ids(view.cut, induced)
    flat = [item for part in parts for item in part]
    assert sorted(flat) == sorted(survivors)
    assert len(flat) == len(set(flat))

    # Source artifacts are untouched.
    assert np.array_equal(before[0], tree.linkage)
    assert before[1] == tree.nodes
    assert np.array_equal(before[2], features)


@pytest.mark.parametrize("seed", range(10))
def test_top_cut_matches_exhaustive_admissible_search(seed: int) -> None:
    tree, features = make_tree(count=18, seed=20 + seed)
    rng = np.random.default_rng(seed)
    survivors = list(rng.choice(tree.item_ids, size=12, replace=False))
    settings = CutSettings(min_width=2, max_width=6, max_depth=1, individual_threshold=0)
    view = recut(tree, survivors, features, settings)
    induced = view.tree
    rows = features[[tree.item_ids.index(i) for i in induced.item_ids]]

    expected = brute_force_top_cut(induced, rows, 2, 6)
    actual = {
        frozenset(induced.nodes[node_id]["descendant_item_ids"])
        for node_id in view.cut["top_partition_node_ids"]
    }
    assert actual == expected


def test_recut_is_not_a_filter_of_the_original_cut() -> None:
    """Removing a whole top-level group must yield a new best partition."""
    tree, features = make_tree(count=60, seed=11)
    full = recut(tree, tree.item_ids, features)
    top = [
        set(tree.nodes[node_id]["descendant_item_ids"])
        for node_id in full.cut["top_partition_node_ids"]
    ]
    largest = max(top, key=len)
    view = recut(tree, largest, features)
    assert view.cut_status == CUT_OPTIMAL
    assert view.cut["top_partition_count"] >= 3
    assert sorted(i for p in cut_partition_item_ids(view.cut, view.tree) for i in p) == sorted(largest)


def test_sparse_and_degenerate_cohorts_are_explicit() -> None:
    tree, features = make_tree()
    small = recut(tree, tree.item_ids[:5], features)
    assert small.cut is None and small.cut_status == CUT_INDIVIDUALS

    single = recut(tree, [tree.item_ids[7]], features)
    assert single.cut_status == CUT_INDIVIDUALS and single.tree.leaf_count == 1

    # Identical feature rows: no admissible scored cut exists, and none is invented.
    flat_features = np.zeros_like(features)
    undefined = recut(tree, tree.item_ids[:12], flat_features, CutSettings())
    assert undefined.cut is None and undefined.cut_status == CUT_UNDEFINED
    assert undefined.cut_reason

    with pytest.raises(ValueError):
        induce_tree(tree, [])
    with pytest.raises(ValueError):
        induce_tree(tree, ["not-an-item"])
