"""Induced hierarchy views and immutable recutting for arbitrary leaf cohorts.

Given an immutable :class:`~orchard.tree.Tree` and any (possibly noncontiguous)
subset of its leaves, :func:`induce_tree` builds the induced binary hierarchy:

* surviving leaves keep their place in the original hierarchy;
* branches with no survivors are dropped;
* unary paths (original nodes with survivors on one side only) are compressed;
* original ancestry, merge ordering, and merge heights are preserved, and every
  induced node records the original node it came from.

:func:`recut` then recomputes the nested Calinski-Harabasz cut on the induced
hierarchy using only the surviving rows of the cultivar's original feature
representation. Nothing is re-embedded, no LLM is called, and the source tree and
feature matrix are never mutated.

Search guarantee (inherited from :func:`orchard.cuts.build_dynamic_cut`): at each
level the search is exhaustive over cluster counts ``k`` in the admissible width
range, restricted to height-ordered (SciPy ``maxclust``) cuts of the induced
hierarchy. It is not a search over every antichain of the tree. Ties go to the
smallest ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from orchardlogic.cultivate.cuts import build_dynamic_cut, validate_dynamic_cut
from orchardlogic.cultivate.tree import Tree

CUT_GUARANTEE = (
    "exhaustive over k within the width range at each level, restricted to "
    "height-ordered cuts of the induced hierarchy; smallest k wins ties"
)

# Cut status values.
CUT_OPTIMAL = "optimal"  # a best admissible CH cut was found
CUT_INDIVIDUALS = "individuals"  # cohort at or below the individual-display threshold
CUT_UNDEFINED = "undefined"  # no admissible cut / CH undefined; individuals shown


@dataclass(frozen=True)
class CutSettings:
    """Presentation constraints for the nested optimal cut."""

    min_width: int = 3
    max_width: int = 10
    target_width: int | None = None
    max_depth: int = 5
    cut_optimizer: str = "calinski_harabasz_score"
    cut_polarity: int = 1
    individual_threshold: int = 8

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_width": self.min_width,
            "max_width": self.max_width,
            "target_width": self.target_width,
            "max_depth": self.max_depth,
            "cut_optimizer": self.cut_optimizer,
            "cut_polarity": self.cut_polarity,
            "individual_threshold": self.individual_threshold,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> CutSettings:
        value = dict(value or {})
        known = {name: value[name] for name in cls.__dataclass_fields__ if name in value}
        return cls(**known)


@dataclass(frozen=True)
class NodeProvenance:
    """Where an induced node came from in the original hierarchy.

    ``source_node_id`` is the lowest original node containing every surviving
    member of the induced node (its tightest original scope).
    ``compressed_source_node_ids`` are original ancestors that collapsed onto this
    induced node because their other side had no survivors, nearest first.
    """

    source_node_id: str
    source_count: int
    surviving_count: int
    compressed_source_node_ids: tuple[str, ...] = ()

    @property
    def restricted(self) -> bool:
        return self.surviving_count < self.source_count

    @property
    def retained_fraction(self) -> float:
        return self.surviving_count / self.source_count


@dataclass(frozen=True)
class InducedView:
    """An induced hierarchy over a cohort plus its recomputed cut."""

    source_tree_id: str
    tree: Tree
    provenance: Mapping[str, NodeProvenance]
    cut: Mapping[str, Any] | None
    cut_status: str
    cut_reason: str | None = None
    settings: CutSettings = field(default_factory=CutSettings)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return self.tree.item_ids


def _source_ids_by_scipy_index(tree: Tree) -> dict[int, str]:
    by_item = {
        node["item_id"]: node_id
        for node_id, node in tree.nodes.items()
        if node["kind"] == "leaf"
    }
    mapping = {index: by_item[item_id] for index, item_id in enumerate(tree.item_ids)}
    for node_id, node in tree.nodes.items():
        if node["kind"] == "internal":
            mapping[int(node["scipy_node_id"])] = node_id
    return mapping


def induce_tree(
    tree: Tree,
    surviving_item_ids: Iterable[str],
    *,
    tree_id: str | None = None,
) -> tuple[Tree, dict[str, NodeProvenance]]:
    """Return the induced hierarchy over ``surviving_item_ids`` and its provenance."""
    survivors = set(surviving_item_ids)
    if not survivors:
        raise ValueError("an induced view needs at least one surviving item")
    unknown = survivors.difference(tree.item_ids)
    if unknown:
        raise ValueError(f"items are not in tree {tree.tree_id!r}: {sorted(unknown)[:5]}")

    full_count = len(tree.item_ids)
    local_item_ids = [item_id for item_id in tree.item_ids if item_id in survivors]
    local_count = len(local_item_ids)
    local_index = {item_id: index for index, item_id in enumerate(local_item_ids)}
    source_ids = _source_ids_by_scipy_index(tree)

    # rep[original scipy index] -> local scipy index of the induced node that
    # represents its surviving members (None when no members survive).
    rep: dict[int, int | None] = {}
    size: dict[int, int] = {}
    source_of: dict[int, int] = {}
    compressed: dict[int, list[int]] = {}
    for index, item_id in enumerate(tree.item_ids):
        if item_id in local_index:
            local = local_index[item_id]
            rep[index] = local
            size[local] = 1
            source_of[local] = index
            compressed[local] = []
        else:
            rep[index] = None

    rows: list[tuple[float, float, float, float]] = []
    z_matrix = np.asarray(tree.linkage, dtype=np.float64)
    for row_index in range(z_matrix.shape[0]):
        original = full_count + row_index
        left = rep[int(z_matrix[row_index, 0])]
        right = rep[int(z_matrix[row_index, 1])]
        if left is not None and right is not None:
            local = local_count + len(rows)
            height = float(z_matrix[row_index, 2])
            size[local] = size[left] + size[right]
            rows.append((float(left), float(right), height, float(size[local])))
            rep[original] = local
            source_of[local] = original
            compressed[local] = []
        else:
            survivor = left if left is not None else right
            rep[original] = survivor
            if survivor is not None:
                compressed[survivor].append(original)

    if len(rows) != local_count - 1:
        raise ValueError("induced hierarchy is not a binary tree over the survivors")
    induced_z = np.asarray(rows, dtype=np.float64).reshape(len(rows), 4)
    # Kept rows follow original merge order, so heights stay monotone whenever the
    # source linkage is monotone; an inversion in the source is surfaced, not hidden.
    if len(rows) > 1 and np.any(np.diff(induced_z[:, 2]) < 0):
        raise ValueError(
            f"tree {tree.tree_id!r} has non-monotone merge heights; "
            "height-ordered recutting is undefined for it"
        )

    induced = Tree.from_linkage(
        induced_z,
        item_ids=local_item_ids,
        tree_id=tree_id or tree.tree_id,
        method=tree.method,
        documents=[
            tree.documents_by_id[item_id]
            for item_id in local_item_ids
            if item_id in tree.documents_by_id
        ],
    )

    local_node_ids = {
        index: node_id
        for index, node_id in _source_ids_by_scipy_index(induced).items()
    }
    provenance: dict[str, NodeProvenance] = {}
    for local, original in source_of.items():
        source_id = source_ids[original]
        provenance[local_node_ids[local]] = NodeProvenance(
            source_node_id=source_id,
            source_count=int(tree.nodes[source_id]["descendant_count"]),
            surviving_count=size[local],
            compressed_source_node_ids=tuple(
                source_ids[value] for value in compressed[local]
            ),
        )
    return induced, provenance


def surviving_rows(
    feature_matrix: np.ndarray,
    full_item_ids: Sequence[str],
    surviving_item_ids: Sequence[str],
) -> np.ndarray:
    """Rows of the original feature representation for the survivors, in order.

    Columns are left untouched: each survivor keeps its complete original feature
    vector, which is what makes a full-cohort recut reproduce the shipped cut.
    """
    matrix = np.asarray(feature_matrix)
    if matrix.shape[0] != len(full_item_ids):
        raise ValueError("feature_matrix rows must match the tree's item IDs")
    index = {item_id: position for position, item_id in enumerate(full_item_ids)}
    return matrix[[index[item_id] for item_id in surviving_item_ids]]


def recut(
    tree: Tree,
    surviving_item_ids: Iterable[str],
    feature_matrix: np.ndarray,
    settings: CutSettings | None = None,
) -> InducedView:
    """Induce the hierarchy over a cohort and recompute its nested optimal cut.

    Cohorts at or below ``settings.individual_threshold`` are not cut: they are
    presented as individuals. When no admissible cut exists or the score is
    undefined, the view is returned with ``cut_status == "undefined"`` and a
    reason instead of an arbitrary fallback partition.
    """
    settings = settings or CutSettings()
    induced, provenance = induce_tree(tree, surviving_item_ids)
    count = induced.leaf_count

    def view(cut: Any, status: str, reason: str | None = None) -> InducedView:
        return InducedView(
            source_tree_id=tree.tree_id,
            tree=induced,
            provenance=provenance,
            cut=cut,
            cut_status=status,
            cut_reason=reason,
            settings=settings,
        )

    if count <= settings.individual_threshold:
        return view(
            None,
            CUT_INDIVIDUALS,
            f"{count} tools is at or below the individual-display threshold "
            f"of {settings.individual_threshold}",
        )

    rows = surviving_rows(feature_matrix, tree.item_ids, induced.item_ids)
    if not np.any(np.ptp(np.asarray(rows, dtype=np.float64), axis=0) > 0):
        # Every survivor has the same feature vector, so every partition scores
        # the same; reporting any of them as the optimum would be arbitrary.
        return view(
            None,
            CUT_UNDEFINED,
            "surviving feature rows are identical, so the cut score is undefined",
        )
    try:
        cut = build_dynamic_cut(
            induced,
            top_criterion="optimal",
            min_width=settings.min_width,
            max_width=settings.max_width,
            target_width=settings.target_width,
            max_depth=settings.max_depth,
            cut_optimizer=settings.cut_optimizer,
            cut_polarity=settings.cut_polarity,
            feature_matrix=rows,
        )
    except ValueError as exc:
        return view(None, CUT_UNDEFINED, str(exc))
    validate_dynamic_cut(cut, induced)
    return view(cut, CUT_OPTIMAL)
