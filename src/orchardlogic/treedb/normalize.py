"""Normalized SciPy linkage from the raw compressed subtree.

**Status: PROPOSED, pending Gate G3 (OD-1).** CONTRACTS INV-16 is a candidate, not an
accepted contract. Nothing here changes the raw output (D-004): ``treedb.read``
returns the PostgreSQL contract unchanged, and this module is a separate, pure
function over those rows.

``to_scipy_linkage(raw_rows)`` -> ``(Z, leaf_item_ids, context)``:

* **Stem row dropped into ``context``.** The unary row the raw output carries when the
  entries' LCA is below the root (D-015) becomes ``context["root_branch_id"]`` and
  ``context["root_coph_distance"]`` (the original root's branch and height), with
  ``context["lca_is_root"] = False``. When there is no stem row the top binary row is
  the root: ``lca_is_root = True`` and the same two keys name it.
* **0-based numbering.** Leaves ``0..m-1``, internal nodes ``m..2m-2`` (INV-8), the
  last row is the LCA. Indices are rebuilt from node ids, not by shifting the raw
  indexes, so NULL raw indexes (non-leaf entries) need no special case.
* **No placeholder leaf.** Unlike ``graph_review.ipynb`` (Disc §2: the stem row's
  missing child became index 0, a phantom leaf, and its count was one short), every
  leaf index is a real entry, and the stem row is not a Z row.
* **Consistent counts.** ``Z[:, 3]`` is recomputed as the number of leaves below each
  node. It must equal the raw ``doc_count`` (raw counts start nodes); a mismatch means
  the raw output is not a tree over distinct entries (for example an item that hangs
  twice after a duplicate attach, PG_BASELINE §9) and raises ``NormalizationError``.

Proposed choices for G3 (each isolated, each documented in docs/porting/P3B.md, N-1..N-5):

* N-1 leaf order ``leaf_order="raw"`` (default): the raw 1-based leaf index minus one
  (the cluster_packer ``-1`` shift, valid for leaves), so normalized leaf ``i`` is raw
  leaf ``i+1``. Leaves without a raw index (non-leaf entries) follow, by node id.
  ``leaf_order="item_id"``: ascending item id (INV-16 candidate); needs ``item_ids_by_uuid``.
* N-2 internal order: a topological order (children first, required by SciPy) taking
  the lowest ``(coph_distance, raw row position)`` first; for monotone heights this is
  the raw row order.
* N-3 left/right: smaller normalized index on the left (SciPy's own ``linkage`` output
  convention); deterministic and structural.
* N-4 a non-leaf entry (INV-7) is one opaque leaf (count 1, as raw ``doc_count`` counts
  it); its ``leaf_item_ids`` slot is ``None`` and its node id is in
  ``context["leaf_node_ids"]`` and ``context["non_leaf_entries"]``. Expanding it into its
  own subtree (INV-16 candidate, OD-8) is not done.
* N-5 a single entry gives ``Z`` of shape ``(0, 4)`` (INV-16 candidate).
"""

from __future__ import annotations

import heapq
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

#: Same column order as ``treedb.read.COMPRESSED_COLUMNS`` (INV-14).
RAW_COLUMNS: tuple[str, ...] = (
    "tree_id", "branch_id", "branch_ix", "l_node_ix", "r_node_ix", "l_node_id", "r_node_id",
    "l_edge_ix", "r_edge_ix", "doc_count", "coph_distance",
)

STATUS = "proposed (OD-1, pending G3)"


class NormalizationError(ValueError):
    """The raw rows do not describe one tree over distinct entries."""


def _as_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {c: row[c] for c in RAW_COLUMNS}
    if len(row) != len(RAW_COLUMNS):
        raise NormalizationError(f"raw row has {len(row)} values, expected {len(RAW_COLUMNS)}")
    return dict(zip(RAW_COLUMNS, row))


def _u(v: Any) -> uuid.UUID | None:
    if v is None:
        return None
    return v if isinstance(v, uuid.UUID) else uuid.UUID(str(v))


def to_scipy_linkage(
    raw_rows: Sequence[Any],
    *,
    item_ids_by_uuid: Mapping[uuid.UUID, str] | None = None,
    leaf_order: str = "raw",
) -> tuple[np.ndarray, list[str | None], dict[str, Any]]:
    """Normalize raw ``compressed_subtree`` rows to a 0-based SciPy linkage (proposed, OD-1).

    ``raw_rows``: tuples in ``RAW_COLUMNS`` order or mappings with those keys, as returned
    by ``treedb.read.compressed_subtree``. ``item_ids_by_uuid`` maps leaf node UUIDs to
    item ids (without it, ``leaf_item_ids`` holds the leaf UUIDs as strings).

    Returns ``Z`` (float64, shape ``(m-1, 4)``), the ``m`` leaf ids in normalized order,
    and ``context`` (see the module docstring).
    """
    if leaf_order not in ("raw", "item_id"):
        raise ValueError("leaf_order must be 'raw' or 'item_id'")
    if leaf_order == "item_id" and item_ids_by_uuid is None:
        raise ValueError("leaf_order='item_id' needs item_ids_by_uuid")
    rows = [_as_dict(r) for r in raw_rows]
    context: dict[str, Any] = {"status": STATUS, "raw_rows": len(rows), "leaf_order": leaf_order}
    if not rows:
        context.update(tree_id=None, lca_is_root=None, root_branch_id=None, root_coph_distance=None,
                       lca_node_id=None, leaf_node_ids=[], node_ids=[], non_leaf_entries=[])
        return np.zeros((0, 4)), [], context
    trees = {_u(r["tree_id"]) for r in rows}
    if len(trees) != 1:
        raise NormalizationError(f"raw rows span {len(trees)} trees")
    context["tree_id"] = trees.pop()

    unary = [i for i, r in enumerate(rows) if r["r_node_id"] is None]
    binary = [i for i, r in enumerate(rows) if r["r_node_id"] is not None]
    for i in unary:
        if rows[i]["l_node_id"] is None:
            raise NormalizationError(f"raw row {i} has no child")
    if len(unary) > 1:
        raise NormalizationError(f"{len(unary)} unary rows; a raw output has at most one stem row (D-015)")

    # Binary structure: branch -> (children, raw position).
    branch_of: dict[uuid.UUID, int] = {}
    for i in binary:
        b = _u(rows[i]["branch_id"])
        if b in branch_of:
            raise NormalizationError(f"branch {b} has more than one raw row")
        branch_of[b] = i
    parent: dict[uuid.UUID, uuid.UUID] = {}
    child_meta: dict[uuid.UUID, tuple[int | None, int | None]] = {}   # node -> (raw ix, edge_ix)
    for i in binary:
        b = _u(rows[i]["branch_id"])
        for s in ("l", "r"):
            c = _u(rows[i][f"{s}_node_id"])
            if c in parent:
                raise NormalizationError(f"node {c} has two parents in the raw rows")
            parent[c] = b
            child_meta[c] = (rows[i][f"{s}_node_ix"], rows[i][f"{s}_edge_ix"])
    tops = [b for b in branch_of if b not in parent]

    if unary:
        stem = rows[unary[0]]
        lca = _u(stem["l_node_id"])
        if binary and (len(tops) != 1 or tops[0] != lca):
            raise NormalizationError("the stem row's child is not the top of the binary rows")
        if not binary:
            child_meta[lca] = (stem["l_node_ix"], stem["l_edge_ix"])
        context.update(lca_is_root=False, root_branch_id=_u(stem["branch_id"]),
                       root_coph_distance=float(stem["coph_distance"]), root_doc_count=int(stem["doc_count"]),
                       lca_node_id=lca, lca_edge_ix=stem["l_edge_ix"])
    else:
        if len(tops) != 1:
            raise NormalizationError(f"{len(tops)} top rows and no stem row")
        top = rows[branch_of[tops[0]]]
        lca = tops[0]
        context.update(lca_is_root=True, root_branch_id=lca, root_coph_distance=float(top["coph_distance"]),
                       root_doc_count=int(top["doc_count"]), lca_node_id=lca, lca_edge_ix=None)

    # Leaves: every child that is not itself a binary branch (or the lone stem child).
    leaves = [n for n in child_meta if n not in branch_of]
    if binary and lca in child_meta:
        raise NormalizationError("the LCA appears as a child of a binary row")

    def item_of(n: uuid.UUID) -> str | None:
        return None if item_ids_by_uuid is None else item_ids_by_uuid.get(n)

    if leaf_order == "raw":                                                       # N-1
        leaves.sort(key=lambda n: (child_meta[n][0] is None, child_meta[n][0] or 0, str(n)))
    else:
        leaves.sort(key=lambda n: (item_of(n) is None, item_of(n) or "", str(n)))
    m = len(leaves)
    index: dict[uuid.UUID, int] = {n: k for k, n in enumerate(leaves)}
    count: dict[uuid.UUID, int] = {n: 1 for n in leaves}

    # Internal nodes: Kahn's algorithm, lowest (coph_distance, raw position) first (N-2).
    pending = {b: 2 for b in branch_of}
    ready: list[tuple[float, int, str]] = []
    kids_of: dict[uuid.UUID, list[uuid.UUID]] = {b: [] for b in branch_of}
    for c, b in parent.items():
        kids_of[b].append(c)
    for c, b in parent.items():
        if c not in branch_of:
            pending[b] -= 1
    for b, k in pending.items():
        if k == 0:
            heapq.heappush(ready, (float(rows[branch_of[b]]["coph_distance"]), branch_of[b], str(b)))
    z_rows: list[list[float]] = []
    order: list[uuid.UUID] = []
    while ready:
        _, pos, bs = heapq.heappop(ready)
        b = uuid.UUID(bs)
        r = rows[pos]
        a, c = sorted((index[k] for k in kids_of[b]))                             # N-3
        index[b] = m + len(z_rows)
        count[b] = sum(count[k] for k in kids_of[b])
        if count[b] != r["doc_count"]:
            raise NormalizationError(
                f"branch {b}: raw doc_count {r['doc_count']} != {count[b]} distinct entries below it")
        z_rows.append([float(a), float(c), float(r["coph_distance"]), float(count[b])])
        order.append(b)
        if b in parent:
            p = parent[b]
            pending[p] -= 1
            if pending[p] == 0:
                heapq.heappush(ready, (float(rows[branch_of[p]]["coph_distance"]), branch_of[p], str(p)))
    if len(z_rows) != len(binary):
        raise NormalizationError("the binary rows contain a cycle")
    if m != len(binary) + 1:
        raise NormalizationError(f"{m} leaves for {len(binary)} binary rows")
    if unary and context["root_doc_count"] != m:
        raise NormalizationError(f"stem doc_count {context['root_doc_count']} != {m} entries")

    Z = np.asarray(z_rows, dtype=float).reshape(len(z_rows), 4)
    leaf_ids: list[str | None] = []
    non_leaf: list[int] = []
    for k, n in enumerate(leaves):
        if child_meta[n][1] != 1:
            non_leaf.append(k)
        if item_ids_by_uuid is None:
            leaf_ids.append(str(n) if child_meta[n][1] == 1 else None)
        else:
            leaf_ids.append(item_ids_by_uuid.get(n) if child_meta[n][1] == 1 else None)
    if item_ids_by_uuid is not None:
        unmapped = [str(n) for k, n in enumerate(leaves) if k not in non_leaf and leaf_ids[k] is None]
        if unmapped:
            raise NormalizationError(f"leaf node(s) without an item id: {unmapped[:10]}")
    context.update(
        leaf_node_ids=list(leaves),
        node_ids=list(leaves) + order,               # normalized index -> node id
        non_leaf_entries=non_leaf,
        leaf_count=m,
    )
    return Z, leaf_ids, context


def scipy_linkage_for_items(con, cultivar, item_ids, *, on_missing: str = "report",
                            leaf_order: str = "raw") -> tuple[np.ndarray, list[str | None], dict[str, Any]]:
    """Convenience over a store: resolve items (``treedb.read.resolve_items``), read the raw
    subtree, normalize it. ``context`` also carries the resolution (``missing_items``,
    ``unknown_items``, ``state_version``), the full tree's leaf row count, and the stem
    path above the LCA from ``get_ancestry_paths`` (``stem``: ``[(branch_id,
    coph_distance)]`` from the LCA's parent up to the root), as INV-16 suggests."""
    from orchardlogic.treedb import read

    res = read.resolve_items(con, cultivar, item_ids, on_missing=on_missing)
    raw = read.compressed_subtree(con, res.orchard_ids)
    by_uuid = {u: i for i, u in con.execute(
        "SELECT item_id, item_uuid FROM treedb.item_registry WHERE list_contains(CAST(? AS VARCHAR[]), item_id)",
        [list(res.item_rows)]).fetchall()}
    Z, leaves, ctx = to_scipy_linkage(raw, item_ids_by_uuid=by_uuid, leaf_order=leaf_order)
    ctx.update(cultivar=res.cultivar, state_version=res.state_version, missing_items=list(res.missing_items),
               unknown_items=list(res.unknown_items),
               full_tree_leaf_rows=con.execute("SELECT count(*) FROM treedb.v_orchard WHERE tree_id = ? AND edge_ix = 1",
                                               [res.tree_id]).fetchone()[0])
    stem: list[tuple[uuid.UUID, float]] = []
    if ctx.get("lca_is_root") is False:
        lca = ctx["lca_node_id"]
        nres = read.resolve_nodes(con, res.tree_id, [lca])
        chain = read.ancestry_paths(con, nres.orchard_ids)
        # depth 0 is the LCA's own incoming row: its branch is the LCA's parent.
        stem = [(r[2], r[5]) for r in sorted(chain, key=lambda r: r[6])]
    ctx["stem"] = stem
    return Z, leaves, ctx


__all__ = ["NormalizationError", "RAW_COLUMNS", "STATUS", "scipy_linkage_for_items", "to_scipy_linkage"]
