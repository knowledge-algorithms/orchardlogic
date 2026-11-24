"""Content-addressed vectors (docs/DATA_ORCHARD_V4.md section 5.1, A10).

``treedb.vector`` holds one float32 vector per (space, sha256 of the embedded text).
``treedb.item_vector`` links items to them. A ``space`` string must encode model, revision,
pooling and text template (``minilm-l6-v2@<rev8>/mean/tt1``); this module does not interpret it.

Reads use ``fetchnumpy`` (not ``fetchall``). Writes go through a text round trip, which is
about 2.5x faster than passing nested Python lists, and exact for float32.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import duckdb
import numpy as np

from orchardlogic.treedb.tx import transaction

_CHUNK = 1000


def _rows_to_text(m: np.ndarray) -> list[str]:
    return ["[" + ",".join(r) + "]" for r in m.astype("U24").tolist()]


def put_vectors(con: duckdb.DuckDBPyConnection, space: str, rows: Iterable[tuple[str, np.ndarray]]) -> int:
    """Store ``(content_sha256, vector)`` rows; existing (space, sha) rows are skipped. Returns rows added."""
    items: dict[str, np.ndarray] = {}
    for sha, vec in rows:
        v = np.asarray(vec, dtype=np.float32)
        if v.ndim != 1 or v.size == 0:
            raise ValueError(f"vector for {sha!r} must be 1-D and non-empty")
        items.setdefault(sha, v)
    if not items:
        return 0
    added = 0
    with transaction(con):
        have_dims = {r[0] for r in con.execute(
            "SELECT DISTINCT dim FROM treedb.vector WHERE space = ?", [space]).fetchall()}
        dims = {v.size for v in items.values()} | have_dims
        if len(dims) != 1:
            raise ValueError(f"space {space!r}: mixed vector dimensions {sorted(dims)}")
        shas = list(items)
        for i in range(0, len(shas), _CHUNK):
            part = shas[i:i + _CHUNK]
            m = np.stack([items[s] for s in part])
            before = con.execute("SELECT count(*) FROM treedb.vector WHERE space = ?", [space]).fetchone()[0]
            con.execute(
                "INSERT INTO treedb.vector (space, content_sha256, dim, vec) "
                "SELECT ?, s, ?, CAST(v AS FLOAT[]) FROM "
                "(SELECT unnest(?::VARCHAR[]) AS s, unnest(?::VARCHAR[]) AS v) "
                "ON CONFLICT DO NOTHING", [space, int(m.shape[1]), part, _rows_to_text(m)])
            added += con.execute("SELECT count(*) FROM treedb.vector WHERE space = ?", [space]).fetchone()[0] - before
    return added


def _stack(col: np.ndarray, dim_hint: int = 0) -> np.ndarray:
    if len(col) == 0:
        return np.zeros((0, dim_hint), dtype=np.float32)
    return np.stack(list(col)).astype(np.float32, copy=False)


def get_vectors(con: duckdb.DuckDBPyConnection, space: str, shas: Sequence[str]) -> dict[str, np.ndarray]:
    """``{content_sha256: float32 vector}`` for the shas that exist (missing ones are absent)."""
    wanted = list(dict.fromkeys(shas))
    if not wanted:
        return {}
    r = con.execute("SELECT content_sha256, vec FROM treedb.vector WHERE space = ? "
                    "AND content_sha256 IN (SELECT unnest(?))", [space, wanted]).fetchnumpy()
    m = _stack(r["vec"])
    return {s: m[i] for i, s in enumerate(r["content_sha256"].tolist())}


def link_items(con: duckdb.DuckDBPyConnection, space: str, pairs: Iterable[tuple[str, str]]) -> None:
    """Upsert ``(item_id, content_sha256)`` links in ``space``."""
    pairs = list(dict(pairs).items())
    if not pairs:
        return
    with transaction(con):
        con.execute(
            "INSERT OR REPLACE INTO treedb.item_vector (item_id, space, content_sha256) "
            "SELECT unnest(?::VARCHAR[]), ?, unnest(?::VARCHAR[])",
            [[p[0] for p in pairs], space, [p[1] for p in pairs]])


def item_matrix(con: duckdb.DuckDBPyConnection, space: str,
                item_ids: Sequence[str] | None = None) -> tuple[list[str], np.ndarray]:
    """``(ids, float32 matrix)``. Rows follow ``item_ids`` order (items without a vector are left out);
    with ``item_ids=None`` every linked item, sorted by item_id."""
    base = ("SELECT iv.item_id, v.vec FROM treedb.item_vector iv JOIN treedb.vector v "
            "ON v.space = iv.space AND v.content_sha256 = iv.content_sha256 WHERE iv.space = ?")
    if item_ids is None:
        r = con.execute(base + " ORDER BY iv.item_id", [space]).fetchnumpy()
        return r["item_id"].tolist(), _stack(r["vec"])
    wanted = list(dict.fromkeys(item_ids))
    if not wanted:
        return [], np.zeros((0, 0), dtype=np.float32)
    r = con.execute(base + " AND iv.item_id IN (SELECT unnest(?))", [space, wanted]).fetchnumpy()
    pos = {s: i for i, s in enumerate(r["item_id"].tolist())}
    ids = [i for i in wanted if i in pos]
    m = _stack(r["vec"])
    return ids, (m[[pos[i] for i in ids]] if ids else m)


def nearest(con: duckdb.DuckDBPyConnection, space: str, query: np.ndarray, k: int = 10,
            item_ids: Sequence[str] | None = None) -> list[tuple[str, float]]:
    """Top-``k`` ``(item_id, cosine)`` by ``list_cosine_similarity`` in SQL, best first."""
    q = np.asarray(query, dtype=np.float32).ravel().tolist()
    sql = ("SELECT iv.item_id, list_cosine_similarity(v.vec, ?::FLOAT[]) AS s FROM treedb.item_vector iv "
           "JOIN treedb.vector v ON v.space = iv.space AND v.content_sha256 = iv.content_sha256 WHERE iv.space = ?")
    params: list = [q, space]
    if item_ids is not None:
        sql += " AND iv.item_id IN (SELECT unnest(?))"
        params.append(list(item_ids))
    sql += " ORDER BY s DESC NULLS LAST, iv.item_id LIMIT ?"
    params.append(int(k))
    return [(i, float(s)) for i, s in con.execute(sql, params).fetchall() if s is not None]


__all__ = ["get_vectors", "item_matrix", "link_items", "nearest", "put_vectors"]
