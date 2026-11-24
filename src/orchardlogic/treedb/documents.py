"""Documents with capture metadata (docs/DATA_ORCHARD_V4.md section 5.1, amendments A8, A10).

``store.get_documents`` stays the read path for ``treedb.documents``. This module adds the
writer ``upsert_documents`` (registers ``item_registry`` rows, normalizes ``captured_at`` /
``updated_at`` to UTC ISO ``...Z``), URL helpers and ``recent_items``.

``v_item_time`` casts ``metadata->>'captured_at'`` with ``TRY_CAST(... AS TIMESTAMP)``, which
silently drops a time-zone offset; hence the UTC ``Z`` normalization here.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import duckdb

from orchardlogic.treedb.fixtures import canonical_json, item_uuid
from orchardlogic.treedb.tx import transaction

TIME_KEYS = ("captured_at", "updated_at")
_OFFSET_RE = re.compile(r"([+-])(\d{2})(?::?(\d{2}))?$")


def normalize_timestamp(value: Any) -> str | None:
    """UTC ISO-8601 with ``Z`` and always 6-digit microseconds (``2026-01-02T03:04:05.000000Z``), or ``None``.

    A naive value (no offset) is taken as UTC. PostgreSQL CSV exports of ``timestamptz`` (for example
    ``2026-01-02 03:04:05+00``) are accepted; export them in UTC or with their offset, never as naive
    local time. Accepts ``datetime`` objects and strings such as ``2026-01-02T03:04:05+02:00``,
    ``2026-01-02 03:04:05+00`` (PostgreSQL CSV), ``...Z`` or naive (taken as UTC).
    Raises ``ValueError`` for anything else.
    """
    if value is None or value == "":
        return None
    if isinstance(value, _dt.datetime):
        dt = value
    elif isinstance(value, _dt.date):
        dt = _dt.datetime(value.year, value.month, value.day)
    else:
        s = str(value).strip().replace(" ", "T", 1)
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        else:
            m = _OFFSET_RE.search(s)
            if m and "T" in s and m.start() > s.index("T"):          # offset like +00 or +0200
                s = s[:m.start()] + f"{m.group(1)}{m.group(2)}:{m.group(3) or '00'}"
        dt = _dt.datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def timestamp_param(value: Any) -> str:
    """UTC wall-clock text (``YYYY-MM-DD HH:MM:SS[.f]``) for ``CAST(? AS TIMESTAMP)``."""
    norm = normalize_timestamp(value)
    if norm is None:
        raise ValueError("timestamp must not be empty")
    return norm[:-1].replace("T", " ")


def normalize_url(url: str) -> str:
    """Lowercase scheme and host, drop the fragment, strip a trailing ``/`` of a path-only URL (A8)."""
    parts = urlsplit(url.strip())
    path = parts.path
    if path == "/" and not parts.query:
        path = ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def item_id_for_url(url: str, prefix: str = "vb:") -> str:
    """``vb:`` + sha256(normalized url)[:16] (A8)."""
    return prefix + hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class UpsertResult:
    inserted: tuple[str, ...]     # item_ids that had no document before
    updated: tuple[str, ...]      # item_ids whose document row was overwritten

    @property
    def item_ids(self) -> tuple[str, ...]:
        return self.inserted + self.updated


def _prepare(doc: Mapping[str, Any]) -> tuple[str, str | None, str, str | None]:
    item_id = str(doc.get("item_id") or "").strip()
    if not item_id:
        raise ValueError("document with an empty item_id")
    text = doc.get("text")
    if not isinstance(text, str):
        raise ValueError(f"document {item_id!r}: text must be a string")
    title = doc.get("title")
    if title is not None and not isinstance(title, str):
        raise ValueError(f"document {item_id!r}: title must be a string")
    meta = doc.get("metadata")
    if meta is not None:
        if not isinstance(meta, Mapping):
            raise ValueError(f"document {item_id!r}: metadata must be an object")
        meta = dict(meta)
        for key in TIME_KEYS:
            if meta.get(key) is not None:
                meta[key] = normalize_timestamp(meta[key])
    return item_id, title, text, None if meta is None else canonical_json(meta)


def upsert_documents(con: duckdb.DuckDBPyConnection, docs: Sequence[Mapping[str, Any]]) -> UpsertResult:
    """Insert or overwrite documents; register ``item_registry`` rows. One transaction (or the caller's).

    Each doc: ``item_id``, ``title``, ``text``, ``metadata`` (object). ``captured_at`` and
    ``updated_at`` in ``metadata`` are normalized to UTC ``Z`` strings. A repeated item_id in
    one call: the last one wins.
    """
    rows: dict[str, tuple] = {}
    for d in docs:
        row = _prepare(d)
        rows[row[0]] = row
    if not rows:
        return UpsertResult((), ())
    ids = list(rows)
    with transaction(con):
        have = {r[0] for r in con.execute(
            "SELECT item_id FROM treedb.documents WHERE item_id IN (SELECT unnest(?))", [ids]).fetchall()}
        con.execute(
            "INSERT INTO treedb.item_registry (item_id, item_uuid) SELECT unnest(?::VARCHAR[]), unnest(?::UUID[]) "
            "ON CONFLICT (item_id) DO NOTHING", [ids, [str(item_uuid(i)) for i in ids]])
        con.execute("DELETE FROM treedb.documents WHERE item_id IN (SELECT unnest(?))", [ids])
        con.execute(
            "INSERT INTO treedb.documents (item_id, title, text, metadata) "
            "SELECT unnest(?::VARCHAR[]), unnest(?::VARCHAR[]), unnest(?::VARCHAR[]), unnest(?::JSON[])",
            [ids, [rows[i][1] for i in ids], [rows[i][2] for i in ids], [rows[i][3] for i in ids]])
    return UpsertResult(tuple(i for i in ids if i not in have), tuple(i for i in ids if i in have))


def existing_item_ids(con: duckdb.DuckDBPyConnection, item_ids: Sequence[str]) -> list[str]:
    """Subset of ``item_ids`` that already has a document row, sorted."""
    rows = con.execute("SELECT item_id FROM treedb.documents WHERE item_id IN (SELECT unnest(?))",
                       [list(item_ids)]).fetchall()
    return sorted(r[0] for r in rows)


def find_by_url(con: duckdb.DuckDBPyConnection, url: str) -> list[str]:
    """item_ids whose ``metadata.url`` equals ``url`` after normalization (A8), sorted."""
    target = normalize_url(url)
    rows = con.execute(
        "SELECT item_id, metadata->>'url' FROM treedb.documents WHERE metadata->>'url' IS NOT NULL").fetchall()
    return sorted(i for i, u in rows if normalize_url(u) == target)


def recent_items(con: duckdb.DuckDBPyConnection, *, since: str | _dt.datetime | None = None,
                 limit: int = 50, with_text: bool = False) -> list[dict[str, Any]]:
    """Documents newest first by ``captured_at`` (items without one last).

    Keys: ``item_id``, ``title``, ``metadata`` (parsed), ``captured_at`` (UTC ``Z`` string or
    ``None``) and ``text`` when ``with_text``. ``since`` keeps ``captured_at >= since``.
    """
    where, params = "", []
    if since is not None:
        where = "WHERE t.captured_at >= CAST(? AS TIMESTAMP)"
        params.append(timestamp_param(since))
    rows = con.execute(
        "SELECT d.item_id, d.title, d.metadata, t.captured_at, d.text FROM treedb.documents d "
        f"JOIN treedb.v_item_time t USING (item_id) {where} "
        "ORDER BY t.captured_at DESC NULLS LAST, d.item_id LIMIT ?", params + [int(limit)]).fetchall()
    out = []
    for item_id, title, meta, cap, text in rows:
        rec = {"item_id": item_id, "title": title,
               "metadata": None if meta is None else json.loads(meta),
               "captured_at": None if cap is None else normalize_timestamp(cap)}
        if with_text:
            rec["text"] = text
        out.append(rec)
    return out


__all__ = ["UpsertResult", "existing_item_ids", "find_by_url", "item_id_for_url", "normalize_timestamp",
           "normalize_url", "recent_items", "timestamp_param", "upsert_documents"]
