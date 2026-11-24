"""item.extract@1: per-document title, centrality summary, dates, key terms, word count."""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from orchardlogic.cultivate.document import Document
from orchardlogic.cultivate.enrichers import text as T
from orchardlogic.cultivate.enrichers.base import make_record, register_enricher

SUMMARY_SENTENCES = 2
KEY_TERMS = 8

RECORD = make_record(
    name="item.extract",
    version="1",
    scope="item",
    method="extractive",
    params={"summary_sentences": SUMMARY_SENTENCES, "key_terms": KEY_TERMS},
    title="Item extract",
    description=(
        "Document title, top sentences by TF-IDF centrality, ISO dates, key terms and word count."
    ),
    module="orchardlogic.cultivate.enrichers.item_extract",
    module_file=__file__,
)

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ISO_PREFIX_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may", "may"),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "september", "sept"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}
_MONTH_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def _valid(year: int, month: int, day: int) -> str | None:
    if not 1900 <= year <= 2100:
        return None
    try:
        return _dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def find_dates(doc: Document) -> list[str]:
    found: set[str] = set()
    for y, m, d in _ISO_RE.findall(doc.text):
        iso = _valid(int(y), int(m), int(d))
        if iso:
            found.add(iso)
    for name, d, y in _MONTH_RE.findall(doc.text):
        iso = _valid(int(y), _MONTHS[name.lower()], int(d))
        if iso:
            found.add(iso)
    for key in ("captured_at", "published"):
        value = doc.metadata.get(key)
        if isinstance(value, str):
            match = _ISO_PREFIX_RE.match(value.strip())
            if match:
                iso = _valid(*(int(g) for g in match.groups()))
                if iso:
                    found.add(iso)
    return sorted(found)


def derive_title(doc: Document) -> str:
    title = T.clean_title(doc.title)
    if title:
        return title
    heading = _HEADING_RE.search(doc.text)
    if heading:
        return T.limit_title_words(T.clean_title(heading.group(1)), 12)
    sentences = T.split_sentences(doc.text, max_sentences=1)
    if sentences:
        return T.limit_title_words(sentences[0], 12)
    first = T.clean_title(T.clean_text(doc.text)[:120])
    return T.limit_title_words(first, 12)


def central_sentences(text: str, k: int = SUMMARY_SENTENCES) -> list[str]:
    """Top ``k`` sentences by TF-IDF cosine to the whole document, in document order."""
    sentences = T.split_sentences(text)
    if len(sentences) <= k:
        return sentences
    vec = TfidfVectorizer(analyzer=T.tokens, sublinear_tf=True)
    try:
        matrix = vec.fit_transform([*sentences, " ".join(sentences)])
    except ValueError:
        return sentences[:k]
    doc_row = matrix[-1]
    sims = np.asarray((matrix[:-1] @ doc_row.T).todense()).ravel()
    order = sorted(range(len(sentences)), key=lambda i: (-sims[i], i))[:k]
    return [sentences[i] for i in sorted(order)]


class ItemExtract:
    record = RECORD

    def enrich(self, docs: Sequence[Document]) -> dict[str, dict]:
        docs = list(docs)
        if not docs:
            return {}
        vec = TfidfVectorizer(analyzer=T.tokens, sublinear_tf=True)
        try:
            matrix = vec.fit_transform([T.doc_stat_text(d, 20000) for d in docs]).tocsr()
            vocab = vec.get_feature_names_out()
        except ValueError:
            matrix, vocab = None, []
        out: dict[str, dict] = {}
        for i, doc in enumerate(docs):
            terms: list[str] = []
            if matrix is not None:
                row = matrix.getrow(i)
                ranked = sorted(zip(row.indices, row.data), key=lambda p: (-p[1], vocab[p[0]]))
                terms = T.pick_distinct((str(vocab[j]) for j, _ in ranked), KEY_TERMS)
            out[doc.item_id] = {
                "title": derive_title(doc),
                "summary": " ".join(central_sentences(doc.text)),
                "dates": find_dates(doc),
                "key_terms": terms,
                "word_count": len(re.findall(r"\w+", doc.text)),
            }
        return out


register_enricher("item.extract@1", ItemExtract)
