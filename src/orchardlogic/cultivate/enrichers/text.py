"""Shared text utilities for the built-in enrichers: cleaning, terms, sentences, node stats.

Pure numpy/scikit-learn and deterministic. Terms avoid stopwords, web boilerplate, URLs and
numbers so that titles read at a glance.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from orchardlogic.cultivate.document import Document

TITLE_SEP = " · "
MAX_TITLE_WORDS = 6

BOILERPLATE = frozenset(
    """
    click clicking subscribe subscribed subscription unsubscribe cookie cookies privacy policy
    newsletter newsletters share sharing tweet facebook twitter linkedin instagram youtube
    email emails login log sign signup register comment comments reply leave menu navigation
    skip footer header sidebar copyright rights reserved terms advertisement sponsored ads
    download join free today read reading learn
    https http www com org net html amp nbsp quot href
    just really also like get got make makes made way ways thing things need needs want wants
    use used using uses new one two three first second even much many may might will would
    could should let lets said say says going know see look find take back well
    lot lots every something anything everything someone people year years day days time
    times now still often always never ever already yet whether another across around
    idea ideas proven unlock work works working best great good better start help helps world
    life check episode episodes show shows week weeks post posts blog article story stories
    tips tip step steps guide welcome introducing introduction
    joined join began begin felt discuss discussed discusses dived dive covered cover watch
    watched mentioned mention listen listened delivered deliver minute minutes sure came
    come went going gone told tell asked ask thought think wanted tried try instead let's
    """.split()
)
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | BOILERPLATE
ACRONYMS = frozenset({"ai", "llm", "llms", "api", "gpt", "pkm", "seo", "ux", "ui", "sql", "pdf"})

_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_NOISE_RE = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)|[*_`~]{1,3}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"”’)]*\s+(?=[\"“‘(]*[A-Z0-9])")
_BOILER_SENT_RE = re.compile(
    r"\b(subscribe|click here|cookie|sign up|log in|privacy policy|all rights reserved|"
    r"newsletter|follow us|share this)\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Unescape HTML entities and drop markdown syntax, images and URLs (keeps link text)."""
    text = html.unescape(text or "")
    text = _MD_IMAGE_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _MD_NOISE_RE.sub("", text)
    return text


def clean_title(title: str) -> str:
    text = html.unescape(title or "")
    text = _URL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> tuple[str, ...]:
    """Lowercase content words: letters only, no stopwords/boilerplate, 3 to 25 chars."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(clean_text(text).lower()):
        token = raw.replace("’", "'")
        if token.endswith("'s"):
            token = token[:-2]
        if 3 <= len(token) <= 25 and "'" not in token and token not in STOPWORDS:
            out.append(token)
    return tuple(out)


# Re-analysing every document on each incremental refresh dominated insert latency.
tokens = lru_cache(maxsize=2048)(_tokens)


def split_sentences(text: str, *, max_sentences: int = 300) -> list[str]:
    """Readable sentences (40-400 chars, mostly letters, no boilerplate), in order."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n+", clean_text(text)):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if len(paragraph) < 40:
            continue
        for sentence in _SENT_SPLIT_RE.split(paragraph):
            sentence = sentence.strip()
            if not 40 <= len(sentence) <= 400 or len(sentence.split()) < 6:
                continue
            letters = sum(ch.isalpha() for ch in sentence)
            if letters / len(sentence) < 0.6 or _BOILER_SENT_RE.search(sentence):
                continue
            sentences.append(sentence)
            if len(sentences) >= max_sentences:
                return sentences
    return sentences


def titlecase_term(term: str) -> str:
    return " ".join(
        "-".join(part.upper() if part in ACRONYMS else part.capitalize() for part in word.split("-"))
        for word in term.split(" ")
    )


def _stem(term: str) -> str:
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(term) > len(suffix) + 3 and term.endswith(suffix) and not term.endswith("ss"):
            return term[: -len(suffix)] + repl
    return term


def is_variant(a: str, b: str) -> bool:
    sa, sb = _stem(a), _stem(b)
    if sa == sb:
        return True
    short, long_ = sorted((sa, sb), key=len)
    return len(short) >= 4 and long_.startswith(short) and len(long_) - len(short) <= 3


def pick_distinct(terms: Iterable[str], k: int, exclude: Sequence[str] = ()) -> list[str]:
    """First ``k`` terms that are not morphological variants of an earlier/excluded term."""
    chosen: list[str] = []
    for term in terms:
        if any(is_variant(term, other) for other in (*exclude, *chosen)):
            continue
        chosen.append(term)
        if len(chosen) == k:
            break
    return chosen


def title_from_terms(terms: Sequence[str]) -> str:
    return TITLE_SEP.join(titlecase_term(t) for t in terms)


def dedupe_sibling_titles(
    titles: Mapping[str, str],
    groups: Iterable[Sequence[str]],
    candidates: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Make titles unique within each sibling group.

    For a collision, the later sibling gets its best candidate term not already in its title
    appended (``Title · Term``); if none helps, a numeric suffix is added.
    """
    result = dict(titles)
    for group in groups:
        seen: dict[str, str] = {}
        for key in group:
            title = result.get(key, "")
            if not title:
                continue
            norm = title.casefold()
            if norm in seen:
                used = {w.casefold() for w in re.split(r"\s*·\s*", title)}
                extra = next(
                    (
                        t
                        for t in candidates.get(key, ())
                        if titlecase_term(t).casefold() not in used
                        and (title + TITLE_SEP + titlecase_term(t)).casefold() not in seen
                    ),
                    None,
                )
                if extra is not None:
                    title = title + TITLE_SEP + titlecase_term(extra)
                else:
                    n = 2
                    while f"{title} ({n})".casefold() in seen:
                        n += 1
                    title = f"{title} ({n})"
                result[key] = title
                norm = title.casefold()
            seen[norm] = key
    return result


def limit_title_words(title: str, max_words: int = MAX_TITLE_WORDS) -> str:
    """Clip an item-derived title to ``max_words`` words."""
    words = title.split()
    if len(words) <= max_words:
        return title
    return " ".join(words[:max_words]).rstrip(",;:-–—")


def doc_stat_text(doc: Document, max_chars: int = 6000) -> str:
    return f"{clean_title(doc.title)} {doc.text[:max_chars]}"


def build_tfidf(texts: Sequence[str]) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Sublinear TF-IDF over ``texts``; returns (matrix, vocabulary array).

    Very common terms (df above 50%) and singletons are dropped when the corpus is big enough
    for that to make sense.
    """
    n = len(texts)
    attempts = [(2, 0.5), (1, 1.0)] if n >= 8 else [(1, 1.0)]
    for min_df, max_df in attempts:
        vec = TfidfVectorizer(
            analyzer=tokens, sublinear_tf=True, min_df=min_df, max_df=max_df, dtype=np.float64
        )
        try:
            matrix = vec.fit_transform(texts)
        except ValueError:
            continue
        if matrix.shape[1] > 0:
            return matrix.tocsr(), np.asarray(vec.get_feature_names_out())
    return sparse.csr_matrix((n, 0)), np.asarray([], dtype=str)


@dataclass
class NodeTerms:
    """TF-IDF statistics aggregated over the cut's nodes."""

    item_row: dict[str, int]
    matrix: sparse.csr_matrix
    binary: sparse.csr_matrix
    vocab: np.ndarray
    members: dict[str, list[str]]  # root/cluster node_key -> item_ids beneath it
    nodes: dict[str, dict]
    children: dict[str, list[str]] = field(default_factory=dict)  # cluster children only

    def siblings(self, node_key: str) -> list[str]:
        parent = self.nodes[node_key].get("parent_key")
        if parent is None:
            return []
        return [k for k in self.children.get(parent, []) if k != node_key]

    def _rows(self, node_keys: Iterable[str]) -> list[int]:
        rows: list[int] = []
        for key in node_keys:
            rows.extend(self.item_row[i] for i in self.members.get(key, ()) if i in self.item_row)
        return sorted(set(rows))

    def mean(self, node_keys: Iterable[str]) -> np.ndarray:
        rows = self._rows(node_keys)
        if not rows or self.matrix.shape[1] == 0:
            return np.zeros(self.matrix.shape[1])
        return np.asarray(self.matrix[rows].mean(axis=0)).ravel()

    def coverage(self, node_keys: Iterable[str]) -> np.ndarray:
        rows = self._rows(node_keys)
        if not rows or self.matrix.shape[1] == 0:
            return np.zeros(self.matrix.shape[1])
        return np.asarray(self.binary[rows].mean(axis=0)).ravel()

    def doc_count(self, node_key: str) -> int:
        return len(self._rows([node_key]))

    def ranked_terms(self, node_key: str, k: int = 8) -> list[str]:
        """Corpus-distinctive terms: mean TF-IDF weighted by sqrt(coverage)."""
        n = self.doc_count(node_key)
        if n == 0 or self.vocab.size == 0:
            return []
        mean = self.mean([node_key])
        cov = self.coverage([node_key])
        score = mean * np.sqrt(cov)
        if n >= 2:
            # a term seen in a single document of a multi-item cluster is not a topic
            score = np.where(cov * n >= 2 - 1e-9, score, 0.0)
        order = sorted(np.flatnonzero(score > 0), key=lambda i: (-score[i], self.vocab[i]))
        return pick_distinct((str(self.vocab[i]) for i in order), k)


_CACHE: dict[str, tuple] = {}


def node_terms(ctx, *, build=None, cls=None, slot: str = "entry") -> NodeTerms:  # ctx: TreeContext
    """Build (and memoise per ``slot`` for the most recent context) node-level term statistics."""
    build = build or build_tfidf
    cls = cls or NodeTerms
    nodes = {n["node_key"]: n for n in ctx.nodes}
    item_rows = [n for n in nodes.values() if n["kind"] == "item"]
    item_ids = sorted({n["item_id"] for n in item_rows})
    key = (id(ctx.docs), tuple(item_ids), tuple((k, n["parent_key"]) for k, n in nodes.items()))
    cached = _CACHE.get(slot)
    if cached is not None and cached[0] == key and cached[1] is ctx.docs:
        return cached[2]
    matrix, vocab = build([doc_stat_text(ctx.docs[i]) for i in item_ids])
    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    members: dict[str, list[str]] = {k: [] for k, n in nodes.items() if n["kind"] != "item"}
    for row in item_rows:
        for anc in row["path"]:
            if anc in members:
                members[anc].append(row["item_id"])
    children: dict[str, list[str]] = {}
    for node in sorted(nodes.values(), key=lambda n: (n["depth"], n["position"])):
        if node["kind"] == "cluster":
            children.setdefault(node["parent_key"], []).append(node["node_key"])
    result = cls(
        item_row={i: r for r, i in enumerate(item_ids)},
        matrix=matrix,
        binary=binary,
        vocab=vocab,
        members=members,
        nodes=nodes,
        children=children,
    )
    _CACHE[slot] = (key, ctx.docs, result)
    return result


def target_nodes(ctx, nodes: Mapping[str, dict]) -> list[str]:
    """Root/cluster node_keys to compute, honouring ``ctx.only``; parents before children."""
    keys = [
        k
        for k, n in sorted(nodes.items(), key=lambda kv: (kv[1]["depth"], kv[1]["position"]))
        if n["kind"] != "item"
    ]
    if ctx.only is not None:
        keys = [k for k in keys if k in ctx.only]
    return keys
