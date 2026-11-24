"""Phrase-aware term statistics for the @2 enrichers (label clarity, success criterion 3).

Differences from ``text.NodeTerms``: bigrams are candidate terms (adjacent content words only),
generic adjectives/adverbs/verbs, weekdays, months and ``-ly`` words are dropped, corpus-very-common
words are dropped by document frequency, terms that look like past-tense verbs are down-weighted,
and a bigram that is frequent in a cluster is preferred over its words. No POS tagger.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from orchardlogic.cultivate.enrichers import text as T

WEEKDAYS = frozenset("monday tuesday wednesday thursday friday saturday sunday".split())
MONTHS = frozenset(
    "january february march april june july august september october november december".split()
)
# Small fixed list; the corpus-derived part is the document-frequency cut in ``build_tfidf2``.
GENERIC = frozenset(
    """
    excellent allowed short late options option case cases default hands hand good bad big small
    long high low early simple easy hard real true full important possible different similar general
    special specific main major whole certain various several recent current previous next last
    little large huge great amazing awesome powerful perfect better best worse worst enough
    actually basically totally completely simply exactly likely probably
    allow allows include includes including included provide provides provided create creates
    created build builds built give gives given keep keeps kept put puts set sets turn turns
    running run runs ran add adds added based related due especially overall essentially
    summer winter spring autumn
    implement implements maintain approach methodology doing included follows putting improving
    marked drive choice breakdown sold video videos follow follows making taking getting using
    towards toward within without whether
    anyone everyone people person thing stuff kind sort type types part parts piece pieces bit
    number numbers level levels point points area areas place places side sides end ends
    top bottom left right
    """.split()
)
STOP2 = T.STOPWORDS | GENERIC | WEEKDAYS | MONTHS
_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
MAX_DF = 0.3  # a term in more than 30% of documents is generic for this corpus


def _keep(word: str) -> bool:
    if not 3 <= len(word) <= 25 or word in STOP2:
        return False
    if len(word) > 4 and word.endswith("ly"):
        return False
    return True


@lru_cache(maxsize=2048)
def analyze(text: str) -> tuple[str, ...]:
    """Unigrams of kept words plus bigrams of directly adjacent kept words."""
    cleaned = T.clean_text(text).lower().replace("’", "'")
    words: list[str | None] = []
    for raw in re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*", cleaned):
        if raw.endswith("'s"):
            raw = raw[:-2]
        words.append(raw if "'" not in raw and _keep(raw) else None)
    out: list[str] = [w for w in words if w]
    out += [f"{a} {b}" for a, b in zip(words, words[1:]) if a and b and a != b]
    return tuple(out)


def build_tfidf2(texts):
    n = len(texts)
    attempts = [(2, MAX_DF), (1, 1.0)] if n >= 8 else [(1, 1.0)]
    for min_df, max_df in attempts:
        vec = TfidfVectorizer(
            analyzer=analyze, sublinear_tf=True, min_df=min_df, max_df=max_df, dtype=np.float64
        )
        try:
            matrix = vec.fit_transform(texts)
        except ValueError:
            continue
        if matrix.shape[1] > 0:
            return matrix.tocsr(), np.asarray(vec.get_feature_names_out())
    return sparse.csr_matrix((n, 0)), np.asarray([], dtype=str)


def _words(term: str) -> list[str]:
    return term.split(" ")


def overlaps(term: str, chosen) -> bool:
    return any(
        T.is_variant(a, b) for other in chosen for a in _words(term) for b in _words(other)
    )


def pick_phrases(terms, k: int, exclude=()) -> list[str]:
    """First ``k`` terms sharing no word (or morphological variant) with earlier/excluded ones."""
    chosen: list[str] = []
    for term in terms:
        if overlaps(term, (*exclude, *chosen)):
            continue
        chosen.append(term)
        if len(chosen) == k:
            break
    return chosen


class NodeTerms2(T.NodeTerms):
    _mult: np.ndarray | None = None

    def multipliers(self) -> np.ndarray:
        if self._mult is None:
            mult = np.ones(len(self.vocab))
            for i, term in enumerate(self.vocab):
                if term.endswith("ed") and " " not in term:
                    mult[i] = 0.6  # likely a past-tense verb, not a topic
                elif " " in term and term.split(" ")[-1].endswith("ed"):
                    mult[i] = 0.6
            self._mult = mult
        return self._mult

    def scored(self, node_key: str) -> list[str]:
        """All candidate terms for the node, best first (bigrams boosted when frequent)."""
        n = self.doc_count(node_key)
        if n == 0 or self.vocab.size == 0:
            return []
        mean = self.mean([node_key])
        cov = self.coverage([node_key])
        score = mean * np.sqrt(cov) * self.multipliers()
        bigram = np.fromiter((" " in t for t in self.vocab), dtype=bool, count=len(self.vocab))
        frequent = bigram & (cov * n >= 3 - 1e-9) & (cov >= 0.3)
        score = np.where(frequent, score * 1.5, np.where(bigram, score * 0.7, score))
        if n >= 2:
            score = np.where(cov * n >= 2 - 1e-9, score, 0.0)
        order = sorted(np.flatnonzero(score > 0), key=lambda i: (-score[i], self.vocab[i]))
        return [str(self.vocab[i]) for i in order]

    def ranked_terms(self, node_key: str, k: int = 8) -> list[str]:
        return pick_phrases(self.scored(node_key), k)


def node_terms2(ctx) -> NodeTerms2:
    return T.node_terms(ctx, build=build_tfidf2, cls=NodeTerms2, slot="v2")


def compose_title(terms, max_terms: int = 3, max_words: int = 5) -> str:
    """Join leading terms with the title separator; at most ``max_words`` words overall."""
    chosen: list[str] = []
    words = 0
    for term in terms:
        w = len(_words(term))
        if chosen and (len(chosen) >= max_terms or words + w > max_words):
            continue
        chosen.append(term)
        words += w
        if len(chosen) >= max_terms:
            break
    return T.title_from_terms(chosen)


_SUFFIX_SPLIT = re.compile(r"\s+[|–—]\s+|\s+-\s+")


_TRAILING = frozenset(
    "a an the of in on to for and or with how why what is are at by from as that this your our my".split()
)


_CONNECT = frozenset("to for in of with and at on from by".split())


def short_item_title(title: str, max_words: int = 7) -> str:
    """A document title without site suffixes, at most ``max_words`` words, cut at a clean point.

    ``Podcast: Topic`` style short prefixes are kept; ``Title | Site`` / ``Title - Site`` lose the
    site; a longer title is cut before its first colon (if that leaves 3+ words) or trimmed so it
    does not end on a connective.
    """
    title = T.clean_title(title).strip("\"'“”‘ ")
    parts = [p for p in _SUFFIX_SPLIT.split(title) if p.strip()]
    if parts:
        title = parts[0] if len(parts[0].split()) >= 2 or len(parts) == 1 else max(parts, key=len)
    words = title.split()
    if len(words) > 5 and "," in title:
        head = title.split(",", 1)[0]
        if len(head.split()) >= 3:
            return head.strip()
    if len(words) > max_words:
        head, sep, _ = title.partition(":")
        if sep and 3 <= len(head.split()) <= max_words:
            return head.strip()
        words = words[:max_words]
        cut = next(
            (i for i in range(len(words) - 1, 2, -1) if words[i].lower() in _CONNECT), None
        )
        if cut is not None:
            words = words[:cut]
        while len(words) > 2 and words[-1].lower().strip(",;:-") in _TRAILING:
            words.pop()
        title = " ".join(words).rstrip(",;:-–—")
    return title
