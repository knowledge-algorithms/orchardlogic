"""Cultivate a topic tree into a TreeDB database, then place a new document into it.

Runs offline. The encoder below is a small hashed bag-of-words stand-in so the example
needs no model download; with ``orchardlogic[embeddings]`` installed, leave ``encoder``
out to use the MiniLM encoder named in the perspective.

    python examples/cultivate_into_treedb.py [output_dir]
"""

from __future__ import annotations

import hashlib
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from orchardlogic.cultivate import Document
from orchardlogic.cultivate.cultivate import cultivate_into_db, insert_items
from orchardlogic.cultivate.perspectives import topic_perspective
from orchardlogic.treedb import cutstore, registry, store

VOCAB = {
    "cooking": "simmer saute braise oven recipe garlic onion broth skillet dough".split(),
    "astronomy": "telescope galaxy nebula orbit planet comet eclipse quasar star lunar".split(),
    "gardening": "compost seedling mulch prune soil trowel perennial bloom harvest weed".split(),
    "football": "striker midfield goalkeeper penalty league offside tackle referee coach corner".split(),
}


def encode(texts, dim: int = 96) -> np.ndarray:
    out = np.zeros((len(texts), dim))
    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z][a-z\-]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            out[row, int.from_bytes(digest[:4], "little") % dim] += 1.0
    return out


def make_docs(per_topic: int = 8, seed: int = 7) -> list[Document]:
    rng = random.Random(seed)
    docs = []
    for topic, vocab in VOCAB.items():
        for n in range(per_topic):
            text = " ".join(" ".join(rng.sample(vocab, 6)).capitalize() + "." for _ in range(3))
            docs.append(Document(text=text, item_id=f"{topic}-{n:02d}", title=f"{topic} note {n}"))
    return docs


def main(out: Path) -> None:
    perspective = topic_perspective(cuts=[{
        "name": "topics", "top_criterion": "cluster_count", "cluster_count": 4,
        "min_width": 3, "max_width": 6, "target_width": 5, "max_depth": 3,
    }])
    db_path = out / "orchard.duckdb"
    imported = cultivate_into_db(
        make_docs(), perspective, name="demo-topic", db=db_path, alias="demo/topic",
        encoder=encode, out_dir=out / "cultivar",
    )
    print("cultivar artifact:", imported.artifact_sha256[:12])

    con = store.connect_with_retry(db_path)
    try:
        rows = cutstore.read_cut(con, registry.resolve(con, "demo/topic"), "topics")
    finally:
        con.close()
    top = [r for r in rows if r["depth"] == 1]
    print(f"stored cut 'topics': {len(rows)} nodes, {len(top)} top-level groups")

    new = Document(text="Garlic and onion simmer in broth; braise in the oven.", item_id="cooking-new")
    result = insert_items(db_path, "demo/topic", [new], encoder=encode)
    print("placed:", result.new_item_ids, "attached to tree:", result.tree_attached)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="orchardlogic-"))
    target.mkdir(parents=True, exist_ok=True)
    main(target)
