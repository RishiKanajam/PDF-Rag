"""
bm25.py
-------
Keyword-based search using BM25 (Best Match 25).

What you learn here:
- BM25: a classical information retrieval algorithm (used by Elasticsearch,
  Lucene, etc.). It scores documents by how many query words they contain,
  adjusted for document length and term frequency — no ML involved at all.
- Why BM25 + semantic is better than either alone:
    • Semantic search excels at meaning/paraphrase but misses exact terms.
    • BM25 excels at exact terms (acronyms, names, numbers) but misses synonyms.
    • Together they cover each other's blind spots.
- Persistence: BM25 doesn't have its own database, so we pickle (serialize) the
  index to disk and reload it on the next run — avoiding a full rebuild.
- Tokenization: we lowercase and split on whitespace. Simple but effective for
  English text. Punctuation is kept, which is fine for BM25's purposes.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

import config


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        # Each entry: {"id": str, "document": str, "metadata": dict}
        self._docs: List[Dict[str, Any]] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, docs: List[Dict[str, Any]]):
        """
        Build the BM25 index from a list of document dicts.

        Args:
            docs: list of {"id": ..., "document": ..., "metadata": ...}
        """
        self._docs = docs
        tokenized = [doc["document"].lower().split() for doc in docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"[bm25] Index built from {len(docs)} documents")

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = config.HYBRID_TOP_K) -> List[Dict[str, Any]]:
        """
        Return top_k documents ranked by BM25 score.
        Each result has the same shape as a semantic retrieval result:
            {"id", "document", "metadata", "score"}
        so they can be merged with semantic results in the retriever.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() or load() first.")

        tokens = query.lower().split()
        scores: np.ndarray = self._bm25.get_scores(tokens)

        # Get indices of top_k highest scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # skip docs with zero relevance
                doc = self._docs[idx]
                results.append(
                    {
                        "id": doc["id"],
                        "document": doc["document"],
                        "metadata": doc["metadata"],
                        "score": float(scores[idx]),
                    }
                )
        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path = config.BM25_INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "docs": self._docs}, f)
        print(f"[bm25] Index saved to {path}")

    def load(self, path: Path = config.BM25_INDEX_PATH) -> bool:
        """Load index from disk. Returns True if successful, False if file missing."""
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._docs = data["docs"]
        print(f"[bm25] Index loaded — {len(self._docs)} documents")
        return True

    def is_empty(self) -> bool:
        return self._bm25 is None or len(self._docs) == 0
