"""
retriever.py
------------
Hybrid retrieval: BM25 keyword search + semantic search, merged with
Reciprocal Rank Fusion, then reranked with a cross-encoder.

What you learn here:
- Reciprocal Rank Fusion (RRF):
    Each result gets a score of  1 / (k + rank)  from each search method.
    These scores are summed across methods. A result that appears at rank 1
    in both searches gets a very high combined score; one that only appears
    in one search gets a lower score. k=60 prevents top-ranked results from
    dominating too heavily.

    Example with k=60:
        BM25 rank 1  → 1/(60+1) = 0.0164
        Semantic rank 1 → 1/(60+1) = 0.0164
        Combined score if in both → 0.0328  ← rises to the top

- Why RRF over score normalization:
    BM25 scores and cosine similarity scores live in completely different
    ranges. Instead of trying to normalize them (which is fragile), RRF
    only uses *ranks*, which are always comparable.

- Two-stage pipeline:
    Stage 1 (fast): BM25 + semantic each return HYBRID_TOP_K=10 candidates.
                    RRF merges them into up to 20 unique candidates.
    Stage 2 (accurate): Cross-encoder reranks the 20 → keeps FINAL_TOP_K=3.
"""

from typing import Any, Dict, List

from .bm25 import BM25Index
from .embedder import EmbeddingManager
from .reranker import Reranker
from .vectorstore import VectorStore
import config


def _reciprocal_rank_fusion(
    *result_lists: List[Dict[str, Any]],
    k: int = config.RRF_K,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        *result_lists: any number of ranked lists (each a list of chunk dicts)
        k: RRF constant (default 60)

    Returns:
        A single merged list, sorted by combined RRF score (best first).
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            doc_id = chunk["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = chunk

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    merged = []
    for doc_id in sorted_ids:
        chunk = dict(docs[doc_id])
        chunk["rrf_score"] = round(scores[doc_id], 6)
        merged.append(chunk)

    return merged


class RAGRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingManager,
        bm25_index: BM25Index,
        reranker: Reranker,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = bm25_index
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = config.FINAL_TOP_K,
        hybrid_k: int = config.HYBRID_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline for a single query.

        Steps:
            1. Embed the query → semantic search in ChromaDB (top hybrid_k)
            2. BM25 keyword search (top hybrid_k)
            3. RRF merge both result lists
            4. Cross-encoder rerank merged list → return top_k

        Returns:
            List of chunk dicts with keys:
                id, document, metadata, score, rrf_score, rerank_score
        """
        # ── 1. Semantic search ─────────────────────────────────────────────────
        query_embedding = self.embedder.embed([query])[0]
        raw = self.vector_store.query(
            query_embedding=query_embedding.tolist(), top_k=hybrid_k
        )

        semantic_results = []
        if raw["documents"] and raw["documents"][0]:
            for doc_id, doc, metadata, distance in zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            ):
                semantic_results.append(
                    {
                        "id": doc_id,
                        "document": doc,
                        "metadata": metadata,
                        "score": round(1 - distance, 4),
                    }
                )

        # ── 2. BM25 keyword search ─────────────────────────────────────────────
        bm25_results = self.bm25.search(query, top_k=hybrid_k)

        # ── 3. Reciprocal Rank Fusion ──────────────────────────────────────────
        merged = _reciprocal_rank_fusion(semantic_results, bm25_results)

        print(
            f"[retriever] '{query[:55]}' → "
            f"semantic={len(semantic_results)}, "
            f"bm25={len(bm25_results)}, "
            f"merged={len(merged)}"
        )

        # ── 4. Cross-encoder rerank ────────────────────────────────────────────
        final = self.reranker.rerank(query, merged, top_k=top_k)
        print(f"[retriever] After reranking → {len(final)} chunks")
        return final
