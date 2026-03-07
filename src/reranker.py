"""
reranker.py
-----------
Re-scores retrieved chunks using a cross-encoder model.

What you learn here:
- Bi-encoder vs Cross-encoder:
    • Bi-encoder (what we use for embedding): encodes query and document
      *separately* into vectors, then measures distance. Fast — you pre-compute
      all document vectors once and reuse them. But it loses fine-grained
      interaction between query words and document words.
    • Cross-encoder: takes the query and a document *together* as a single
      input and outputs a relevance score. Slower (can't pre-compute), but much
      more accurate because the model sees both at once and can reason about
      their interaction word-by-word.
- Two-stage retrieval: use the fast bi-encoder to get 10 candidates cheaply,
  then use the slow cross-encoder to pick the best 3 accurately. Best of both.
- ms-marco: the model is trained on the MS MARCO passage ranking dataset —
  millions of real (query, passage, relevance) triples from Bing searches.
  This makes it very good at judging if a passage answers a question.
"""

from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

import config


class Reranker:
    def __init__(self, model_name: str = config.RERANKER_MODEL):
        print(f"[reranker] Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        print("[reranker] Ready")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = config.FINAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Re-score chunks against the query and return the top_k best ones.

        The cross-encoder sees (query, chunk_text) together and outputs a
        relevance score. Higher score = more relevant.

        Args:
            query:  the user's question
            chunks: candidate results from hybrid search (each has 'document' key)
            top_k:  how many to return after reranking

        Returns:
            top_k chunks sorted by cross-encoder score (best first),
            with a new 'rerank_score' key added to each.
        """
        if not chunks:
            return []

        pairs = [(query, chunk["document"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        # Attach rerank scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)

        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
