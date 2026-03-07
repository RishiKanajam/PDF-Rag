"""
embedder.py
-----------
Converts text into numerical vectors (embeddings).

What you learn here:
- Embeddings: a sentence/paragraph is turned into a fixed-size list of numbers
  (384 numbers for all-MiniLM-L6-v2) that capture its *meaning*. Two chunks
  about similar topics will have vectors that are close together in space.
- SentenceTransformer: runs locally on your machine — no API call, no cost.
  all-MiniLM-L6-v2 is small (80 MB), fast, and good enough for most RAG tasks.
- Why this matters for RAG: at query time we embed the user's question and find
  the document chunks whose vectors are nearest to it — that's semantic search.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

import config


class EmbeddingManager:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self):
        print(f"[embedder] Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_sentence_embedding_dimension()
        print(f"[embedder] Ready — embedding dimension: {dim}")

    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Return a (N, dim) numpy array of embeddings for the given texts."""
        if not self.model:
            raise RuntimeError("Embedding model is not loaded.")
        embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        return embeddings
