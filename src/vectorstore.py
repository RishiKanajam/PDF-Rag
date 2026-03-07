"""
vectorstore.py
--------------
Manages the ChromaDB persistent vector database.

What you learn here:
- ChromaDB: a local vector database that stores embeddings on disk so you
  don't have to re-embed documents every time you run the program.
- collection.upsert() vs collection.add():
    - add()    → crashes if a document with the same ID already exists
    - upsert() → inserts if new, updates if exists (safe to call repeatedly)
- Deterministic IDs: we hash each chunk's content with MD5 so the same chunk
  always gets the same ID. This is what prevents duplicate vectors — if you
  run the pipeline twice, the second run just overwrites the same IDs.
"""

import hashlib
import os
from typing import Any, Dict, List

import chromadb
import numpy as np
from langchain_core.documents import Document

import config


def _make_id(text: str) -> str:
    """Generate a stable, unique ID from the chunk's text content."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class VectorStore:
    def __init__(
        self,
        collection_name: str = config.COLLECTION_NAME,
        persist_directory: str = str(config.VECTOR_STORE_DIR),
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Vector DB for RAG documents"},
        )
        print(
            f"[vectorstore] Collection '{self.collection_name}' ready — "
            f"{self.collection.count()} vectors stored"
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def upsert_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Add or update documents in the vector store.
        Uses content-based IDs so re-running never creates duplicates.
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings must match.")

        ids, texts, metadatas, vecs = [], [], [], []

        for doc, embedding in zip(documents, embeddings):
            doc_id = _make_id(doc.page_content)

            metadata = {k: str(v) for k, v in doc.metadata.items()}
            metadata["content_length"] = str(len(doc.page_content))

            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(metadata)
            vecs.append(embedding.tolist())

        self.collection.upsert(
            ids=ids,
            embeddings=vecs,
            metadatas=metadatas,
            documents=texts,
        )
        print(
            f"[vectorstore] Upserted {len(documents)} chunks — "
            f"total vectors now: {self.collection.count()}"
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = config.HYBRID_TOP_K,
    ) -> Dict[str, Any]:
        """Return the top_k nearest chunks for a given query embedding."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Fetch every stored chunk as a list of dicts.
        Used to rebuild the BM25 index from an existing ChromaDB store
        without re-loading and re-chunking the source documents.
        """
        results = self.collection.get(include=["documents", "metadatas"])
        docs = []
        for doc_id, text, metadata in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            docs.append({"id": doc_id, "document": text, "metadata": metadata})
        return docs

    def is_empty(self) -> bool:
        return self.collection.count() == 0
