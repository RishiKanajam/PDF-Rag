"""
rag.py
------
Orchestrates the full RAG pipeline: load → chunk → embed → store → retrieve → generate.
Supports both single-turn query() and multi-turn chat() with memory.

What you learn here:
- RAG pattern: instead of asking an LLM to answer from its training data alone,
  we *retrieve* relevant text from our own documents, *augment* the prompt with
  that context, then let the LLM *generate* a grounded answer.
- Two-phase design:
    build()  → runs once. Loads, chunks, embeds, stores in ChromaDB + BM25.
    query()  → single-turn: retrieves context and calls the LLM.
    chat()   → multi-turn: reformulates the question using history, retrieves,
               answers, then saves the exchange to memory.
- Question reformulation: before retrieval, a fast LLM call rewrites a
  follow-up question into a self-contained question. This is critical —
  retrieval operates on the reformulated question, not the raw follow-up,
  so "tell me more about it" actually finds the right chunks.
- Smart build: only rebuilds what's actually missing (ChromaDB or BM25).
- Prompt engineering: the prompt tells the LLM to use ONLY the provided context.
"""

import hashlib
from typing import Any, Dict, List

from langchain_groq import ChatGroq

import config
from .bm25 import BM25Index
from .chunker import split_documents
from .embedder import EmbeddingManager
from .loader import load_all
from .memory import ConversationMemory
from .reranker import Reranker
from .retriever import RAGRetriever
from .vectorstore import VectorStore


# ── Prompt templates ───────────────────────────────────────────────────────────

_ANSWER_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the context below.
- If asked to summarize or give an overview, summarize all the provided context thoroughly.
- If the answer is genuinely not present in the context, say "I don't have enough information to answer that."
- Do not refuse to answer just because the question uses casual phrasing like "uploaded file" or "this document" — treat those as referring to the context provided.
{history_block}
Context from documents:
{context}

Question: {question}

Answer:"""

_REFORMULATE_TEMPLATE = """\
Given the conversation history below and a follow-up question, rewrite the \
follow-up as a standalone question that makes sense without the history.
If the question is already standalone, return it unchanged. Return ONLY the \
rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


class RAGPipeline:
    def __init__(self):
        self.embedder     = EmbeddingManager()
        self.vector_store = VectorStore()
        self.bm25         = BM25Index()
        self.reranker     = Reranker()
        self.retriever    = RAGRetriever(
            self.vector_store, self.embedder, self.bm25, self.reranker
        )
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

    def build(self, force: bool = False):
        """
        Index all documents into ChromaDB and build the BM25 index.

        Smart behaviour (no force):
          - ChromaDB has data + BM25 index exists  → skip entirely
          - ChromaDB has data + BM25 index missing → rebuild BM25 from stored docs
          - ChromaDB empty                          → full build

        Args:
            force: if True, re-loads, re-chunks, re-embeds, and rebuilds everything.
        """
        chroma_has_data = not self.vector_store.is_empty()
        bm25_loaded     = (not force) and self.bm25.load()

        if not force and chroma_has_data and bm25_loaded:
            print(
                f"[rag] Index already built — {self.vector_store.count} vectors, "
                "BM25 loaded from disk. Use --rebuild to force."
            )
            return

        if not force and chroma_has_data and not bm25_loaded:
            # ChromaDB is fine; just rebuild BM25 from what's already stored
            print("[rag] BM25 index missing — rebuilding from stored vectors...")
            all_docs = self.vector_store.get_all_documents()
            self.bm25.build(all_docs)
            self.bm25.save()
            print("[rag] BM25 rebuild complete.")
            return

        # ── Full build ─────────────────────────────────────────────────────────
        print("[rag] Building full index...")
        documents = load_all()
        if not documents:
            raise RuntimeError("No documents found. Add PDFs or text files to Data/")

        chunks = split_documents(documents)
        texts  = [c.page_content for c in chunks]

        print(f"[rag] Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.embed(texts, show_progress=True)

        self.vector_store.upsert_documents(chunks, embeddings)

        # Build BM25 with the same MD5 IDs used by the vector store
        bm25_docs = [
            {
                "id":       hashlib.md5(chunk.page_content.encode()).hexdigest(),
                "document": chunk.page_content,
                "metadata": {k: str(v) for k, v in chunk.metadata.items()},
            }
            for chunk in chunks
        ]
        self.bm25.build(bm25_docs)
        self.bm25.save()
        print("[rag] Full index build complete.")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _retrieve_and_format(self, question: str, top_k: int):
        """Run retrieval and return (chunks, context_string, sources_list)."""
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            return [], "", []
        context = "\n\n---\n\n".join(c["document"] for c in chunks)
        sources = [
            {
                "source":       c["metadata"].get("source", "unknown"),
                "page":         c["metadata"].get("page", "N/A"),
                "rerank_score": c.get("rerank_score", "N/A"),
            }
            for c in chunks
        ]
        return chunks, context, sources

    def _reformulate(self, question: str, memory: ConversationMemory) -> str:
        """
        Rewrite a follow-up question into a standalone question using history.
        If there's no history, returns the question unchanged (no LLM call).
        """
        if memory.is_empty:
            return question
        prompt = _REFORMULATE_TEMPLATE.format(
            history=memory.format_for_prompt(),
            question=question,
        )
        response = self.llm.invoke(prompt)
        standalone = response.content.strip()
        if standalone != question:
            print(f"[rag] Reformulated: '{question}' → '{standalone}'")
        return standalone

    # ── Public API ─────────────────────────────────────────────────────────────

    def delete_source(self, source_name: str):
        """
        Remove all vectors for a given source file, then rebuild BM25 from what remains.
        Also deletes the file from disk.
        """
        import shutil
        # Remove from disk
        for directory in (config.PDF_DIR, config.TEXT_DIR):
            candidate = directory / source_name
            if candidate.exists():
                candidate.unlink()
                break

        # Remove from ChromaDB
        deleted = self.vector_store.delete_by_source(source_name)

        # Rebuild BM25 from remaining vectors
        remaining = self.vector_store.get_all_documents()
        if remaining:
            self.bm25.build(remaining)
            self.bm25.save()
        else:
            # Nothing left — remove BM25 file too
            if config.BM25_INDEX_PATH.exists():
                config.BM25_INDEX_PATH.unlink()

        return deleted

    def query(self, question: str, top_k: int = config.FINAL_TOP_K) -> Dict[str, Any]:
        """
        Single-turn question answering (no memory).

        Returns:
            answer   – the LLM's response string
            sources  – list of {source, page, rerank_score} for traceability
            chunks   – raw retrieved chunks (for debugging)
        """
        if self.vector_store.is_empty():
            raise RuntimeError("Index is empty. Run build() first.")

        chunks, context, sources = self._retrieve_and_format(question, top_k)

        if not chunks:
            return {
                "answer":  "I couldn't find any relevant information in the documents.",
                "sources": [],
                "chunks":  [],
            }

        prompt = _ANSWER_TEMPLATE.format(
            history_block="",
            context=context,
            question=question,
        )
        response = self.llm.invoke(prompt)
        return {"answer": response.content, "sources": sources, "chunks": chunks}

    def chat(
        self,
        question: str,
        memory: ConversationMemory,
        top_k: int = config.FINAL_TOP_K,
    ) -> Dict[str, Any]:
        """
        Multi-turn question answering with conversation memory.

        Steps:
            1. Reformulate the question using history (so retrieval works for follow-ups).
            2. Retrieve relevant chunks using the standalone question.
            3. Build a prompt that includes both history and retrieved context.
            4. Call the LLM and save the exchange to memory.

        Args:
            question: the user's raw (possibly follow-up) question
            memory:   ConversationMemory instance shared across turns
            top_k:    number of chunks to retrieve

        Returns:
            Same structure as query(): {answer, sources, chunks}
            Plus: standalone_question — what was actually used for retrieval
        """
        if self.vector_store.is_empty():
            raise RuntimeError("Index is empty. Run build() first.")

        # Step 1: reformulate
        standalone = self._reformulate(question, memory)

        # Step 2: retrieve with the standalone question
        chunks, context, sources = self._retrieve_and_format(standalone, top_k)

        if not chunks:
            answer = "I couldn't find any relevant information in the documents."
            memory.add_user(question)
            memory.add_assistant(answer)
            return {"answer": answer, "sources": [], "chunks": [], "standalone_question": standalone}

        # Step 3: build prompt with history
        history_text = memory.format_for_prompt()
        history_block = (
            f"\nPrevious conversation:\n{history_text}\n"
            if history_text else ""
        )
        prompt = _ANSWER_TEMPLATE.format(
            history_block=history_block,
            context=context,
            question=question,  # show the original question in the prompt, not reformulated
        )

        # Step 4: generate and save to memory
        response = self.llm.invoke(prompt)
        answer = response.content

        memory.add_user(question)
        memory.add_assistant(answer)

        return {
            "answer":              answer,
            "sources":             sources,
            "chunks":              chunks,
            "standalone_question": standalone,
        }
