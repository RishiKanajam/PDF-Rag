"""
chunker.py
----------
Splits documents into chunks using two strategies:

  "sentence"  (default) — NLTK sentence-aware splitting
  "recursive"           — character-based (original approach)

What you learn here:

WHY CHUNKING STRATEGY MATTERS MORE THAN YOU THINK:
  The chunk is the unit of retrieval. If a chunk cuts mid-sentence, the
  retrieved text is incomplete and the LLM may produce a wrong answer even
  though the right information was in your documents.

SENTENCE-AWARE CHUNKING (strategy="sentence"):
  1. NLTK tokenizes the page text into individual sentences.
  2. Sentences are accumulated until the chunk would exceed CHUNK_SIZE chars.
  3. When the limit is hit, we save the chunk and start a new one — but we
     carry forward the last few sentences as overlap so context isn't lost
     at boundaries.
  → Result: every chunk starts and ends at a clean sentence boundary.

RECURSIVE CHARACTER SPLITTING (strategy="recursive"):
  Tries \\n\\n → \\n → space → character as fallback separators.
  Fast and simple, but can cut mid-sentence when no good separator is nearby.

METADATA ENRICHMENT:
  Every chunk gets extra metadata attached:
    chunk_index  — position of this chunk within its source document
    total_chunks — how many chunks came from the same source document
    word_count   — number of words in this chunk
  This lets you filter retrieved results and gives full traceability.
"""

from typing import List

import nltk
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# Download the NLTK sentence tokenizer model if not already present (~1 MB).
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ── Sentence-aware chunker ─────────────────────────────────────────────────────

def _sentence_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into chunks that always start and end at sentence boundaries.

    Algorithm:
        1. Tokenize into sentences with NLTK.
        2. Greedily accumulate sentences until adding one more would exceed chunk_size.
        3. Save the chunk, then backtrack by overlap-chars worth of sentences
           so the next chunk has context without cutting mid-sentence.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(sentences):
        current_chars = 0
        end = start

        while end < len(sentences):
            added = len(sentences[end]) + (1 if end > start else 0)
            if current_chars + added > chunk_size and end > start:
                break
            current_chars += added
            end += 1

        chunks.append(" ".join(sentences[start:end]))

        if end >= len(sentences):
            break

        # Step back overlap-worth of sentences for the next chunk
        overlap_chars = 0
        overlap_start = end
        while overlap_start > start and overlap_chars < chunk_overlap:
            overlap_start -= 1
            overlap_chars += len(sentences[overlap_start])

        start = max(overlap_start, start + 1)  # never get stuck

    return chunks


# ── Recursive character chunker (original) ────────────────────────────────────

def _recursive_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


# ── Metadata enrichment ────────────────────────────────────────────────────────

def _enrich_metadata(chunks: List[Document], total: int, offset: int) -> List[Document]:
    """Add chunk_index, total_chunks, and word_count to every chunk's metadata."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"]  = offset + i
        chunk.metadata["total_chunks"] = total
        chunk.metadata["word_count"]   = len(chunk.page_content.split())
    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────

def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    strategy: str = config.CHUNKING_STRATEGY,
) -> List[Document]:
    """
    Split a list of Documents into chunks using the chosen strategy.
    Metadata is preserved and enriched (chunk_index, total_chunks, word_count).

    Args:
        documents:     list of LangChain Document objects
        chunk_size:    max characters per chunk
        chunk_overlap: characters of overlap between adjacent chunks
        strategy:      "sentence" (NLTK) or "recursive" (character-based)
    """
    if strategy not in ("sentence", "recursive"):
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Use 'sentence' or 'recursive'.")

    chunk_fn = _sentence_chunks if strategy == "sentence" else _recursive_chunks
    all_chunks: List[Document] = []

    for doc in documents:
        raw_texts = chunk_fn(doc.page_content, chunk_size, chunk_overlap)
        doc_chunks = [
            Document(page_content=text, metadata=dict(doc.metadata))
            for text in raw_texts
            if text.strip()
        ]
        _enrich_metadata(doc_chunks, len(doc_chunks), len(all_chunks))
        all_chunks.extend(doc_chunks)

    print(
        f"[chunker] {len(documents)} documents → {len(all_chunks)} chunks "
        f"(strategy='{strategy}', size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks
