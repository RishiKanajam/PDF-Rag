# PDFRag

A production-quality **Retrieval-Augmented Generation (RAG)** system that lets you chat with your PDF and text documents using AI.

Built from scratch with a full learning-first approach — every module has inline explanations of the concepts behind it.

---

## Features

- **Hybrid Search** — BM25 keyword search + semantic vector search, merged with Reciprocal Rank Fusion (RRF)
- **Cross-encoder Reranking** — a second-stage reranker scores (query, chunk) pairs together for high accuracy
- **Sentence-aware Chunking** — NLTK-based splitting that never cuts mid-sentence, with rich metadata per chunk
- **Conversational Memory** — multi-turn chat with question reformulation so follow-ups like "tell me more" work correctly
- **LLM-as-judge Evaluation** — built-in evaluation script scoring Faithfulness, Relevancy, and Context Quality
- **Streamlit UI** — upload PDFs, chat, see sources, rebuild index — all in a web interface
- **CLI** — `python main.py` for terminal-based chat

---

## Architecture

```
PDFRag/
├── app.py              ← Streamlit web UI
├── main.py             ← CLI entry point
├── evaluate.py         ← RAG quality evaluation
├── config.py           ← all settings in one place
├── src/
│   ├── loader.py       ← PDF + text file loading
│   ├── chunker.py      ← sentence-aware chunking (NLTK)
│   ├── embedder.py     ← local embeddings (all-MiniLM-L6-v2)
│   ├── vectorstore.py  ← ChromaDB persistent vector store
│   ├── bm25.py         ← BM25 keyword index
│   ├── reranker.py     ← cross-encoder reranker
│   ├── retriever.py    ← hybrid RRF retrieval pipeline
│   ├── memory.py       ← conversation memory
│   └── rag.py          ← orchestrates the full pipeline
└── Data/
    ├── pdfs/           ← put your PDFs here
    └── text_files/     ← put your .txt files here
```

---

## Retrieval Pipeline

```
User query
  ├─ BM25 search        → top 10 chunks  ─┐
  ├─ Semantic search    → top 10 chunks  ─┤ Reciprocal Rank Fusion
  │                                        ↓
  │                                   merged top 20
  │                                        ↓
  │                              Cross-encoder Reranker
  └─────────────────────────────────── final top 3 → LLM → Answer
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/PDFRag.git
cd PDFRag
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

### 3. Add your documents

Drop PDFs into `Data/pdfs/` and/or text files into `Data/text_files/`.

### 4. Run

**Web UI (recommended):**
```bash
streamlit run app.py
```

**CLI:**
```bash
python main.py --rebuild    # first run, builds the index
python main.py              # subsequent runs
```

**Evaluate quality:**
```bash
python evaluate.py --rebuild    # run all 8 test questions
python evaluate.py --id 1 2 3  # run specific questions
```

---

## Configuration

All settings are in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `CHUNKING_STRATEGY` | `"sentence"` | `"sentence"` (NLTK) or `"recursive"` (char-based) |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `HYBRID_TOP_K` | `10` | Candidates each search fetches before reranking |
| `FINAL_TOP_K` | `3` | Chunks sent to the LLM after reranking |
| `MEMORY_MAX_TURNS` | `5` | Conversation turns kept in context |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq LLM |

---

## Tech Stack

| Component | Library |
|---|---|
| Document loading | LangChain Community |
| Text splitting | NLTK + LangChain |
| Embeddings | sentence-transformers (local) |
| Vector store | ChromaDB (local, persistent) |
| Keyword search | rank-bm25 |
| Reranker | sentence-transformers CrossEncoder |
| LLM | Groq (llama-3.1-8b-instant) |
| Web UI | Streamlit |

---

## License

MIT
