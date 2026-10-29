# PDFRag

A production-quality **Retrieval-Augmented Generation (RAG)** system that lets you chat with your PDF and text documents using AI.

Built from scratch with a full learning-first approach — every module has inline explanations of the concepts behind it.

---

## Features

- **Hybrid Search** — BM25 keyword search + semantic vector search, merged with Reciprocal Rank Fusion (RRF)
- **Cross-encoder Reranking** — a second-stage reranker scores (query, chunk) pairs together for high accuracy
- **Sentence-aware Chunking** — NLTK-based splitting that never cuts mid-sentence, with rich metadata per chunk
- **Conversational Memory** — multi-turn chat with question reformulation so follow-ups like "tell me more" work correctly
- **Reproducible evaluation** — 40-question test set, Recall@k / MRR for four retrieval modes, and an LLM judge scoring faithfulness, answer relevance, context relevance and correctness ([Evaluation](#evaluation))
- **Streamlit UI** — upload PDFs, chat, see sources, rebuild index — all in a web interface
- **CLI** — `python main.py` for terminal-based chat

---

## Architecture

```
PDFRag/
├── app.py              ← Streamlit web UI
├── main.py             ← CLI entry point
├── evaluate.py         ← quick LLM-judge check on your own documents (Data/)
├── config.py           ← all settings and pinned model versions
├── eval/               ← reproducible evaluation (docs, questions, run_eval.py, judge, results)
├── tests/              ← unit tests (run in CI on every push)
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

**Evaluate quality:** see [Evaluation](#evaluation) below.
```bash
python eval/run_eval.py --all   # full, reproducible evaluation
```

**Run the unit tests:**
```bash
pip install pytest && python -m pytest
```

---

## Configuration

All settings are in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `RETRIEVAL_MODE` | `"hybrid_rerank"` | `dense`, `bm25`, `hybrid` (BM25 + dense, RRF) or `hybrid_rerank` (hybrid + cross-encoder) |
| `CHUNKING_STRATEGY` | `"sentence"` | `"sentence"` (NLTK) or `"recursive"` (char-based) |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `HYBRID_TOP_K` | `10` | Candidates each search fetches before fusion/reranking |
| `FINAL_TOP_K` | `3` | Chunks sent to the LLM |
| `MEMORY_MAX_TURNS` | `5` | Conversation turns kept in context |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedder, pinned to a Hugging Face commit |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker, pinned to a Hugging Face commit |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq LLM that answers questions |
| `JUDGE_MODEL` | `llama-3.3-70b-versatile` | Groq LLM that grades answers in the evaluation |

---

## Evaluation

### Dataset

- **Documents** ([eval/SOURCES.md](eval/SOURCES.md)): NIST AI Risk Management Framework 1.0 (48 pages) and NIST Cybersecurity Framework 2.0 (32 pages). Both are US-government works in the public domain. They split into 277 chunks.
- **Questions** ([eval/questions.jsonl](eval/questions.jsonl)): 40 hand-written questions. Each has a gold answer, document, page, and an exact supporting quote.

  | Type | Count | Example |
  |---|---|---|
  | Single fact | 15 | *What are the four CSF Tiers?* |
  | Exact term (identifiers) | 10 | *What is CSF subcategory ID.AM-01?* |
  | Paraphrased | 10 | *Can protecting people's privacy make an AI model less accurate?* |
  | Unanswerable | 5 | *Which cloud provider hosts the CSF 2.0 Reference Tool?* |

- Every gold quote is checked against the extracted PDF text by `python eval/verify_questions.py` (35/35 pass).

### Metrics

- **Retrieval** (no LLM, 35 answerable questions):
  - A retrieved chunk counts as a hit if it comes from the gold document and contains the gold quote. Matching ignores spacing and punctuation, or accepts a ≥80% contiguous overlap.
  - Reported as Recall@3, Recall@5 and MRR@10.
- **Generation:** the app's own `RAGPipeline.query()` answers at `k = 3`, the shipped setting. `llama-3.3-70b-versatile` then grades each answer at temperature 0, using the prompt in [eval/judge_prompt.md](eval/judge_prompt.md). It scores:
  - faithfulness to the retrieved context
  - answer relevance
  - context relevance
  - correctness against the gold answer
  - whether the system abstained ("I don't know")
- **Operational:** latency p50/p95, tokens per question, and approximate cost per question.

The judge (70B) is a different and larger model than the answering model (8B). Judge providers sit behind one interface ([eval/judge.py](eval/judge.py)), so another provider can be added and selected with `--judge-provider`.

### Reproduce

```bash
export GROQ_API_KEY=...                   # read from the environment / .env only
python eval/run_eval.py --all             # retrieval + generation + k=5 ablation
python eval/run_eval.py --retrieval-only  # no API key needed
```

- **Outputs:** each run writes raw per-question rows to `eval/results/<timestamp>.jsonl` and a summary to `eval/results/<timestamp>.md`. Both record the model names, Hugging Face revisions, Groq system fingerprints, judge-prompt hash and git commit.
- **Caching:** embeddings and every LLM response are cached in `eval/cache/`, so reruns are free and identical. A run interrupted by a rate limit resumes from the cache.
- **Not cached across machines:** the answering model runs at the app's temperature (0.1), so a fresh machine without the cache will not reproduce answers byte for byte.

### Results

**Retrieval** (35 answerable questions), from [the run after the BM25 tokenizer fix](eval/results/20261029T234421Z_after-bm25-tokenizer-fix.md):

| Mode | Recall@3 | Recall@5 | MRR@10 | R@5 single-fact | R@5 exact-term | R@5 paraphrase |
|---|---|---|---|---|---|---|
| dense | 51.4% | 60.0% | 0.493 | 73.3% | 50.0% | 50.0% |
| bm25 | 80.0% | 88.6% | 0.760 | 93.3% | 100.0% | 70.0% |
| hybrid (RRF) | 68.6% | 82.9% | 0.594 | 86.7% | 100.0% | 60.0% |
| hybrid + rerank | 82.9% | 91.4% | 0.755 | 93.3% | 100.0% | 80.0% |

What this shows:

- **Hybrid with re-ranking has the best Recall@5** (91.4% vs 60.0% for dense alone).
- **Plain RRF fusion without the re-ranker is worse than BM25 alone** on every retrieval metric here.
- **BM25 has a slightly higher MRR@10 than hybrid + rerank** (0.760 vs 0.755).
- **Dense embeddings are the weakest retriever on these documents**, including on paraphrased questions, where dense search was expected to help most.

**BM25 tokenizer fix.** The original tokenizer split on whitespace only, so `ID.AM-01:` in a document never matched `ID.AM-01?` in a question. The fix keeps identifiers whole and strips surrounding punctuation. Effect on Recall@5, measured before and after on the same index ([before](eval/results/20261029T234325Z_before-bm25-tokenizer-fix.md) / [after](eval/results/20261029T234421Z_after-bm25-tokenizer-fix.md)):

| Mode | Before | After |
|---|---|---|
| bm25 | 68.6% (exact-term 50%) | 88.6% (exact-term 100%) |
| hybrid | 71.4% | 82.9% |
| hybrid + rerank | 88.6% | 91.4% |

The fix did not help everywhere: hybrid + rerank paraphrase Recall@5 fell from 90% to 80% (one question), because different BM25 candidates entered the fused list.

**Generation (LLM-as-judge):** not run yet. It needs `GROQ_API_KEY`. No generation numbers are reported until a real run produces them.

**Judge–human agreement:** pending. `eval/human_check.csv` (15 random judged answers) is created by the first `--all` run. Once the human columns are filled in, `python eval/agreement.py` reports exact agreement, Cohen's κ and mean absolute difference.

### Limitations

- **Small test set.** With 35 answerable questions, one question moves Recall by 2.9 points, and a per-type figure (10–15 questions) by 7–10 points. Differences of a few points are within noise.
- **Two documents, one author.** Both are NIST risk-management frameworks with formal, repetitive wording and many identifiers. That favours keyword search, and the results may not transfer to other kinds of documents.
- **Questions written by reading the documents.** The questions were written from the source text, so their wording tends to echo it. That also favours BM25. The paraphrased set exists to offset this, but it is small.
- **LLM judge bias.** The judge is a Llama model grading a Llama model. It is told to score faithfulness against the retrieved context only, but it has known biases (e.g. towards longer answers). It is only as trustworthy as its agreement with human scores.
- **Single-turn only.** The evaluation does not measure conversational memory or question reformulation.
- **Approximate costs.** Costs use Groq's last published list prices, which Groq no longer shows on its models page. Token counts are exact.

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
