# Evaluation run 20261029T234325Z_before-bm25-tokenizer-fix

- Questions: 40 (35 answerable, 5 unanswerable); documents: NIST.AI.100-1.pdf, NIST.CSWP.29.pdf → 277 chunks
- Embedder: `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a`
- BM25 tokenizer: `whitespace-v0`
- Git: `d7aa77503f`

## Retrieval (35 answerable questions, no LLM)

| Mode | Recall@3 | Recall@5 | MRR@10 | R@5 single-fact | R@5 exact-term | R@5 paraphrase | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| dense | 51.4% | 60.0% | 0.493 | 73.3% | 50.0% | 50.0% | 7 / 11 ms |
| bm25 | 65.7% | 68.6% | 0.612 | 86.7% | 50.0% | 60.0% | 0 / 1 ms |
| hybrid | 68.6% | 71.4% | 0.557 | 86.7% | 50.0% | 70.0% | 7 / 9 ms |
| hybrid_rerank | 82.9% | 88.6% | 0.765 | 93.3% | 80.0% | 90.0% | 50 / 106 ms |
