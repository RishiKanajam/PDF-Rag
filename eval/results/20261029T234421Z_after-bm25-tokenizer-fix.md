# Evaluation run 20261029T234421Z_after-bm25-tokenizer-fix

- Questions: 40 (35 answerable, 5 unanswerable); documents: NIST.AI.100-1.pdf, NIST.CSWP.29.pdf → 277 chunks
- Embedder: `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a`
- BM25 tokenizer: `regex-v1`
- Git: `e9e3b9b551`

## Retrieval (35 answerable questions, no LLM)

| Mode | Recall@3 | Recall@5 | MRR@10 | R@5 single-fact | R@5 exact-term | R@5 paraphrase | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| dense | 51.4% | 60.0% | 0.493 | 73.3% | 50.0% | 50.0% | 6 / 12 ms |
| bm25 | 80.0% | 88.6% | 0.760 | 93.3% | 100.0% | 70.0% | 0 / 0 ms |
| hybrid | 68.6% | 82.9% | 0.594 | 86.7% | 100.0% | 60.0% | 6 / 8 ms |
| hybrid_rerank | 82.9% | 91.4% | 0.755 | 93.3% | 100.0% | 80.0% | 58 / 106 ms |
