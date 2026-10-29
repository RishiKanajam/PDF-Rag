# Evaluation documents

| File | Title | URL | Licence | SHA-256 |
|---|---|---|---|---|
| `docs/NIST.AI.100-1.pdf` | Artificial Intelligence Risk Management Framework (AI RMF 1.0), NIST AI 100-1, January 2023 (48 pages) | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf | Public domain: a work of the US federal government, not subject to copyright in the United States (17 U.S.C. § 105) | `7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1` |
| `docs/NIST.CSWP.29.pdf` | The NIST Cybersecurity Framework (CSF) 2.0, NIST CSWP 29, February 26, 2024 (32 pages) | https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf | Public domain: a work of the US federal government, not subject to copyright in the United States (17 U.S.C. § 105) | `3c31f46fee98cac0c4323453e5109291a213b4de7fef8c058af9bf67f717433c` |

Downloaded 2026-10-30.

Why these two documents: both are full of short identifiers (`GOVERN 1.6`, `ID.AM-01`, `PR.DS-11`), which exact-term questions need. Both also contain explanatory prose that can be asked about in paraphrase. Both cover risk management, so they share vocabulary, which makes retrieval harder than it would be with unrelated documents.

`gold_page` in `questions.jsonl` is the **0-based** page index that `PyPDFLoader` stores in chunk metadata, not the page number printed on the page.

To use your own PDFs instead, replace the files in `docs/`, rewrite `questions.jsonl`, and run `python eval/verify_questions.py`.
