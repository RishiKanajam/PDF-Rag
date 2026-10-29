"""
run_eval.py
-----------
Reproducible evaluation of the four retrieval modes on eval/questions.jsonl.

Usage:
    python eval/run_eval.py --all              # everything below
    python eval/run_eval.py --retrieval-only   # retrieval metrics only; no LLM, no API key

What --all runs:
    1. Retrieval   — dense, bm25, hybrid, hybrid_rerank: Recall@3, Recall@5, MRR@10
                     over the 35 answerable questions (no LLM).
    2. Generation  — the same four modes at k = config.FINAL_TOP_K (3, what the app
                     ships), all 40 questions, scored by the LLM judge.
    3. Ablation    — hybrid_rerank at k = 5, reported separately.
    4. Human check — exports 15 random judged answers to eval/human_check.csv
                     (only if that file does not exist yet, so your scores are never
                     overwritten).

Reproducibility:
    - Index is built from eval/docs only, with the app's chunker and models.
    - Chunk embeddings and every LLM response (answers and judge) are cached in
      eval/cache/, so a rerun is free and returns identical results. A run that
      hits a rate limit can simply be rerun; it resumes from the cache.
    - Fixed seeds; model names and revisions come from config.py and are
      written into every results file.

Outputs:
    eval/results/<timestamp>[_<label>].jsonl  — one run_meta row, then one row per
                                                (question, mode[, k])
    eval/results/<timestamp>[_<label>].md     — summary tables
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import config
from eval.metrics import chunk_matches, first_relevant_rank, mrr_at_k, percentile, recall_at_k

EVAL_DIR     = ROOT / "eval"
DOCS_DIR     = EVAL_DIR / "docs"
CACHE_DIR    = EVAL_DIR / "cache"
RESULTS_DIR  = EVAL_DIR / "results"
HUMAN_CHECK  = EVAL_DIR / "human_check.csv"

SEED            = 42
RETRIEVAL_DEPTH = 10                       # MRR@10 needs the top 10
GENERATION_K    = config.FINAL_TOP_K       # the shipped config
ABLATION        = ("hybrid_rerank", 5)
HUMAN_CHECK_N   = 15
MODES           = ("dense", "bm25", "hybrid", "hybrid_rerank")

# Groq list prices in USD per million tokens, from the per-model pages on
# console.groq.com (checked 2026-10-30). Output tokens include reasoning
# tokens. Cached-input discounts are ignored, so cost is an upper bound.
PRICES_USD_PER_M = {
    "openai/gpt-oss-20b":  {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b": {"input": 0.15,  "output": 0.60},
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


@contextlib.contextmanager
def quiet():
    """Silence the pipeline's per-query print() logging."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def cost_usd(model: str, usage: Dict[str, int] | None) -> float:
    if not usage or model not in PRICES_USD_PER_M:
        return 0.0
    p = PRICES_USD_PER_M[model]
    return (usage.get("input_tokens", 0) * p["input"]
            + usage.get("output_tokens", 0) * p["output"]) / 1e6


def git_state() -> Dict[str, Any]:
    def run(*args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--", "src", "config.py", "eval"))}


def load_questions(limit: int | None) -> List[Dict[str, Any]]:
    questions = [json.loads(l) for l in open(EVAL_DIR / "questions.jsonl")]
    return questions[:limit] if limit else questions


class _NoLLM:
    """Placeholder LLM for --retrieval-only runs; fails loudly if ever called."""
    def invoke(self, prompt):
        raise RuntimeError("LLM called during a retrieval-only run")


# ── Index ──────────────────────────────────────────────────────────────────────

def embed_with_cache(embedder, texts: List[str]) -> np.ndarray:
    path = CACHE_DIR / f"embeddings_{embedder.model_name.replace('/', '__')}_{embedder.revision[:12]}.npz"
    cache: Dict[str, np.ndarray] = dict(np.load(path)) if path.exists() else {}
    keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    if missing:
        vecs = embedder.embed([t for _, t in missing])
        cache.update({k: v for (k, _), v in zip(missing, vecs)})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **cache)
    print(f"[eval] Embeddings: {len(texts) - len(missing)} cached, {len(missing)} computed")
    return np.stack([cache[k] for k in keys])


def build_index(pipeline) -> int:
    from src.chunker import split_documents
    from src.loader import load_pdfs

    with quiet():
        chunks = split_documents(load_pdfs(DOCS_DIR))
    texts = [c.page_content for c in chunks]
    embeddings = embed_with_cache(pipeline.embedder, texts)

    with quiet():
        pipeline.vector_store.clear_all()
        pipeline.vector_store.upsert_documents(chunks, embeddings)
        # Same construction as RAGPipeline.build()
        pipeline.bm25.build([
            {
                "id":       hashlib.md5(c.page_content.encode()).hexdigest(),
                "document": c.page_content,
                "metadata": {k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ])
    print(f"[eval] Index built: {len(chunks)} chunks from {DOCS_DIR.relative_to(ROOT)}")
    return len(chunks)


# ── Retrieval ──────────────────────────────────────────────────────────────────

def run_retrieval(pipeline, questions, modes) -> List[Dict[str, Any]]:
    rows = []
    answerable = [q for q in questions if q["type"] != "unanswerable"]
    for mode in modes:
        for q in answerable:
            start = time.perf_counter()
            with quiet():
                chunks = pipeline.retriever.retrieve(q["question"], top_k=RETRIEVAL_DEPTH, mode=mode)
            latency = time.perf_counter() - start
            relevant = [chunk_matches(c, q["gold_doc"], q["gold_quote"]) for c in chunks]
            rows.append({
                "kind": "retrieval", "mode": mode, "id": q["id"], "type": q["type"],
                "recall@3": recall_at_k(relevant, 3),
                "recall@5": recall_at_k(relevant, 5),
                "mrr@10": mrr_at_k(relevant, 10),
                "first_relevant_rank": first_relevant_rank(relevant),
                "retrieval_latency_s": round(latency, 4),
                "retrieved": [
                    {"id": c["id"], "source": os.path.basename(str(c["metadata"].get("source"))),
                     "page": c["metadata"].get("page"), "relevant": r}
                    for c, r in zip(chunks, relevant)
                ],
            })
        print(f"[eval] Retrieval done: {mode}")
    return rows


# ── Generation ─────────────────────────────────────────────────────────────────

def run_generation(pipeline, answer_llm, judge, questions, mode, k) -> List[Dict[str, Any]]:
    rows = []
    for i, q in enumerate(questions, 1):
        start = time.perf_counter()
        with quiet():
            result = pipeline.query(q["question"], top_k=k, mode=mode)
        wall = time.perf_counter() - start

        called_llm = result["usage"] is not None
        llm = answer_llm.last if called_llm else None
        # A cached answer took ~0s just now; add back the latency of the original call.
        latency = wall + (llm.latency_s if llm and llm.cached else 0.0)

        context = "\n\n---\n\n".join(c["document"] for c in result["chunks"])
        verdict = judge.score(q["question"], q["gold_answer"], context, result["answer"])

        rows.append({
            "kind": "generation", "mode": mode, "k": k, "id": q["id"], "type": q["type"],
            "question": q["question"], "gold_answer": q["gold_answer"],
            "answer": result["answer"], "context": context,
            "gold_in_context": (
                any(chunk_matches(c, q["gold_doc"], q["gold_quote"]) for c in result["chunks"])
                if q["type"] != "unanswerable" else None
            ),
            "sources": result["sources"],
            **{f"judge_{f}": v for f, v in verdict.to_dict().items() if f not in ("usage", "cached")},
            "latency_s": round(latency, 4),
            "answer_usage": llm.usage_metadata if llm else None,
            "answer_cost_usd": cost_usd(config.LLM_MODEL, llm.usage_metadata if llm else None),
            "answer_cached": llm.cached if llm else None,
            "answer_fingerprint": (llm.response_metadata or {}).get("system_fingerprint") if llm else None,
            "answer_finish_reason": (llm.response_metadata or {}).get("finish_reason") if llm else None,
            "judge_usage": verdict.usage,
            "judge_cost_usd": cost_usd(judge.model, verdict.usage),
            "judge_cached": verdict.cached,
            "judge_fingerprint": ((judge.llm.last.response_metadata or {}).get("system_fingerprint")
                                  if judge.llm.last else None),
            "judge_finish_reason": ((judge.llm.last.response_metadata or {}).get("finish_reason")
                                    if judge.llm.last else None),
        })
        if i % 10 == 0 or i == len(questions):
            print(f"[eval] Generation {mode} k={k}: {i}/{len(questions)}")
    return rows


# ── Summaries ──────────────────────────────────────────────────────────────────

def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(v, pct=False, digits=3) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.{digits}f}"


def summarize_retrieval(rows) -> List[Dict[str, Any]]:
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    out = []
    for mode in MODES:
        rs = by_mode.get(mode)
        if not rs:
            continue
        lat = [r["retrieval_latency_s"] for r in rs]
        summary = {
            "mode": mode, "n": len(rs),
            "recall@3": _mean(r["recall@3"] for r in rs),
            "recall@5": _mean(r["recall@5"] for r in rs),
            "mrr@10": _mean(r["mrr@10"] for r in rs),
            "latency_p50_s": percentile(lat, 50), "latency_p95_s": percentile(lat, 95),
        }
        for t in ("single_fact", "exact_term", "paraphrase"):
            summary[f"recall@5_{t}"] = _mean(r["recall@5"] for r in rs if r["type"] == t)
        out.append(summary)
    return out


def summarize_generation(rows) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["k"])].append(r)
    out = []
    for (mode, k), rs in groups.items():
        ok = [r for r in rs if not r["judge_parse_error"]]
        ans = [r for r in ok if r["type"] != "unanswerable"]
        unans = [r for r in ok if r["type"] == "unanswerable"]
        lat = [r["latency_s"] for r in rs]
        tokens = [(r["answer_usage"] or {}).get("total_tokens", 0) for r in rs]
        out.append({
            "mode": mode, "k": k, "n": len(rs), "judge_parse_errors": len(rs) - len(ok),
            "answers_truncated": sum(r["answer_finish_reason"] == "length" for r in rs),
            "judge_truncated": sum(r["judge_finish_reason"] == "length" for r in rs),
            "faithfulness": _mean(r["judge_faithfulness"] for r in ok),
            "answer_relevance": _mean(r["judge_answer_relevance"] for r in ok),
            "context_relevance": _mean(r["judge_context_relevance"] for r in ans),
            "correctness": _mean(r["judge_correctness"] for r in ans),
            "abstention_correct": sum(bool(r["judge_abstained"]) for r in unans),
            "abstention_n": len(unans),
            "false_abstention": _mean(float(bool(r["judge_abstained"])) for r in ans),
            "gold_in_context": _mean(float(r["gold_in_context"]) for r in rs if r["gold_in_context"] is not None),
            "latency_p50_s": percentile(lat, 50), "latency_p95_s": percentile(lat, 95),
            "tokens_per_q": _mean(tokens),
            "cost_per_q_usd": _mean(r["answer_cost_usd"] for r in rs),
            "judge_cost_total_usd": sum(r["judge_cost_usd"] for r in rs),
        })
    return out


def write_markdown(path: Path, meta, retrieval_summary, generation_summary):
    lines = [f"# Evaluation run {meta['run_id']}", ""]
    lines += [
        f"- Questions: {meta['n_questions']} ({meta['n_answerable']} answerable, "
        f"{meta['n_questions'] - meta['n_answerable']} unanswerable); "
        f"documents: {', '.join(meta['documents'])} → {meta['n_chunks']} chunks",
        f"- Embedder: `{meta['models']['embedding']}`; reranker: `{meta['models']['reranker']}`",
        f"- BM25 tokenizer: `{meta['bm25_tokenizer']}`",
        f"- Git: `{meta['git']['commit'][:10]}`{' (dirty)' if meta['git']['dirty'] else ''}",
    ]
    if generation_summary:
        lines += [
            f"- Answer LLM: `{meta['models']['answer_llm']}` ({meta['models']['answer_llm_settings']}); "
            f"judge: `{meta['models']['judge']}` ({meta['models']['judge_settings']}, "
            f"prompt sha256 `{meta['judge_prompt_sha256'][:12]}`)",
            f"- Groq system fingerprints — answer: {meta['fingerprints']['answer'] or '—'}; "
            f"judge: {meta['fingerprints']['judge'] or '—'}",
        ]
    lines += ["", f"## Retrieval ({meta['n_answerable']} answerable questions, no LLM)", "",
              "| Mode | Recall@3 | Recall@5 | MRR@10 | R@5 single-fact | R@5 exact-term | R@5 paraphrase | Latency p50 / p95 |",
              "|---|---|---|---|---|---|---|---|"]
    for s in retrieval_summary:
        lines.append(
            f"| {s['mode']} | {_fmt(s['recall@3'], pct=True)} | {_fmt(s['recall@5'], pct=True)} | "
            f"{_fmt(s['mrr@10'])} | {_fmt(s['recall@5_single_fact'], pct=True)} | "
            f"{_fmt(s['recall@5_exact_term'], pct=True)} | {_fmt(s['recall@5_paraphrase'], pct=True)} | "
            f"{s['latency_p50_s'] * 1000:.0f} / {s['latency_p95_s'] * 1000:.0f} ms |"
        )

    def gen_table(rows):
        out = ["| Mode | k | Faithfulness | Answer relevance | Context relevance | Correctness | "
               "Abstention (unanswerable) | False abstention (answerable) | Gold in context | "
               "Latency p50 / p95 | Tokens / q | Cost / q |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in rows:
            out.append(
                f"| {s['mode']} | {s['k']} | {_fmt(s['faithfulness'], digits=2)} | "
                f"{_fmt(s['answer_relevance'], digits=2)} | {_fmt(s['context_relevance'], digits=2)} | "
                f"{_fmt(s['correctness'], digits=2)} | {s['abstention_correct']}/{s['abstention_n']} | "
                f"{_fmt(s['false_abstention'], pct=True)} | {_fmt(s['gold_in_context'], pct=True)} | "
                f"{s['latency_p50_s']:.2f} / {s['latency_p95_s']:.2f} s | {s['tokens_per_q']:.0f} | "
                f"${s['cost_per_q_usd']:.6f} |"
            )
        return out

    if generation_summary:
        main = [s for s in generation_summary if s["k"] == GENERATION_K]
        ablation = [s for s in generation_summary if (s["mode"], s["k"]) == ABLATION]
        lines += ["", f"## Generation at k={GENERATION_K} (shipped config), LLM-as-judge, all questions", ""]
        lines += gen_table(main)
        lines += ["", f"## Ablation: {ABLATION[0]} at k={ABLATION[1]}", ""]
        lines += gen_table(ablation)
        errors = sum(s["judge_parse_errors"] for s in generation_summary)
        truncated = sum(s["answers_truncated"] for s in generation_summary)
        judge_truncated = sum(s["judge_truncated"] for s in generation_summary)
        judge_cost = sum(s["judge_cost_total_usd"] for s in generation_summary)
        lines += [
            "",
            "Faithfulness and answer relevance are averaged over all questions; context relevance, "
            "correctness, false abstention and gold-in-context over answerable questions only. "
            "Scores are on a 0 / 0.5 / 1 scale (see `eval/judge_prompt.md`). "
            f"Judge parse errors (excluded): {errors}. "
            f"Responses cut off at max_tokens — answers: {truncated}, judge: {judge_truncated}. "
            f"Latency is retrieval + answer generation end to end; for cached answers the "
            f"original API latency is used. Cost is approximate (see `PRICES_USD_PER_M`); "
            f"total judge cost for this run ≈ ${judge_cost:.4f}.",
        ]
    path.write_text("\n".join(lines) + "\n")


# ── Human check export ─────────────────────────────────────────────────────────

def export_human_check(generation_rows, n=HUMAN_CHECK_N, seed=SEED) -> bool:
    if HUMAN_CHECK.exists():
        print(f"[eval] {HUMAN_CHECK.relative_to(ROOT)} exists — not overwriting your scores.")
        return False
    pool, seen = [], set()
    for r in generation_rows:
        key = (r["id"], r["answer"])
        if r["k"] == GENERATION_K and not r["judge_parse_error"] and key not in seen:
            seen.add(key)
            pool.append(r)
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    fields = ["item", "id", "mode", "type", "question", "gold_answer", "context", "answer",
              "judge_faithfulness", "judge_answer_relevance", "judge_correctness",
              "human_faithfulness", "human_answer_relevance", "human_correctness", "notes"]
    with open(HUMAN_CHECK, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow({"item": i, **{k: r[k] for k in ("id", "mode", "type", "question",
                                                       "gold_answer", "context", "answer",
                                                       "judge_faithfulness", "judge_answer_relevance",
                                                       "judge_correctness")}})
    print(f"[eval] Wrote {len(sample)} items to {HUMAN_CHECK.relative_to(ROOT)}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="retrieval + generation + ablation + human-check export")
    g.add_argument("--retrieval-only", action="store_true", help="retrieval metrics only (no API key needed)")
    p.add_argument("--label", default="", help="suffix for the results filename")
    p.add_argument("--judge-provider", default=config.JUDGE_PROVIDER)
    p.add_argument("--judge-model", default=config.JUDGE_MODEL)
    p.add_argument("--limit", type=int, help="only the first N questions (debugging)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    generation = args.all

    if generation and not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is not set. Generation and judging need it "
              "(export it, or put it in .env). Use --retrieval-only to run without it.")
        return 2

    set_seeds()
    questions = load_questions(args.limit)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + (f"_{args.label}" if args.label else "")

    from src.bm25 import BM25Index
    from src.rag import RAGPipeline
    from src.vectorstore import VectorStore

    answer_llm, judge = _NoLLM(), None
    if generation:
        from langchain_groq import ChatGroq
        from eval.judge import make_judge
        from eval.llm_cache import CachedChatModel

        answer_llm = CachedChatModel(
            ChatGroq(api_key=os.environ["GROQ_API_KEY"], model_name=config.LLM_MODEL,
                     temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS,
                     reasoning_effort=config.LLM_REASONING_EFFORT, max_retries=6),
            model_name=f"groq/{config.LLM_MODEL}",
            params={"temperature": config.LLM_TEMPERATURE, "max_tokens": config.LLM_MAX_TOKENS,
                    "reasoning_effort": config.LLM_REASONING_EFFORT},
            cache_path=CACHE_DIR / "llm_cache.sqlite",
        )
        judge = make_judge(args.judge_provider, args.judge_model, CACHE_DIR)

    with quiet():
        pipeline = RAGPipeline(
            vector_store=VectorStore(collection_name="eval_documents",
                                     persist_directory=str(CACHE_DIR / "chroma")),
            bm25=BM25Index(),
            llm=answer_llm,
        )
    n_chunks = build_index(pipeline)

    retrieval_rows = run_retrieval(pipeline, questions, MODES)

    generation_rows: List[Dict[str, Any]] = []
    if generation:
        try:
            for mode in MODES:
                generation_rows += run_generation(pipeline, answer_llm, judge, questions, mode, GENERATION_K)
            generation_rows += run_generation(pipeline, answer_llm, judge, questions, *ABLATION)
        except Exception as e:
            print(f"\nERROR during generation: {type(e).__name__}: {e}\n"
                  "Every response so far is cached in eval/cache/; rerun the same command to resume.")
            return 1
        print(f"[eval] LLM cache — answers: {answer_llm.hits} hits / {answer_llm.misses} calls; "
              f"judge: {judge.llm.hits} hits / {judge.llm.misses} calls")

    meta = {
        "kind": "run_meta", "run_id": run_id,
        "n_questions": len(questions),
        "n_answerable": sum(q["type"] != "unanswerable" for q in questions),
        "documents": sorted(p.name for p in DOCS_DIR.glob("*.pdf")),
        "n_chunks": n_chunks,
        "chunking": {"strategy": config.CHUNKING_STRATEGY, "size": config.CHUNK_SIZE,
                     "overlap": config.CHUNK_OVERLAP},
        "retrieval": {"hybrid_top_k": config.HYBRID_TOP_K, "rrf_k": config.RRF_K,
                      "depth": RETRIEVAL_DEPTH, "generation_k": GENERATION_K, "ablation": ABLATION},
        "bm25_tokenizer": getattr(BM25Index, "TOKENIZER_VERSION", "whitespace-v0"),
        "models": {
            "embedding": f"{config.EMBEDDING_MODEL}@{config.EMBEDDING_MODEL_REVISION}",
            "reranker": f"{config.RERANKER_MODEL}@{config.RERANKER_MODEL_REVISION}",
            "answer_llm": f"groq/{config.LLM_MODEL}" if generation else None,
            "answer_llm_settings": {"temperature": config.LLM_TEMPERATURE,
                                    "max_tokens": config.LLM_MAX_TOKENS,
                                    "reasoning_effort": config.LLM_REASONING_EFFORT} if generation else None,
            "judge": f"{args.judge_provider}/{args.judge_model}" if generation else None,
            "judge_settings": {"temperature": 0, "max_tokens": config.JUDGE_MAX_TOKENS,
                               "reasoning_effort": config.JUDGE_REASONING_EFFORT} if generation else None,
        },
        "judge_prompt_sha256": hashlib.sha256((EVAL_DIR / "judge_prompt.md").read_bytes()).hexdigest(),
        "fingerprints": {
            "answer": sorted({r["answer_fingerprint"] for r in generation_rows if r["answer_fingerprint"]}),
            "judge": sorted({r["judge_fingerprint"] for r in generation_rows if r["judge_fingerprint"]}),
        },
        "seed": SEED, "prices_usd_per_m": PRICES_USD_PER_M if generation else None,
        "git": git_state(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS_DIR / f"{run_id}.jsonl"
    with open(jsonl_path, "w") as f:
        for row in [meta, *retrieval_rows, *generation_rows]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    retrieval_summary = summarize_retrieval(retrieval_rows)
    generation_summary = summarize_generation(generation_rows) if generation_rows else []
    md_path = RESULTS_DIR / f"{run_id}.md"
    write_markdown(md_path, meta, retrieval_summary, generation_summary)

    if generation_rows:
        export_human_check(generation_rows)

    print(f"\n[eval] Raw results: {jsonl_path.relative_to(ROOT)}")
    print(f"[eval] Summary:     {md_path.relative_to(ROOT)}\n")
    print(md_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
