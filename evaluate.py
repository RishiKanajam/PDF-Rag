"""
evaluate.py
-----------
Evaluates the RAG pipeline quality using an LLM-as-judge approach.

Usage:
    python evaluate.py                    # run all questions
    python evaluate.py --id 1 3 5         # run specific question IDs
    python evaluate.py --save             # also save results to Data/eval_results.json

What you learn here:

WHY EVALUATION MATTERS:
  Without measurement, you're guessing. You might tune your chunking strategy
  or change the reranker model and *think* things improved — but without scores
  you have no evidence. Evaluation closes the feedback loop.

LLM-AS-JUDGE:
  We ask the same LLM to evaluate its own answers using a structured rubric.
  This sounds circular, but it works surprisingly well in practice — LLMs are
  good at following scoring instructions, and the scores correlate strongly with
  human judgements. This is also how RAGAS, G-Eval, and other popular
  evaluation frameworks work internally.

THE THREE METRICS WE MEASURE:

  1. FAITHFULNESS (0-5): Does the answer only use information from the
     retrieved chunks? A score of 5 means every claim in the answer can be
     traced to the context. A score of 1 means the LLM hallucinated.
     → Detects: hallucination, the LLM "making things up"

  2. ANSWER RELEVANCY (0-5): Does the answer actually address the question?
     A score of 5 means the answer is directly on-topic. A score of 1 means
     the answer went off on a tangent.
     → Detects: vague answers, topic drift

  3. CONTEXT QUALITY (0-5): Are the retrieved chunks relevant to the question?
     A score of 5 means all retrieved chunks are on-topic. A score of 1 means
     retrieval found completely unrelated passages.
     → Detects: retrieval failures, embedding model weaknesses

INTERPRETING RESULTS:
  - All scores 4+: pipeline is working well
  - Low faithfulness: LLM is hallucinating — try a stricter prompt
  - Low relevancy:   answers are vague — check LLM temperature / max_tokens
  - Low context:     retrieval is broken — check chunking or embedding model
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import config

config.validate_config()

from src.rag import RAGPipeline


# ── Scoring prompts ────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Score the faithfulness of the answer.

FAITHFULNESS means: does the answer use ONLY information present in the context?
A faithful answer does not add outside knowledge or contradict the context.

Context:
{context}

Question: {question}

Answer: {answer}

Score the faithfulness on a scale of 1 to 5:
5 = Every claim in the answer is directly supported by the context
4 = Almost all claims are from context; one minor addition
3 = Some claims are from context, some from outside knowledge
2 = The answer barely uses the context
1 = The answer ignores or contradicts the context

Reply in this exact format:
SCORE: <number>
REASON: <one sentence>"""


_RELEVANCY_PROMPT = """\
You are an evaluation judge. Score how well the answer addresses the question.

Question: {question}

Answer: {answer}

Score the answer relevancy on a scale of 1 to 5:
5 = The answer directly and completely addresses the question
4 = The answer mostly addresses the question with minor gaps
3 = The answer partially addresses the question
2 = The answer barely addresses the question
1 = The answer does not address the question at all

Reply in this exact format:
SCORE: <number>
REASON: <one sentence>"""


_CONTEXT_QUALITY_PROMPT = """\
You are an evaluation judge. Score how relevant the retrieved context is.

Question: {question}

Retrieved context:
{context}

Score the context quality on a scale of 1 to 5:
5 = All retrieved passages are highly relevant to the question
4 = Most passages are relevant; one is tangential
3 = About half the passages are relevant
2 = Only one passage is relevant; the rest are off-topic
1 = No retrieved passages are relevant to the question

Reply in this exact format:
SCORE: <number>
REASON: <one sentence>"""


# ── Score parsing ──────────────────────────────────────────────────────────────

def _parse_score(response_text: str) -> tuple[float, str]:
    """Extract SCORE and REASON from LLM judge response."""
    score_match = re.search(r"SCORE:\s*([1-5])", response_text)
    reason_match = re.search(r"REASON:\s*(.+)", response_text)
    score  = float(score_match.group(1)) if score_match else 0.0
    reason = reason_match.group(1).strip() if reason_match else "Could not parse reason"
    return score, reason


# ── Evaluator ─────────────────────────────────────────────────────────────────

class RAGEvaluator:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.llm      = pipeline.llm  # reuse the same LLM

    def _judge(self, prompt: str) -> tuple[float, str]:
        response = self.llm.invoke(prompt)
        return _parse_score(response.content)

    def evaluate_one(self, question: str) -> Dict[str, Any]:
        """
        Run a single question through the pipeline and score it on 3 metrics.

        Returns a dict with:
            question, answer, context_preview,
            faithfulness, faithfulness_reason,
            relevancy, relevancy_reason,
            context_quality, context_quality_reason,
            avg_score, num_chunks_retrieved
        """
        # ── Run the pipeline ───────────────────────────────────────────────────
        result = self.pipeline.query(question)
        answer = result["answer"]
        chunks = result["chunks"]

        if not chunks:
            return {
                "question":   question,
                "answer":     answer,
                "error":      "No chunks retrieved",
                "faithfulness":    0.0,
                "relevancy":       0.0,
                "context_quality": 0.0,
                "avg_score":       0.0,
                "num_chunks":      0,
            }

        context = "\n\n---\n\n".join(c["document"] for c in chunks)

        # ── Score on 3 dimensions ──────────────────────────────────────────────
        faith_score, faith_reason = self._judge(
            _FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
        )
        relev_score, relev_reason = self._judge(
            _RELEVANCY_PROMPT.format(question=question, answer=answer)
        )
        ctx_score, ctx_reason = self._judge(
            _CONTEXT_QUALITY_PROMPT.format(question=question, context=context)
        )

        avg = round((faith_score + relev_score + ctx_score) / 3, 2)

        return {
            "question":               question,
            "answer":                 answer,
            "context_preview":        context[:300] + "..." if len(context) > 300 else context,
            "faithfulness":           faith_score,
            "faithfulness_reason":    faith_reason,
            "relevancy":              relev_score,
            "relevancy_reason":       relev_reason,
            "context_quality":        ctx_score,
            "context_quality_reason": ctx_reason,
            "avg_score":              avg,
            "num_chunks":             len(chunks),
        }

    def evaluate_all(
        self, questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results = []
        total = len(questions)
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{total}] {q['question'][:70]}...")
            result = self.evaluate_one(q["question"])
            result["id"]    = q.get("id", i)
            result["topic"] = q.get("topic", "")
            results.append(result)
            print(
                f"       Faithfulness={result['faithfulness']:.0f}  "
                f"Relevancy={result['relevancy']:.0f}  "
                f"Context={result['context_quality']:.0f}  "
                f"Avg={result['avg_score']}"
            )
        return results


# ── Report printing ────────────────────────────────────────────────────────────

def print_report(results: List[Dict[str, Any]]):
    """Print a formatted evaluation report to the terminal."""
    print("\n" + "=" * 70)
    print("  RAG EVALUATION REPORT")
    print("=" * 70)

    header = f"{'#':>3}  {'Question':<42}  {'Faith':>5}  {'Relev':>5}  {'Ctx':>5}  {'Avg':>5}"
    print(header)
    print("-" * 70)

    for r in results:
        q_short = r["question"][:40] + ".." if len(r["question"]) > 40 else r["question"]
        error = " ⚠ no chunks" if r.get("error") else ""
        print(
            f"{r.get('id', '?'):>3}  {q_short:<42}  "
            f"{r['faithfulness']:>5.1f}  "
            f"{r['relevancy']:>5.1f}  "
            f"{r['context_quality']:>5.1f}  "
            f"{r['avg_score']:>5.2f}"
            f"{error}"
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("-" * 70)
    valid = [r for r in results if not r.get("error")]
    if valid:
        avg_faith = sum(r["faithfulness"]    for r in valid) / len(valid)
        avg_relev = sum(r["relevancy"]       for r in valid) / len(valid)
        avg_ctx   = sum(r["context_quality"] for r in valid) / len(valid)
        avg_all   = sum(r["avg_score"]       for r in valid) / len(valid)
        print(
            f"{'AVERAGE':<47}  "
            f"{avg_faith:>5.2f}  "
            f"{avg_relev:>5.2f}  "
            f"{avg_ctx:>5.2f}  "
            f"{avg_all:>5.2f}"
        )

    print("\nScore guide: 5=excellent  4=good  3=ok  2=poor  1=failing")
    print("\nWhat low scores mean:")
    print("  Faithfulness < 3  → LLM is hallucinating; tighten the prompt")
    print("  Relevancy < 3     → Answers are vague; check LLM temperature")
    print("  Context < 3       → Retrieval is broken; check chunking/embeddings")
    print("=" * 70)

    # Per-question details
    print("\n── Per-question details ──────────────────────────────────────────────")
    for r in results:
        if r.get("error"):
            continue
        print(f"\nQ{r.get('id','?')}: {r['question']}")
        print(f"  Faithfulness ({r['faithfulness']:.0f}/5): {r['faithfulness_reason']}")
        print(f"  Relevancy    ({r['relevancy']:.0f}/5): {r['relevancy_reason']}")
        print(f"  Context      ({r['context_quality']:.0f}/5): {r['context_quality_reason']}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline using LLM-as-judge scoring"
    )
    parser.add_argument(
        "--id",
        nargs="+",
        type=int,
        metavar="N",
        help="Run only specific question IDs (e.g. --id 1 3 5)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to Data/eval_results.json",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the index before evaluating",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load question set
    dataset_path = Path("Data/eval_questions.json")
    if not dataset_path.exists():
        print(f"ERROR: Eval dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path) as f:
        all_questions = json.load(f)

    # Filter by ID if requested
    if args.id:
        questions = [q for q in all_questions if q["id"] in args.id]
        if not questions:
            print(f"ERROR: No questions found with IDs {args.id}")
            sys.exit(1)
    else:
        questions = all_questions

    print(f"[eval] Running evaluation on {len(questions)} question(s)...")

    # Build pipeline
    pipeline = RAGPipeline()
    pipeline.build(force=args.rebuild)

    # Run evaluation
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.evaluate_all(questions)

    # Print report
    print_report(results)

    # Save results
    if args.save:
        out_path = Path("Data/eval_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval] Results saved to {out_path}")


if __name__ == "__main__":
    main()
