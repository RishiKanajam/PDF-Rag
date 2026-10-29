"""
agreement.py
------------
How often does the LLM judge agree with a human?

1. `python eval/run_eval.py --all` writes eval/human_check.csv: 15 randomly
   sampled judged answers (seed 42).
2. Fill in human_faithfulness, human_answer_relevance and human_correctness
   for each row using the same 0 / 0.5 / 1 rubric as eval/judge_prompt.md.
   Leave a cell blank to skip it.
3. `python eval/agreement.py`

Reports, per metric: number of scored items, exact agreement, Cohen's kappa
(agreement corrected for chance), and mean absolute difference.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import cohens_kappa

CSV_PATH = Path(__file__).resolve().parent / "human_check.csv"
METRICS = ("faithfulness", "answer_relevance", "correctness")


def _score(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    score = float(value)
    if score not in (0.0, 0.5, 1.0):
        raise ValueError(f"scores must be 0, 0.5 or 1, got {value!r}")
    return score


def main(path: Path = CSV_PATH) -> int:
    if not path.exists():
        print(f"{path} not found — run `python eval/run_eval.py --all` first.")
        return 1
    rows = list(csv.DictReader(open(path)))

    print(f"Judge vs human on {path.name} ({len(rows)} items)\n")
    print(f"{'Metric':<18} {'n':>3}  {'Exact agree':>11}  {'Cohen κ':>8}  {'Mean |Δ|':>8}")
    any_scored = False
    for metric in METRICS:
        pairs = [
            (_score(r[f"judge_{metric}"]), _score(r[f"human_{metric}"]))
            for r in rows
        ]
        pairs = [(j, h) for j, h in pairs if j is not None and h is not None]
        if not pairs:
            print(f"{metric:<18} {0:>3}  {'—':>11}  {'—':>8}  {'—':>8}")
            continue
        any_scored = True
        judge, human = zip(*pairs)
        exact = sum(j == h for j, h in pairs) / len(pairs)
        kappa = cohens_kappa(judge, human)
        mad = sum(abs(j - h) for j, h in pairs) / len(pairs)
        kappa_txt = "n/a" if kappa != kappa else f"{kappa:.2f}"
        print(f"{metric:<18} {len(pairs):>3}  {exact * 100:>10.0f}%  {kappa_txt:>8}  {mad:>8.2f}")

    if not any_scored:
        print("\nNo human scores yet — fill in the human_* columns and rerun.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
