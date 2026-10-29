"""
verify_questions.py
-------------------
Checks every gold_quote in eval/questions.jsonl appears verbatim (after
normalisation, see metrics.normalize) on its gold_page of its gold_doc, using
the same PDF loader the pipeline indexes with.

Usage:
    python eval/verify_questions.py        # exits 1 if any quote fails
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFLoader

from eval.metrics import normalize

EVAL_DIR = Path(__file__).resolve().parent
DOCS_DIR = EVAL_DIR / "docs"


def main() -> int:
    questions = [json.loads(l) for l in open(EVAL_DIR / "questions.jsonl")]
    pages = {
        pdf.name: [normalize(p.page_content) for p in PyPDFLoader(str(pdf)).load()]
        for pdf in sorted(DOCS_DIR.glob("*.pdf"))
    }

    failures = 0
    for q in questions:
        if q["type"] == "unanswerable":
            continue
        doc_pages = pages.get(q["gold_doc"])
        if doc_pages is None:
            print(f"FAIL {q['id']}: gold_doc {q['gold_doc']} not in {DOCS_DIR}")
            failures += 1
            continue
        quote = normalize(q["gold_quote"])
        found_on = [i for i, text in enumerate(doc_pages) if quote in text]
        if q["gold_page"] in found_on:
            print(f"ok   {q['id']}: page {q['gold_page']}")
        else:
            print(f"FAIL {q['id']}: quote not on page {q['gold_page']} (found on {found_on})")
            failures += 1

    answerable = sum(q["type"] != "unanswerable" for q in questions)
    print(f"\n{answerable - failures}/{answerable} gold quotes verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
