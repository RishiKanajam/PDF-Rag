"""
metrics.py
----------
Pure functions for retrieval scoring. No models, no network — unit-tested in
tests/test_metrics.py.

Gold-passage matching:
  A retrieved chunk "contains the gold passage" if it comes from the gold
  document and, after normalisation, either contains the gold quote verbatim
  or shares a contiguous run covering >= 80% of the quote (this catches quotes
  split across a chunk boundary or mangled by PDF extraction).

  Normalisation keeps only lowercase letters and digits. PDF text extraction
  inserts stray spaces and line-break hyphens ("mea-\\nsure", "P R.DS- 01"),
  so comparing alphanumerics only is what makes exact matching reliable.
"""

import math
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence

OVERLAP_THRESHOLD = 0.8


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^a-z0-9]", "", text)


def quote_overlap(quote: str, text: str) -> float:
    """Fraction of the normalised quote covered by its longest contiguous match in text."""
    q, t = normalize(quote), normalize(text)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    m = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return m.size / len(q)


def chunk_matches(chunk: Dict[str, Any], gold_doc: str, gold_quote: str,
                  threshold: float = OVERLAP_THRESHOLD) -> bool:
    source = os.path.basename(str(chunk.get("metadata", {}).get("source", "")))
    if source != gold_doc:
        return False
    return quote_overlap(gold_quote, chunk["document"]) >= threshold


def first_relevant_rank(relevant: Sequence[bool]) -> int | None:
    """1-based rank of the first relevant result, or None."""
    for i, rel in enumerate(relevant, 1):
        if rel:
            return i
    return None


def recall_at_k(relevant: Sequence[bool], k: int) -> float:
    """1.0 if any of the top-k results is relevant (single gold passage per question)."""
    return 1.0 if any(relevant[:k]) else 0.0


def mrr_at_k(relevant: Sequence[bool], k: int) -> float:
    rank = first_relevant_rank(relevant[:k])
    return 1.0 / rank if rank else 0.0


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return float("nan")
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]
