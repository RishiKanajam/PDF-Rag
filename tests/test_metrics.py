import pytest

from eval.metrics import (
    chunk_matches,
    mrr_at_k,
    normalize,
    percentile,
    quote_overlap,
    recall_at_k,
)


# ── recall@k / MRR@k ──────────────────────────────────────────────────────────

def test_recall_at_k():
    relevant = [False, False, True, False]
    assert recall_at_k(relevant, 2) == 0.0
    assert recall_at_k(relevant, 3) == 1.0
    assert recall_at_k([], 5) == 0.0


def test_mrr_at_k():
    assert mrr_at_k([True, False], 10) == 1.0
    assert mrr_at_k([False, False, True], 10) == pytest.approx(1 / 3)
    assert mrr_at_k([False, False, True], 2) == 0.0   # beyond the cutoff
    assert mrr_at_k([False, True, True], 10) == 0.5   # first relevant only


def test_percentile_nearest_rank():
    values = [5, 1, 4, 2, 3]
    assert percentile(values, 50) == 3
    assert percentile(values, 95) == 5
    assert percentile(list(range(1, 21)), 95) == 19


# ── quote matching ────────────────────────────────────────────────────────────

def test_normalize_ignores_pdf_extraction_noise():
    assert normalize("ID .RA -01") == normalize("ID.RA-01")
    assert normalize("mea-\nsure the risk") == normalize("measure the risk")
    assert normalize("Community P rofile") == normalize("Community Profile")
    assert normalize("event’s “quoted”") == normalize("event's \"quoted\"")


def test_exact_quote_inside_chunk():
    chunk = {"document": "Intro. GOVERN 1.6: Mechanisms are in place to inventory AI systems. More.",
             "metadata": {"source": "/abs/path/NIST.AI.100-1.pdf"}}
    assert chunk_matches(chunk, "NIST.AI.100-1.pdf", "GOVERN 1.6: Mechanisms are in place to inventory AI systems")


def test_wrong_document_never_matches():
    chunk = {"document": "Suppliers are known and prioritized by criticality",
             "metadata": {"source": "other.pdf"}}
    assert not chunk_matches(chunk, "NIST.CSWP.29.pdf", "Suppliers are known and prioritized by criticality")


def test_partial_overlap_threshold():
    quote = "abcdefghij" * 10  # 100 normalised chars
    assert quote_overlap(quote, "xx" + quote[:85] + "yy") == pytest.approx(0.85)
    assert chunk_matches({"document": quote[:85], "metadata": {"source": "d.pdf"}}, "d.pdf", quote)
    assert not chunk_matches({"document": quote[:70], "metadata": {"source": "d.pdf"}}, "d.pdf", quote)


def test_unrelated_text_does_not_match():
    assert quote_overlap("residual risk remains after treatment", "the cat sat on the mat") < 0.3


# ── judge–human agreement ─────────────────────────────────────────────────────

def test_cohens_kappa():
    from eval.metrics import cohens_kappa

    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # observed 0.5, expected 0.5 → chance-level agreement
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    # Wikipedia example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no
    # po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5 → kappa = 0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)
