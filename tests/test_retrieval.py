import pytest

from src.retriever import reciprocal_rank_fusion


def _docs(*ids):
    return [{"id": i, "document": f"text {i}", "metadata": {}} for i in ids]


def test_rrf_known_example():
    # dense: a, b, c    bm25: c, a, d    (k = 60)
    merged = reciprocal_rank_fusion(_docs("a", "b", "c"), _docs("c", "a", "d"), k=60)
    expected = {
        "a": 1 / 61 + 1 / 62,
        "c": 1 / 63 + 1 / 61,
        "b": 1 / 62,
        "d": 1 / 63,
    }
    assert [m["id"] for m in merged] == ["a", "c", "b", "d"]
    for m in merged:
        assert m["rrf_score"] == pytest.approx(expected[m["id"]], abs=1e-6)


def test_rrf_single_list_preserves_order():
    merged = reciprocal_rank_fusion(_docs("x", "y", "z"))
    assert [m["id"] for m in merged] == ["x", "y", "z"]


def test_rrf_does_not_mutate_inputs():
    dense = _docs("a")
    reciprocal_rank_fusion(dense, _docs("a"))
    assert "rrf_score" not in dense[0]


def test_rrf_empty():
    assert reciprocal_rank_fusion([], []) == []
