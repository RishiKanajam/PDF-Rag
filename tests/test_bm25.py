from src.bm25 import BM25Index, tokenize


def test_lowercases():
    assert tokenize("Neural MESH Flow") == ["neural", "mesh", "flow"]


def test_strips_surrounding_punctuation():
    assert tokenize("BM25, (dense) search; done.") == ["bm25", "dense", "search", "done"]
    assert tokenize("What is it?") == tokenize("what is it")


def test_keeps_codes_whole_and_adds_their_parts():
    tokens = tokenize("ID.AM-01: Inventories")
    assert "id.am-01" in tokens
    assert {"id", "am", "01"} <= set(tokens)
    assert "inventories" in tokens


def test_code_matches_regardless_of_trailing_punctuation():
    # The old whitespace tokenizer produced "id.am-01:" vs "id.am-01?" — no match.
    assert "id.am-01" in tokenize("o ID.AM-01: Inventories of hardware")
    assert "id.am-01" in tokenize("What is ID.AM-01?")


def test_numbers_and_decimals():
    assert tokenize("GOVERN 1.6:") == ["govern", "1.6", "1", "6"]
    assert "116-283" in tokenize("(P.L. 116-283)")
    assert "2020" in tokenize("Act of 2020.")


def test_search_finds_exact_code():
    docs = [
        {"id": "a", "document": "o ID.AM-01: Inventories of hardware are maintained", "metadata": {}},
        {"id": "b", "document": "o ID.AM-02: Inventories of software are maintained", "metadata": {}},
        {"id": "c", "document": "Risk tolerance is contextual.", "metadata": {}},
    ]
    index = BM25Index()
    index.build(docs)
    assert index.search("What is ID.AM-01?", top_k=1)[0]["id"] == "a"


def test_load_rejects_index_built_with_other_tokenizer(tmp_path):
    import pickle

    path = tmp_path / "bm25.pkl"
    with open(path, "wb") as f:
        pickle.dump({"bm25": object(), "docs": []}, f)  # pre-fix format, no tokenizer tag
    assert BM25Index().load(path) is False
