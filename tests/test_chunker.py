import nltk
from langchain_core.documents import Document

from src.chunker import _sentence_chunks, split_documents

TEXT = (
    "Dr. Smith measured 3.5 kg on Jan. 4. The results were clear. "
    "Risk is the composite measure of an event's probability and its consequences. "
    "Residual risk is the risk remaining after risk treatment. "
    "Each function is divided into categories. Subcategories divide each category further. "
    "Short one. Another short sentence here."
)


def _sentences(text):
    return nltk.sent_tokenize(text)


def test_chunks_contain_only_whole_sentences():
    sentences = set(_sentences(TEXT))
    for chunk in _sentence_chunks(TEXT, chunk_size=120, chunk_overlap=40):
        for sentence in _sentences(chunk):
            assert sentence in sentences, f"split mid-sentence: {sentence!r}"


def test_every_sentence_is_covered():
    chunks = _sentence_chunks(TEXT, chunk_size=120, chunk_overlap=40)
    for sentence in _sentences(TEXT):
        assert any(sentence in c for c in chunks)


def test_chunks_respect_size_unless_single_sentence_is_longer():
    for chunk in _sentence_chunks(TEXT, chunk_size=120, chunk_overlap=40):
        assert len(chunk) <= 120 or len(_sentences(chunk)) == 1


def test_overlap_repeats_whole_sentences():
    chunks = _sentence_chunks(TEXT, chunk_size=120, chunk_overlap=40)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_sentences = _sentences(prev)
        if len(prev_sentences) == 1:
            continue  # nothing to overlap without stalling; the chunker moves on
        assert _sentences(nxt)[0] in prev_sentences


def test_split_documents_adds_metadata():
    docs = [Document(page_content=TEXT, metadata={"source": "a.pdf", "page": 0})]
    chunks = split_documents(docs, chunk_size=120, chunk_overlap=40, strategy="sentence")
    assert all(c.metadata["source"] == "a.pdf" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["total_chunks"] == len(chunks) for c in chunks)
