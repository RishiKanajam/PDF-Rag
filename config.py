import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "Data"
PDF_DIR         = DATA_DIR / "pdfs"
TEXT_DIR        = DATA_DIR / "text_files"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
BM25_INDEX_PATH  = DATA_DIR / "bm25_index.pkl"

# ── Models (pinned) ──────────────────────────────────────────────────────────
# Every model the pipeline and the evaluation use is pinned here. Hugging Face
# models are pinned to a commit hash; Groq models are pinned by name (Groq
# exposes no finer version, so eval results also record the system_fingerprint).
EMBEDDING_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER_MODEL           = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_REVISION  = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
LLM_MODEL                = "llama-3.1-8b-instant"
JUDGE_PROVIDER           = "groq"
JUDGE_MODEL              = "llama-3.3-70b-versatile"

# ── Chunking ───────────────────────────────────────────────────────────────────
# strategy: "sentence" (NLTK sentence-aware, never cuts mid-sentence)
#           "recursive" (character-based, the old approach)
CHUNKING_STRATEGY = "sentence"
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200

# ── Vector Store ───────────────────────────────────────────────────────────────
COLLECTION_NAME = "pdf_documents"

# ── Retrieval ──────────────────────────────────────────────────────────────────
# HYBRID_TOP_K: candidates each search (BM25 + semantic) fetches before merging.
HYBRID_TOP_K    = 10
# FINAL_TOP_K: chunks the reranker keeps and sends to the LLM.
FINAL_TOP_K     = 3
SCORE_THRESHOLD = 0.0
# k constant for Reciprocal Rank Fusion — 60 is the standard value.
RRF_K           = 60
# "dense"         — embeddings only
# "bm25"          — keyword only
# "hybrid"        — BM25 + dense fused with RRF
# "hybrid_rerank" — hybrid, then the cross-encoder reranks the fused candidates
RETRIEVAL_MODE  = "hybrid_rerank"

# ── Memory ────────────────────────────────────────────────────────────────────
# How many past conversation turns to keep in context (1 turn = 1 Q&A pair).
# Higher = more context, but uses more LLM tokens per query.
MEMORY_MAX_TURNS = 5

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1000


def validate_config():
    """Fail fast if required environment variables are missing."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
