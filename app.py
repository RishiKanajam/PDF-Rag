"""
app.py
------
Streamlit web interface for PDFRag.

Run with:
    streamlit run app.py

What you learn here:

STREAMLIT EXECUTION MODEL (critical to understand):
  Streamlit reruns the ENTIRE script from top to bottom on every user
  interaction — every button click, every message sent, every file upload.
  This means:
    - Variables you set at the top get reset on every rerun.
    - You MUST use st.session_state to persist anything across reruns.
    - You MUST use @st.cache_resource for expensive objects (models, DB
      connections) — otherwise they reload on every message.

@st.cache_resource:
  Runs the decorated function ONCE and caches the return value for the
  lifetime of the Streamlit server. Subsequent calls return the cached object.
  Perfect for: ML models, database connections, the RAGPipeline.

st.session_state:
  A dict-like object that persists between reruns within the same browser
  session. We store:
    - messages   → the chat history to display (list of {role, content, ...})
    - memory     → the ConversationMemory object used for reformulation
  These reset if the user refreshes the page (new session), which is fine.

PDF UPLOAD WORKFLOW:
  1. User uploads a PDF via the sidebar file uploader.
  2. We save it to Data/pdfs/.
  3. User clicks "Rebuild Index" to re-index all documents including the new one.
  4. The cached pipeline is cleared and rebuilt with the new file included.
"""

import shutil
from pathlib import Path

import streamlit as st

import config

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="PDFRag",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Validate config before anything else ──────────────────────────────────────
try:
    config.validate_config()
except EnvironmentError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

from src.memory import ConversationMemory
from src.rag import RAGPipeline


# ── Pipeline (cached — loads once, reused across all reruns) ──────────────────

@st.cache_resource(show_spinner="Loading models and index...")
def load_pipeline() -> RAGPipeline:
    """
    Load and build the RAG pipeline.
    @st.cache_resource ensures this runs only once per Streamlit server start.
    The models (embedding, reranker) and ChromaDB connection are expensive to
    initialize — caching saves ~10-30 seconds on every rerun.
    """
    pipeline = RAGPipeline()
    pipeline.build()
    return pipeline


# ── Session state initialisation ───────────────────────────────────────────────
# session_state persists within a browser session (lost on page refresh).
# We store chat history and the ConversationMemory here.

if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, sources?, standalone?}

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 PDFRag")
    st.caption("Ask questions about your documents")
    st.divider()

    # ── Document upload ────────────────────────────────────────────────────────
    st.subheader("Add Documents")
    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a new PDF to add to the knowledge base. Click 'Rebuild Index' after uploading.",
    )

    if uploaded is not None:
        dest = config.PDF_DIR / uploaded.name
        if not dest.exists():
            config.PDF_DIR.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                shutil.copyfileobj(uploaded, f)
            st.success(f"Saved **{uploaded.name}**. Click 'Rebuild Index' to include it.")
        else:
            st.info(f"**{uploaded.name}** is already in the knowledge base.")

    # ── Index controls ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Index Controls")

    if st.button("🔄 Rebuild Index", use_container_width=True):
        # Clear the cached pipeline so load_pipeline() runs again with force=True
        load_pipeline.clear()
        with st.spinner("Rebuilding index..."):
            pipeline = load_pipeline()
        st.success("Index rebuilt successfully.")
        st.rerun()

    # ── Stats ──────────────────────────────────────────────────────────────────
    try:
        pipeline = load_pipeline()
        vec_count = pipeline.vector_store.count
        turn_count = st.session_state.memory.turn_count()
        st.divider()
        st.subheader("Stats")
        col1, col2 = st.columns(2)
        col1.metric("Vectors", vec_count)
        col2.metric("Turns", f"{turn_count}/{config.MEMORY_MAX_TURNS}")

        # List indexed PDFs
        pdf_files = list(config.PDF_DIR.glob("*.pdf"))
        txt_files = list(config.TEXT_DIR.glob("*.txt"))
        if pdf_files or txt_files:
            st.divider()
            st.subheader("Indexed Documents")
            for f in pdf_files:
                st.markdown(f"📄 {f.name}")
            for f in txt_files:
                st.markdown(f"📝 {f.name}")
    except Exception:
        pass

    # ── Clear chat ─────────────────────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = ConversationMemory()
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────

st.header("Chat with your documents")

# Replay all previous messages from session_state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show reformulated question (assistant messages only)
        if msg.get("standalone") and msg["standalone"] != msg.get("original_question"):
            st.caption(f'🔍 Searched for: *"{msg["standalone"]}"*')

        # Show source citations (assistant messages only)
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                seen = set()
                for s in msg["sources"]:
                    key = (s["source"], str(s["page"]))
                    if key not in seen:
                        seen.add(key)
                        score = s.get("rerank_score", "N/A")
                        st.markdown(
                            f"**{s['source']}** — "
                            f"page {s['page']} &nbsp;·&nbsp; "
                            f"rerank score: `{score}`"
                        )


# ── Chat input ─────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask anything about your documents..."):

    # Load pipeline (from cache — instant if already loaded)
    pipeline = load_pipeline()

    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = pipeline.chat(question, st.session_state.memory)

        answer   = result["answer"]
        sources  = result["sources"]
        standalone = result.get("standalone_question", question)

        st.markdown(answer)

        # Show reformulated question if it differs
        if standalone != question:
            st.caption(f'🔍 Searched for: *"{standalone}"*')

        # Show sources
        if sources:
            with st.expander("Sources", expanded=False):
                seen = set()
                for s in sources:
                    key = (s["source"], str(s["page"]))
                    if key not in seen:
                        seen.add(key)
                        score = s.get("rerank_score", "N/A")
                        st.markdown(
                            f"**{s['source']}** — "
                            f"page {s['page']} &nbsp;·&nbsp; "
                            f"rerank score: `{score}`"
                        )

    # Persist assistant message to session_state for replay on next rerun
    st.session_state.messages.append({
        "role":              "assistant",
        "content":           answer,
        "sources":           sources,
        "standalone":        standalone,
        "original_question": question,
    })
