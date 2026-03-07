"""
loader.py
---------
Responsible for loading documents from disk.
Supports PDFs (via PyPDFLoader) and plain text files (via TextLoader).

What you learn here:
- LangChain Document object: every loaded page becomes a Document with
  .page_content (the text) and .metadata (source, page number, etc.)
- DirectoryLoader: scans an entire folder and loads all matching files
- Why we attach metadata: it lets us trace every answer back to its source
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)

import config


def load_pdfs(pdf_dir: Path = config.PDF_DIR) -> List[Document]:
    """Load all PDF files from pdf_dir. Each page becomes one Document."""
    if not pdf_dir.exists():
        print(f"[loader] PDF directory not found: {pdf_dir}")
        return []

    loader = DirectoryLoader(
        str(pdf_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False,
    )
    docs = loader.load()
    print(f"[loader] Loaded {len(docs)} pages from PDFs in '{pdf_dir}'")
    return docs


def load_text_files(text_dir: Path = config.TEXT_DIR) -> List[Document]:
    """Load all .txt files from text_dir. Each file becomes one Document."""
    if not text_dir.exists():
        print(f"[loader] Text directory not found: {text_dir}")
        return []

    loader = DirectoryLoader(
        str(text_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    print(f"[loader] Loaded {len(docs)} text file(s) from '{text_dir}'")
    return docs


def load_all(
    pdf_dir: Path = config.PDF_DIR,
    text_dir: Path = config.TEXT_DIR,
) -> List[Document]:
    """Load all PDFs and text files and return them as a single list."""
    docs = load_pdfs(pdf_dir) + load_text_files(text_dir)
    print(f"[loader] Total documents loaded: {len(docs)}")
    return docs
