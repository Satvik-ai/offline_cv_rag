"""Load local API documentation and build a persistent ChromaDB index.

Supported formats: PDF, Markdown, plain text, reStructuredText, HTML,
Python source, Jupyter notebooks, JSON, YAML, TOML and CSV.

Usage:
    python ingest.py --reset
    python ingest.py --docs-dir ./docs --persist-dir ./chroma_db --reset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import chromadb
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "offline_cv_docs")
SUPPORTED_EXTENSIONS = {
    ".pdf", ".md", ".markdown", ".txt", ".rst", ".html", ".htm",
    ".py", ".ipynb", ".json", ".yaml", ".yml", ".toml", ".csv",
}


def _read_text(path: Path) -> str:
    """Read text robustly when documentation encoding is not UTF-8."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path, relative_source: str) -> list[Document]:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("PDF support requires PyMuPDF. Install with: pip install pymupdf") from exc

    docs: list[Document] = []
    with pymupdf.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            # sort=True usually gives a more natural top-to-bottom reading order.
            text = _clean_text(page.get_text("text", sort=True))
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": relative_source,
                            "file_type": "pdf",
                            "page": page_number,
                        },
                    )
                )
    if not docs:
        print(
            f"[WARN] No selectable text found in PDF: {relative_source}. "
            "It may be scanned/image-only; OCR it before indexing."
        )
    return docs


def _load_notebook(path: Path, relative_source: str) -> str:
    notebook = json.loads(_read_text(path))
    blocks: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if not source.strip():
            continue
        cell_type = cell.get("cell_type", "unknown")
        blocks.append(f"\n# Jupyter {cell_type} cell\n{source}")
    return "\n".join(blocks)


def load_documentation(docs_dir: str | Path) -> list[Document]:
    """Read every supported file recursively, preserving source/page metadata."""
    root = Path(docs_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Documentation directory does not exist: {root}")

    documents: list[Document] = []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise RuntimeError(
            f"No supported documentation files found in {root}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    for path in files:
        relative_source = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                file_docs = _load_pdf(path, relative_source)
                documents.extend(file_docs)
                print(f"[LOAD] {relative_source}: {len(file_docs)} text page(s)")
                continue

            if suffix in {".html", ".htm"}:
                soup = BeautifulSoup(_read_text(path), "html.parser")
                for unwanted in soup(["script", "style", "noscript", "svg"]):
                    unwanted.decompose()
                text = soup.get_text(separator="\n")
            elif suffix == ".ipynb":
                text = _load_notebook(path, relative_source)
            else:
                text = _read_text(path)

            text = _clean_text(text)
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": relative_source, "file_type": suffix.lstrip(".")},
                    )
                )
                print(f"[LOAD] {relative_source}: {len(text):,} characters")
            else:
                print(f"[WARN] Empty document skipped: {relative_source}")
        except Exception as exc:
            print(f"[WARN] Could not read {relative_source}: {exc}")

    if not documents:
        raise RuntimeError(
            "No readable text was extracted. For scanned PDFs, run OCR first; "
            "for other files, check that they are not empty or corrupted."
        )
    return documents


def create_embeddings() -> OllamaEmbeddings:
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaEmbeddings(model=model, base_url=base_url)


def _chunk_ids(chunks: Iterable[Document]) -> list[str]:
    ids: list[str] = []
    for chunk in chunks:
        identity = "|".join(
            [
                str(chunk.metadata.get("source", "")),
                str(chunk.metadata.get("page", "")),
                str(chunk.metadata.get("start_index", "")),
                chunk.page_content,
            ]
        )
        ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())
    return ids


def build_vectorstore(
    docs_dir: str | Path = "./docs",
    persist_dir: str | Path = "./chroma_db",
    *,
    reset: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Chroma:
    """Build or update the persistent Chroma index.

    Use reset=True after changing the embedding model or replacing/removing docs.
    Without reset, deterministic IDs make repeated ingestion update existing chunks.
    """
    docs = load_documentation(docs_dir)
    size = chunk_size or int(os.getenv("CHUNK_SIZE", "1200"))
    overlap = chunk_overlap if chunk_overlap is not None else int(os.getenv("CHUNK_OVERLAP", "180"))
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise RuntimeError("The documentation produced no chunks.")

    persist_path = Path(persist_dir).expanduser().resolve()
    persist_path.mkdir(parents=True, exist_ok=True)

    if reset:
        client = chromadb.PersistentClient(path=str(persist_path))
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            # Collection may not exist yet. Continue with a fresh collection.
            pass

    embeddings = create_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )
    BATCH_SIZE = 8

    total_chunks = len(chunks)

    for start in range(0, total_chunks, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total_chunks)

        batch = chunks[start:end]
        batch_ids = _chunk_ids(batch)

        print(
            f"[INDEX] Processing "
            f"{start + 1}-{end}/{total_chunks}"
        )

        vectorstore.add_documents(
            documents=batch,
            ids=batch_ids
        )

    print("[INDEX] Indexing completed!")

    print(f"\n[INDEX] Documents/pages: {len(docs)}")
    print(f"[INDEX] Chunks: {len(chunks)}")
    print(f"[INDEX] Database: {persist_path}")
    print(f"[INDEX] Collection: {COLLECTION_NAME}")
    print("[INDEX] ChromaDB persists automatically to the directory above.")
    return vectorstore


def load_vectorstore(persist_dir: str | Path = "./chroma_db") -> Chroma:
    """Open an existing persistent Chroma collection and verify it is non-empty."""
    persist_path = Path(persist_dir).expanduser().resolve()
    if not persist_path.exists():
        raise FileNotFoundError(f"Chroma database not found: {persist_path}")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(persist_path),
    )
    try:
        sample = vectorstore.get(limit=1)
        if not sample.get("ids"):
            raise RuntimeError("Chroma collection is empty. Run: python ingest.py --reset")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Could not open the Chroma collection. Rebuild it with `python ingest.py --reset`. "
            "Also check that the embedding model is the same one used during ingestion."
        ) from exc
    return vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local CV/API documentation into ChromaDB.")
    parser.add_argument("--docs-dir", default=os.getenv("DOCS_DIR", "./docs"))
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_DIR", "./chroma_db"))
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the collection")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()
    build_vectorstore(
        args.docs_dir,
        args.persist_dir,
        reset=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
