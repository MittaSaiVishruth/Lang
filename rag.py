import os
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_documents(file_name: str, file_bytes: bytes) -> list[Document]:
    """Load PDF/TXT/Markdown data into LangChain Documents."""
    suffix = Path(file_name).suffix.lower()

    if suffix == ".pdf":
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path, mode="page")
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_name"] = file_name

            return documents
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    if suffix in {".txt", ".md"}:
        text = file_bytes.decode("utf-8", errors="replace")
        return [
            Document(
                page_content=text,
                metadata={"source": file_name, "source_name": file_name},
            )
        ]

    raise ValueError("Unsupported file type. Use PDF, TXT, or Markdown.")


def build_vector_store(file_name: str, file_bytes: bytes, embeddings):
    """Load, split, embed, and index a document."""
    documents = _load_documents(file_name, file_bytes)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i + 1
        chunk.metadata["source_name"] = file_name

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)

    return vector_store


def format_sources(documents: list[Document]) -> str:
    """Format retrieved source metadata for display."""
    if not documents:
        return "No sources retrieved."

    lines = []
    seen = set()

    for doc in documents:
        source_name = doc.metadata.get("source_name") or doc.metadata.get("source") or "Unknown source"
        page = doc.metadata.get("page")

        if page is not None:
            label = f"- **{source_name}**, page {page + 1}"
        else:
            label = f"- **{source_name}**"

        if label not in seen:
            lines.append(label)
            seen.add(label)

    return "\n".join(lines)
