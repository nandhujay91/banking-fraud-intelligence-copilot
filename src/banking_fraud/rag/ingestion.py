
"""Ingest knowledge-base documents with duplicate detection and chunking."""

from pathlib import Path
from typing import Any

from banking_fraud.rag.chunking import split_into_chunks
from banking_fraud.rag.document_loader import load_document
from banking_fraud.rag.document_metadata import create_document_metadata
from banking_fraud.rag.document_registry import DocumentRegistry

DEFAULT_REGISTRY_PATH = (
    Path("data")
    / "knowledge_base"
    / "metadata"
    / "document_registry.json"
)


def ingest_document(
    file_path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict[str, Any]:
    """Extract, validate, register, and chunk a knowledge-base document."""

    metadata = create_document_metadata(file_path)

    registry = DocumentRegistry(registry_path)

    if registry.is_duplicate(metadata["sha256"]):
        raise ValueError(
            f"Duplicate document detected: {metadata['filename']}"
        )

    extracted_text = load_document(file_path)

    chunks = split_into_chunks(
        text=extracted_text,
        document_id=metadata["document_id"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    registry.register(metadata)

    return {
        "metadata": metadata,
        "text": extracted_text,
        "chunks": chunks,
    }
