
import pytest
from docx import Document

from banking_fraud.rag.document_loader import DocumentLoaderError
from banking_fraud.rag.ingestion import ingest_document


def test_ingest_txt_document_returns_text_and_metadata(tmp_path):
    file_path = tmp_path / "fraud_policy.txt"
    registry_path = tmp_path / "metadata" / "registry.json"

    content = "Suspicious transactions require investigation."
    file_path.write_text(content, encoding="utf-8")

    result = ingest_document(file_path, registry_path)

    assert result["text"] == content

    metadata = result["metadata"]

    assert metadata["filename"] == "fraud_policy.txt"
    assert metadata["file_format"] == "txt"
    assert metadata["file_size_bytes"] == len(content.encode("utf-8"))
    assert metadata["document_id"]
    assert metadata["sha256"]
    assert metadata["created_at"]


def test_ingest_docx_document_returns_text_and_metadata(tmp_path):
    file_path = tmp_path / "aml_policy.docx"
    registry_path = tmp_path / "metadata" / "registry.json"

    document = Document()
    document.add_paragraph("Anti-money laundering controls.")
    document.save(file_path)

    result = ingest_document(file_path, registry_path)

    assert result["text"] == "Anti-money laundering controls."
    assert result["metadata"]["filename"] == "aml_policy.docx"
    assert result["metadata"]["file_format"] == "docx"


def test_ingest_missing_document_is_rejected(tmp_path):
    file_path = tmp_path / "missing.txt"
    registry_path = tmp_path / "metadata" / "registry.json"

    with pytest.raises(FileNotFoundError):
        ingest_document(file_path, registry_path)


def test_duplicate_document_is_rejected(tmp_path):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    registry_path = tmp_path / "metadata" / "registry.json"

    content = "Suspicious transaction monitoring policy."

    first_file.write_text(content, encoding="utf-8")
    second_file.write_text(content, encoding="utf-8")

    ingest_document(first_file, registry_path)

    with pytest.raises(ValueError, match="Duplicate document detected"):
        ingest_document(second_file, registry_path)


def test_failed_extraction_does_not_register_document(tmp_path):
    file_path = tmp_path / "empty.txt"
    registry_path = tmp_path / "metadata" / "registry.json"

    file_path.write_text("", encoding="utf-8")

    with pytest.raises(DocumentLoaderError):
        ingest_document(file_path, registry_path)

    from banking_fraud.rag.document_registry import DocumentRegistry

    registry = DocumentRegistry(registry_path)

    assert registry.get_all() == []

def test_ingestion_generates_document_chunks(tmp_path):
    source_file = tmp_path / "fraud_policy.txt"
    source_file.write_text(
        "Banking fraud investigation policy. " * 100,
        encoding="utf-8",
    )

    registry_path = tmp_path / "registry.json"

    result = ingest_document(
        file_path=source_file,
        registry_path=registry_path,
        chunk_size=500,
        chunk_overlap=100,
    )

    metadata = result["metadata"]
    chunks = result["chunks"]

    assert len(chunks) > 1

    assert all(
        chunk["document_id"] == metadata["document_id"]
        for chunk in chunks
    )

    assert all(
        chunk["text"]
        for chunk in chunks
    )

    assert [
        chunk["chunk_index"] for chunk in chunks
    ] == list(range(len(chunks)))
