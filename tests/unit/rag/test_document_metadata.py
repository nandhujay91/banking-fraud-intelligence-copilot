
import hashlib

import pytest

from banking_fraud.rag.document_metadata import (
    calculate_sha256,
    create_document_metadata,
)


def test_calculate_sha256(tmp_path):
    file_path = tmp_path / "policy.txt"
    content = "Banking fraud investigation policy."
    file_path.write_text(content, encoding="utf-8")

    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()

    assert calculate_sha256(file_path) == expected


def test_create_document_metadata(tmp_path):
    file_path = tmp_path / "fraud_policy.pdf"
    file_path.write_bytes(b"Sample PDF content")

    metadata = create_document_metadata(file_path)

    assert metadata["filename"] == "fraud_policy.pdf"
    assert metadata["file_format"] == "pdf"
    assert metadata["source_path"] == str(file_path.resolve())
    assert metadata["file_size_bytes"] == len(b"Sample PDF content")
    assert len(metadata["document_id"]) == 36
    assert len(metadata["sha256"]) == 64
    assert metadata["created_at"]


def test_same_content_produces_same_checksum(tmp_path):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    content = "Suspicious transaction monitoring."

    first_file.write_text(content, encoding="utf-8")
    second_file.write_text(content, encoding="utf-8")

    assert calculate_sha256(first_file) == calculate_sha256(second_file)


def test_different_content_produces_different_checksum(tmp_path):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text("Fraud detection", encoding="utf-8")
    second_file.write_text("AML compliance", encoding="utf-8")

    assert calculate_sha256(first_file) != calculate_sha256(second_file)


def test_missing_document_is_rejected(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Document not found"):
        create_document_metadata(file_path)


def test_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        create_document_metadata(tmp_path)
