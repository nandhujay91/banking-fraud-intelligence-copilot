
import pytest

from banking_fraud.rag.document_registry import DocumentRegistry


def create_metadata(checksum="abc123"):
    return {
        "document_id": "doc-001",
        "filename": "fraud_policy.pdf",
        "file_format": "pdf",
        "sha256": checksum,
    }


def test_registry_creates_json_file(tmp_path):
    registry_path = tmp_path / "metadata" / "registry.json"

    DocumentRegistry(registry_path)

    assert registry_path.exists()


def test_register_document(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")
    metadata = create_metadata()

    registry.register(metadata)

    documents = registry.get_all()

    assert len(documents) == 1
    assert documents[0]["filename"] == "fraud_policy.pdf"


def test_detect_duplicate_document(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")

    registry.register(create_metadata())

    assert registry.is_duplicate("abc123") is True
    assert registry.is_duplicate("different456") is False


def test_reject_duplicate_registration(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")

    registry.register(create_metadata())

    with pytest.raises(ValueError, match="Duplicate document"):
        registry.register(create_metadata())


def test_registry_persists_documents_after_restart(tmp_path):
    registry_path = tmp_path / "registry.json"

    first_registry = DocumentRegistry(registry_path)
    first_registry.register(create_metadata())

    second_registry = DocumentRegistry(registry_path)

    assert len(second_registry.get_all()) == 1
    assert second_registry.is_duplicate("abc123") is True


def test_reject_empty_checksum(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")

    with pytest.raises(ValueError, match="cannot be empty"):
        registry.is_duplicate("   ")


def test_reject_metadata_without_checksum(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")

    with pytest.raises(ValueError, match="valid SHA-256"):
        registry.register({"filename": "policy.pdf"})


def test_reject_invalid_metadata_type(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.json")

    with pytest.raises(TypeError, match="Metadata must be a dictionary"):
        registry.register("invalid metadata")
