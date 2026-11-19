
import pytest

from banking_fraud.rag.chunking import split_into_chunks


def test_split_text_into_chunks():
    text = "A" * 2500

    chunks = split_into_chunks(
        text=text,
        document_id="doc-001",
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert len(chunks) == 3
    assert len(chunks[0]["text"]) == 1000
    assert len(chunks[1]["text"]) == 1000
    assert len(chunks[2]["text"]) == 900


def test_chunks_have_unique_ids():
    chunks = split_into_chunks(
        text="A" * 2000,
        document_id="doc-001",
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunks_preserve_document_id():
    chunks = split_into_chunks(
        text="Fraud investigation policy.",
        document_id="doc-123",
    )

    assert all(
        chunk["document_id"] == "doc-123"
        for chunk in chunks
    )


def test_chunk_indexes_are_sequential():
    chunks = split_into_chunks(
        text="A" * 2500,
        document_id="doc-001",
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert [
        chunk["chunk_index"] for chunk in chunks
    ] == [0, 1, 2]


def test_chunk_overlap_is_preserved():
    text = "A" * 2000

    chunks = split_into_chunks(
        text=text,
        document_id="doc-001",
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert chunks[0]["text"][-200:] == chunks[1]["text"][:200]


def test_reject_empty_text():
    with pytest.raises(ValueError, match="cannot be empty"):
        split_into_chunks("", "doc-001")


def test_reject_invalid_chunk_size():
    with pytest.raises(ValueError, match="greater than zero"):
        split_into_chunks(
            "Some text",
            "doc-001",
            chunk_size=0,
        )


def test_reject_invalid_overlap():
    with pytest.raises(ValueError, match="smaller than chunk size"):
        split_into_chunks(
            "Some text",
            "doc-001",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_reject_negative_overlap():
    with pytest.raises(ValueError, match="cannot be negative"):
        split_into_chunks(
            "Some text",
            "doc-001",
            chunk_overlap=-1,
        )


def test_reject_empty_document_id():
    with pytest.raises(ValueError, match="Document ID cannot be empty"):
        split_into_chunks("Some text", "   ")
