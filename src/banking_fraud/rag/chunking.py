
"""Split extracted documents into overlapping text chunks."""

from uuid import uuid4


def split_into_chunks(
    text: str,
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, object]]:
    """Split document text into overlapping text chunks."""

    if not text or not text.strip():
        raise ValueError("Document text cannot be empty.")

    if not document_id or not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    cleaned_text = text.strip()
    chunks: list[dict[str, object]] = []

    step = chunk_size - chunk_overlap
    start = 0
    chunk_index = 0

    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))

        chunk_text = cleaned_text[start:end]

        chunks.append(
            {
                "chunk_id": str(uuid4()),
                "document_id": document_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
            }
        )

        chunk_index += 1

        if end == len(cleaned_text):
            break

        start += step

    return chunks
