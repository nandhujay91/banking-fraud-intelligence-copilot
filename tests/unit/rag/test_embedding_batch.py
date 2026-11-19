
import httpx
import pytest

from banking_fraud.rag.embedding_client import (
    EmbeddingError,
    OllamaEmbeddingClient,
)


def test_embed_batch_returns_multiple_vectors():
    def mock_response(request):
        payload = request.read()
        assert b'"input":["Fraud transaction","Suspicious transfer"]' in payload

        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
            },
        )

    client = OllamaEmbeddingClient(
        transport=httpx.MockTransport(mock_response)
    )

    vectors = client.embed_batch(
        ["Fraud transaction", "Suspicious transfer"]
    )

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.4, 0.5, 0.6]


def test_embed_batch_processes_multiple_batches():
    request_count = 0

    def mock_response(request):
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2, 0.3]
                    for _ in range(2 if request_count == 1 else 1)
                ],
            },
        )

    client = OllamaEmbeddingClient(
        transport=httpx.MockTransport(mock_response)
    )

    vectors = client.embed_batch(
        ["Text 1", "Text 2", "Text 3"],
        batch_size=2,
    )

    assert len(vectors) == 3
    assert request_count == 2


def test_embed_batch_rejects_empty_input():
    client = OllamaEmbeddingClient()

    with pytest.raises(ValueError, match="Texts cannot be empty"):
        client.embed_batch([])


def test_embed_batch_rejects_invalid_batch_size():
    client = OllamaEmbeddingClient()

    with pytest.raises(ValueError, match="Batch size"):
        client.embed_batch(["Fraud"], batch_size=0)


def test_embed_batch_rejects_empty_text():
    client = OllamaEmbeddingClient()

    with pytest.raises(ValueError, match="non-empty strings"):
        client.embed_batch(["Fraud", "   "])


def test_embed_batch_rejects_wrong_embedding_count():
    def mock_response(request):
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2, 0.3]]},
        )

    client = OllamaEmbeddingClient(
        transport=httpx.MockTransport(mock_response)
    )

    with pytest.raises(EmbeddingError, match="count"):
        client.embed_batch(["Text 1", "Text 2"])


def test_embed_batch_rejects_inconsistent_dimensions():
    def mock_response(request):
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2],
                    [0.3, 0.4, 0.5],
                ],
            },
        )

    client = OllamaEmbeddingClient(
        transport=httpx.MockTransport(mock_response)
    )

    with pytest.raises(EmbeddingError, match="dimensions"):
        client.embed_batch(["Text 1", "Text 2"])


def test_embed_batch_rejects_non_numeric_values():
    def mock_response(request):
        return httpx.Response(
            200,
            json={"embeddings": [["invalid"]]},
        )

    client = OllamaEmbeddingClient(
        transport=httpx.MockTransport(mock_response)
    )

    with pytest.raises(EmbeddingError, match="invalid values"):
        client.embed_batch(["Fraud transaction"])
