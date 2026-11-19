
import httpx
import pytest

from banking_fraud.rag.embedding_client import (
    EmbeddingError,
    OllamaEmbeddingClient,
)


def test_embed_returns_768_dimensional_vector():
    def mock_response(request):
        return httpx.Response(
            200,
            json={
                "embeddings": [[0.1] * 768],
            },
        )

    transport = httpx.MockTransport(mock_response)

    client = OllamaEmbeddingClient(transport=transport)

    embedding = client.embed("Suspicious banking transaction")

    assert len(embedding) == 768
    assert all(isinstance(value, float) for value in embedding)


def test_embed_rejects_empty_text():
    client = OllamaEmbeddingClient()

    with pytest.raises(ValueError, match="Text cannot be empty"):
        client.embed("   ")


def test_embed_rejects_invalid_response():
    def mock_response(request):
        return httpx.Response(
            200,
            json={"unexpected": "response"},
        )

    transport = httpx.MockTransport(mock_response)

    client = OllamaEmbeddingClient(transport=transport)

    with pytest.raises(EmbeddingError, match="invalid embedding"):
        client.embed("Banking fraud")


def test_embed_rejects_empty_embedding():
    def mock_response(request):
        return httpx.Response(
            200,
            json={"embeddings": [[]]},
        )

    transport = httpx.MockTransport(mock_response)

    client = OllamaEmbeddingClient(transport=transport)

    with pytest.raises(EmbeddingError, match="empty embedding"):
        client.embed("Banking fraud")


def test_embed_handles_http_error():
    def mock_response(request):
        return httpx.Response(500)

    transport = httpx.MockTransport(mock_response)

    client = OllamaEmbeddingClient(transport=transport)

    with pytest.raises(EmbeddingError, match="Could not communicate"):
        client.embed("Banking fraud")


def test_embed_rejects_non_numeric_values():
    def mock_response(request):
        return httpx.Response(
            200,
            json={"embeddings": [["invalid"]]},
        )

    transport = httpx.MockTransport(mock_response)

    client = OllamaEmbeddingClient(transport=transport)

    with pytest.raises(EmbeddingError, match="invalid values"):
        client.embed("Banking fraud")
