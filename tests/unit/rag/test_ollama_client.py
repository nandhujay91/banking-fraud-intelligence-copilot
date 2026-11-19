
import json

import httpx
import pytest

from banking_fraud.llm.ollama_client import (
    OllamaClient,
    OllamaError,
)


def test_generate_returns_model_response():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)

        assert request.url.path == "/api/chat"
        assert payload["model"] == "qwen2.5:3b"
        assert payload["stream"] is False

        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "Fraud detection identifies suspicious transactions."
                }
            },
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    result = client.generate("Explain fraud detection.")

    assert result == (
        "Fraud detection identifies suspicious transactions."
    )


def test_generate_rejects_empty_prompt():
    client = OllamaClient()

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("   ")


def test_generate_rejects_invalid_response():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={})
    )
    client = OllamaClient(transport=transport)

    with pytest.raises(OllamaError, match="invalid response"):
        client.generate("Explain fraud detection.")


def test_generate_rejects_http_error():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(500)
    )
    client = OllamaClient(transport=transport)

    with pytest.raises(OllamaError, match="Could not communicate"):
        client.generate("Explain fraud detection.")
