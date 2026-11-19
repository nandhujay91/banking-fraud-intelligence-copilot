
"""Generate text embeddings using a local Ollama model."""

from typing import Any

import httpx


class EmbeddingError(Exception):
    """Base exception for embedding generation errors."""


class OllamaEmbeddingClient:
    """Generate embeddings through the Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.transport = transport

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for one text input."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text cannot be empty.")

        vectors = self.embed_batch([text])
        return vectors[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate embeddings for multiple text inputs."""

        if not texts:
            raise ValueError("Texts cannot be empty.")

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if not all(
            isinstance(text, str) and text.strip()
            for text in texts
        ):
            raise ValueError("All texts must be non-empty strings.")

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            payload: dict[str, Any] = {
                "model": self.model,
                "input": batch,
            }

            try:
                with httpx.Client(
                    timeout=self.timeout,
                    transport=self.transport,
                ) as client:
                    response = client.post(
                        f"{self.base_url}/api/embed",
                        json=payload,
                    )
                    response.raise_for_status()

            except httpx.TimeoutException as exc:
                raise EmbeddingError(
                    "Embedding request timed out."
                ) from exc

            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"Could not communicate with Ollama: {exc}"
                ) from exc

            try:
                result = response.json()
                embeddings = result["embeddings"]

            except (ValueError, KeyError, TypeError) as exc:
                raise EmbeddingError(
                    "Ollama returned an invalid embedding response."
                ) from exc

            if not isinstance(embeddings, list):
                raise EmbeddingError(
                    "Ollama returned an invalid embedding response."
                )

            if len(embeddings) != len(batch):
                raise EmbeddingError(
                    "Embedding count does not match input count."
                )

            for vector in embeddings:
                if not isinstance(vector, list) or not vector:
                    raise EmbeddingError(
                        "Ollama returned an empty embedding."
                    )

                if not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in vector
                ):
                    raise EmbeddingError(
                        "Embedding contains invalid values."
                    )

                all_embeddings.append(
                    [float(value) for value in vector]
                )

        dimensions = {len(vector) for vector in all_embeddings}

        if len(dimensions) != 1:
            raise EmbeddingError(
                "Embedding dimensions are inconsistent."
            )

        return all_embeddings
