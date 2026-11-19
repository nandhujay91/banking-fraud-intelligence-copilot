
"""HTTP client for communicating with a local Ollama LLM."""

from typing import Any

import httpx


class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaClient:
    """Client for generating responses from a local Ollama model."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.transport = transport

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response from the configured Ollama model."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {"role": "system", "content": system_prompt}
            )

        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        try:
            with httpx.Client(
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama request timed out.") from exc

        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Could not communicate with Ollama: {exc}"
            ) from exc

        try:
            result = response.json()
            answer = result["message"]["content"]

        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError(
                "Ollama returned an invalid response."
            ) from exc

        if not isinstance(answer, str) or not answer.strip():
            raise OllamaError("Ollama returned an empty answer.")

        return answer.strip()
