
"""Persistent JSON registry for ingested knowledge-base documents."""

import json
from pathlib import Path
from typing import Any


class DocumentRegistry:
    """Store and retrieve document metadata using a JSON file."""

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.registry_path.exists():
            self._save([])

    def _load(self) -> list[dict[str, Any]]:
        """Load registered documents from the JSON registry."""

        try:
            with self.registry_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                documents = json.load(file)

        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                "Could not read document registry."
            ) from exc

        if not isinstance(documents, list):
            raise TypeError(
                "Invalid document registry structure."
            )

        if not all(
            isinstance(document, dict)
            for document in documents
        ):
            raise TypeError(
                "Registry entries must be dictionaries."
            )

        return documents

    def _save(self, documents: list[dict[str, Any]]) -> None:
        """Save registered documents to the JSON registry."""

        with self.registry_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                documents,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def is_duplicate(self, sha256: str) -> bool:
        """Check whether a document checksum already exists."""

        if not sha256.strip():
            raise ValueError("SHA-256 checksum cannot be empty.")

        documents = self._load()

        return any(
            document.get("sha256") == sha256
            for document in documents
        )

    def register(self, metadata: dict[str, Any]) -> None:
        """Register document metadata if its checksum is new."""

        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary.")

        checksum = metadata.get("sha256")

        if not isinstance(checksum, str) or not checksum.strip():
            raise ValueError(
                "Document metadata must contain a valid SHA-256 checksum."
            )

        documents = self._load()

        if any(
            document.get("sha256") == checksum
            for document in documents
        ):
            raise ValueError("Duplicate document detected.")

        documents.append(metadata)
        self._save(documents)

    def get_all(self) -> list[dict[str, Any]]:
        """Return all registered documents."""

        return self._load()
