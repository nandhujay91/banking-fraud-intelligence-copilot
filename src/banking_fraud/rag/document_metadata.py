
"""Create traceable metadata for knowledge-base documents."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def calculate_sha256(file_path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    path = Path(file_path)
    checksum = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)

    return checksum.hexdigest()


def create_document_metadata(file_path: str | Path) -> dict:
    """Create metadata describing a source document."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return {
        "document_id": str(uuid4()),
        "filename": path.name,
        "file_format": path.suffix.lower().lstrip("."),
        "source_path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
        "created_at": datetime.now(UTC).isoformat(),
    }
