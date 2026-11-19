
"""Load and extract text from PDF, DOCX, and TXT documents."""

from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentLoaderError(Exception):
    """Raised when a document cannot be loaded."""


def load_document(file_path: str | Path) -> str:
    """Extract text from a supported document."""

    path = Path(file_path)

    if not path.exists():
        raise DocumentLoaderError(f"Document not found: {path}")

    if not path.is_file():
        raise DocumentLoaderError(f"Path is not a file: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoaderError(
            f"Unsupported document format: {extension}"
        )

    try:
        if extension == ".pdf":
            text = _load_pdf(path)
        elif extension == ".docx":
            text = _load_docx(path)
        else:
            text = _load_txt(path)

    except DocumentLoaderError:
        raise
    except Exception as exc:
        raise DocumentLoaderError(
            f"Could not read document: {path.name}"
        ) from exc

    cleaned_text = text.strip()

    if not cleaned_text:
        raise DocumentLoaderError(
            f"Document contains no extractable text: {path.name}"
        )

    return cleaned_text


def _load_pdf(path: Path) -> str:
    """Extract text from all pages of a PDF."""

    reader = PdfReader(path)

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


def _load_docx(path: Path) -> str:
    """Extract text from paragraphs in a DOCX document."""

    document = Document(path)

    return "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )


def _load_txt(path: Path) -> str:
    """Read a UTF-8 text document."""

    return path.read_text(encoding="utf-8")
