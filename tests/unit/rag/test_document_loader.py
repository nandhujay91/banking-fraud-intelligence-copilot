
import pytest
from docx import Document
from pypdf import PdfWriter

from banking_fraud.rag.document_loader import (
    DocumentLoaderError,
    load_document,
)


def test_load_txt_document(tmp_path):
    file_path = tmp_path / "fraud_policy.txt"
    file_path.write_text(
        "Suspicious transactions must be investigated.",
        encoding="utf-8",
    )

    result = load_document(file_path)

    assert result == "Suspicious transactions must be investigated."


def test_load_docx_document(tmp_path):
    file_path = tmp_path / "fraud_policy.docx"

    document = Document()
    document.add_paragraph("Review suspicious transactions.")
    document.save(file_path)

    result = load_document(file_path)

    assert result == "Review suspicious transactions."


def test_load_pdf_document(tmp_path):
    file_path = tmp_path / "fraud_policy.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    with file_path.open("wb") as file:
        writer.write(file)

    with pytest.raises(
        DocumentLoaderError,
        match="no extractable text",
    ):
        load_document(file_path)


def test_reject_missing_document(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(DocumentLoaderError, match="not found"):
        load_document(file_path)


def test_reject_unsupported_extension(tmp_path):
    file_path = tmp_path / "fraud_policy.csv"
    file_path.write_text("transaction,amount", encoding="utf-8")

    with pytest.raises(
        DocumentLoaderError,
        match="Unsupported document format",
    ):
        load_document(file_path)


def test_reject_empty_txt_document(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(
        DocumentLoaderError,
        match="no extractable text",
    ):
        load_document(file_path)
