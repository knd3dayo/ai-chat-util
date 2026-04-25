from __future__ import annotations

from typing import Any

import fitz
from fitz import Document
from pdfminer.high_level import extract_text


def _extract_content(doc: Document) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        text = text.strip() if isinstance(text, str) else ""

        if text:
            results.append({"type": "text", "text": text})

        pix = page.get_pixmap()
        results.append({"type": "image", "bytes": pix.tobytes("png")})

    return results


def extract_content_from_bytes(pdf_bytes: bytes) -> list[dict[str, Any]]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return _extract_content(doc)


def extract_content_from_file(pdf_path: str) -> list[dict[str, Any]]:
    with fitz.open(pdf_path) as doc:
        return _extract_content(doc)


class PDFUtil:
    @classmethod
    def extract_text_from_pdf(cls, filename: str) -> str:
        """PDFファイルからテキストを抽出する。"""
        return extract_text(filename)


__all__ = [
    "PDFUtil",
    "extract_content_from_bytes",
    "extract_content_from_file",
]

