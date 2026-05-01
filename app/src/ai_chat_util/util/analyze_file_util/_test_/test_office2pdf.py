from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_chat_util.core.common.config.runtime import Office2PDFSection
from ai_chat_util.util.analyze_file_util.office2pdf import Office2PDFUtil


def _config(**office2pdf: object) -> SimpleNamespace:
    return SimpleNamespace(office2pdf=Office2PDFSection.model_validate(office2pdf))


def test_is_conversion_available_for_libreoffice_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Office2PDFUtil, "try_find_libreoffice_binary", classmethod(lambda cls, explicit_path=None, configured_path=None: "/usr/bin/soffice"))

    assert Office2PDFUtil.is_conversion_available(config=_config(method="libreoffice_exec")) is True


def test_is_conversion_available_for_pywin32_is_false_on_non_windows() -> None:
    assert Office2PDFUtil.is_conversion_available(config=_config(method="pywin32")) is False


def test_create_pdf_from_document_file_rejects_unimplemented_pywin32(tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")

    with pytest.raises(NotImplementedError, match="pywin32"):
        Office2PDFUtil.create_pdf_from_document_file(
            input_path=source,
            config=_config(method="pywin32"),
        )