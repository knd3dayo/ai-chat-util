from pathlib import Path
from types import SimpleNamespace
import asyncio

import pytest

from ai_chat_util.core.analysis import analyze_pdf as analyze_pdf_mod
from ai_chat_util.core.common.config.runtime import Office2PDFSection
from ai_chat_util.util.analyze_file_util.office2pdf import Office2PDFUtil


def _config(**office2pdf: object) -> SimpleNamespace:
    return SimpleNamespace(office2pdf=Office2PDFSection.model_validate(office2pdf))


def test_is_conversion_available_for_libreoffice_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Office2PDFUtil, "try_find_libreoffice_binary", classmethod(lambda cls, explicit_path=None, configured_path=None: "/usr/bin/soffice"))

    assert Office2PDFUtil.is_conversion_available(config=_config(method="libreoffice_exec")) is True


def test_is_conversion_available_for_pywin32_is_false_on_non_windows() -> None:
    assert Office2PDFUtil.is_conversion_available(config=_config(method="pywin32")) is False


def test_create_pdf_from_document_file_rejects_pywin32_on_non_windows(tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")

    with pytest.raises(RuntimeError, match="only supported on Windows"):
        Office2PDFUtil.create_pdf_from_document_file(
            input_path=source,
            config=_config(method="pywin32"),
        )


def test_create_pdf_from_document_file_rejects_uno_without_module(tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")

    with pytest.raises(RuntimeError, match="UNO"):
        Office2PDFUtil.create_pdf_from_document_file(
            input_path=source,
            config=_config(method="libreoffice_uno"),
        )


def test_convert_office_files_to_pdf_uses_runtime_method_pywin32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")
    config = _config(method="pywin32")

    def _fake_convert(cls, input_path, output_path=None, libreoffice_path=None, configured_libreoffice_path=None, timeout=Office2PDFUtil.DEFAULT_TIMEOUT_SECONDS, config=None):
        assert config is not None
        assert config.office2pdf.method == "pywin32"
        target = Path(output_path) if output_path is not None else Path(input_path).with_suffix(".pdf")
        target.write_bytes(b"pdf")
        return target

    monkeypatch.setattr(analyze_pdf_mod, "get_runtime_config", lambda: config)
    monkeypatch.setattr(Office2PDFUtil, "create_pdf_from_document_file", classmethod(_fake_convert))

    result = asyncio.run(analyze_pdf_mod.convert_office_files_to_pdf([str(source)]))

    assert result == [{"source_path": str(source), "pdf_path": str(source.with_suffix('.pdf'))}]


def test_convert_office_files_to_pdf_uses_runtime_method_uno(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")
    config = _config(method="libreoffice_uno")

    def _fake_convert(cls, input_path, output_path=None, libreoffice_path=None, configured_libreoffice_path=None, timeout=Office2PDFUtil.DEFAULT_TIMEOUT_SECONDS, config=None):
        assert config is not None
        assert config.office2pdf.method == "libreoffice_uno"
        target = Path(output_path) if output_path is not None else Path(input_path).with_suffix(".pdf")
        target.write_bytes(b"pdf")
        return target

    monkeypatch.setattr(analyze_pdf_mod, "get_runtime_config", lambda: config)
    monkeypatch.setattr(Office2PDFUtil, "create_pdf_from_document_file", classmethod(_fake_convert))

    result = asyncio.run(analyze_pdf_mod.convert_office_files_to_pdf([str(source)]))

    assert result == [{"source_path": str(source), "pdf_path": str(source.with_suffix('.pdf'))}]