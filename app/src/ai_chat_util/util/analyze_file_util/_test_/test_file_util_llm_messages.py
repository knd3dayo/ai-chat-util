from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_chat_util.core.common.config.runtime import Office2PDFSection
from ai_chat_util.util.analyze_file_util.file_util_llm_messages import FileUtilLLMMessages
from ai_chat_util.util.analyze_file_util.office2pdf import LibreOfficeUnoOffice2PDFUtil


def _config(**office2pdf: object) -> SimpleNamespace:
    return SimpleNamespace(office2pdf=Office2PDFSection.model_validate(office2pdf))


def test_create_office_content_from_file_uses_original_path_for_uno(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.xlsx"
    source.write_bytes(b"dummy")
    captured: dict[str, str] = {}

    llm_client = MagicMock()
    llm_client.get_config.return_value = _config(
        method=LibreOfficeUnoOffice2PDFUtil.METHOD_NAME,
        libreoffice_uno={
            "api_url": "http://127.0.0.1:2004",
        },
    )

    monkeypatch.setattr(LibreOfficeUnoOffice2PDFUtil, "is_available", classmethod(lambda cls, api_url: True))

    def _fake_convert(
        cls,
        input_path,
        output_path,
        *,
        api_url,
        print_orientation=None,
        fit_width_pages=None,
        fit_height_pages=None,
    ):
        captured["input_path"] = input_path
        captured["output_path"] = output_path
        captured["api_url"] = api_url
        Path(output_path).write_bytes(b"pdf")
        return Path(output_path)

    monkeypatch.setattr(
        LibreOfficeUnoOffice2PDFUtil,
        "create_pdf_from_document_file",
        classmethod(_fake_convert),
    )
    monkeypatch.setattr(FileUtilLLMMessages, "create_pdf_content_from_file", lambda self, file_path, detail="auto": [])
    monkeypatch.setattr(FileUtilLLMMessages, "create_text_content", lambda self, text: text)

    result = FileUtilLLMMessages(llm_client).create_office_content_from_file(str(source))

    assert len(result) == 1
    assert captured["input_path"] == str(source.resolve())
    assert captured["api_url"] == "http://127.0.0.1:2004"
    assert Path(captured["output_path"]).parent.parent == source.parent