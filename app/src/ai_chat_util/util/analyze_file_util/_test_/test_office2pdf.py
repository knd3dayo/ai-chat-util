from pathlib import Path
import subprocess
from types import SimpleNamespace
import asyncio
from unittest.mock import MagicMock

import pytest
import psutil

from ai_chat_util.core.analysis import analyze_pdf as analyze_pdf_mod
from ai_chat_util.core.common.config.runtime import Office2PDFSection
from ai_chat_util.util.analyze_file_util.office2pdf import (
    LibreOfficeExecOffice2PDFUtil,
    LibreOfficeUnoOffice2PDFUtil,
    Pywin32Office2PDFUtil,
    _build_default_output_path,
)


def _config(**office2pdf: object) -> SimpleNamespace:
    return SimpleNamespace(office2pdf=Office2PDFSection.model_validate(office2pdf))


def test_try_find_libreoffice_binary_for_libreoffice_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LibreOfficeExecOffice2PDFUtil, "try_find_libreoffice_binary", classmethod(lambda cls, explicit_path=None, configured_path=None: "/usr/bin/soffice"))

    assert LibreOfficeExecOffice2PDFUtil.try_find_libreoffice_binary() == "/usr/bin/soffice"


def test_is_conversion_available_for_pywin32_is_false_on_non_windows() -> None:
    assert Pywin32Office2PDFUtil.is_available(None) is False


def test_create_pdf_from_document_file_rejects_pywin32_on_non_windows(tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")

    with pytest.raises(RuntimeError, match="only supported on Windows"):
        Pywin32Office2PDFUtil.create_pdf_from_document_file(
            input_path=str(source),
            output_path=str(source.with_suffix(".pdf")),
        )


def test_create_pdf_from_document_file_rejects_uno_without_module(tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")

    with pytest.raises(RuntimeError, match="api_url"):
        LibreOfficeUnoOffice2PDFUtil.create_pdf_from_document_file(
            input_path=str(source),
            output_path=str(source.with_suffix(".pdf")),
            api_url="",
        )


def test_create_pdf_from_document_file_rejects_layout_override_for_non_uno(tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")

    with pytest.raises(RuntimeError, match="libreoffice_uno"):
        Pywin32Office2PDFUtil.create_pdf_from_document_file(
            input_path=str(source),
            output_path=str(source.with_suffix(".pdf")),
            print_orientation="landscape",
        )


def test_create_pdf_from_document_file_via_exec_resolves_binary_and_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")
    captured: dict[str, object] = {}

    def _fake_find_binary(cls, explicit_path=None, configured_path=None):
        captured["libreoffice_path"] = explicit_path
        return "/usr/bin/soffice"

    def _fake_convert(cls, input_path, output_path, libreoffice_path, timeout=LibreOfficeExecOffice2PDFUtil.DEFAULT_TIMEOUT_SECONDS):
        captured["source"] = Path(input_path)
        resolved_output = Path(output_path) if output_path is not None else Path(input_path).with_suffix(".pdf")
        captured["target"] = resolved_output
        captured["resolved_binary"] = libreoffice_path
        resolved_output.write_bytes(b"pdf")
        return resolved_output

    monkeypatch.setattr(LibreOfficeExecOffice2PDFUtil, "find_libreoffice_binary", classmethod(_fake_find_binary))
    monkeypatch.setattr(LibreOfficeExecOffice2PDFUtil, "create_pdf_from_document_file_via_libreoffice_exec", classmethod(_fake_convert))

    result = LibreOfficeExecOffice2PDFUtil.create_pdf_from_document_file(
        input_path=str(source),
        output_path=_build_default_output_path(str(source)),
        libreoffice_path="/custom/soffice",
    )

    assert result == source.with_suffix(".pdf")
    assert captured["source"] == source.resolve()
    assert captured["target"] == source.with_suffix(".pdf")
    assert captured["libreoffice_path"] == "/custom/soffice"
    assert captured["resolved_binary"] == "/usr/bin/soffice"


def test_create_pdf_from_document_file_via_exec_uses_generated_pdf_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")
    target = tmp_path / "sample.pdf"

    def _fake_run(cls, command, timeout):
        target.write_bytes(b"pdf")
        proc = MagicMock()
        proc.pid = 123
        return subprocess.CompletedProcess(command, 0, b"", b""), proc

    monkeypatch.setattr(
        LibreOfficeExecOffice2PDFUtil,
        "_run_command_with_timeout_return_proc",
        classmethod(_fake_run),
    )

    result = LibreOfficeExecOffice2PDFUtil.create_pdf_from_document_file_via_libreoffice_exec(
        input_path=str(source),
        output_path=str(target),
        libreoffice_path="/usr/bin/soffice",
        timeout=5,
    )

    assert result == target


def test_create_pdf_from_document_file_calls_uno_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200
        text = ""
        content = b"pdf"

    def _fake_post(url, files, data, timeout):
        captured["url"] = url
        captured["filename"] = files["file"][0]
        captured["convert_to"] = data["convert_to"]
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("ai_chat_util.util.analyze_file_util.office2pdf.requests.post", _fake_post)

    result = LibreOfficeUnoOffice2PDFUtil.create_pdf_from_document_file(
        input_path=str(source),
        output_path=_build_default_output_path(str(source)),
        api_url="http://127.0.0.1:2004",
    )

    assert result == source.with_suffix(".pdf")
    assert captured["url"] == "http://127.0.0.1:2004/convert"
    assert captured["filename"] == "sample.docx"
    assert captured["convert_to"] == "pdf"
    assert captured["timeout"] == 600


def test_create_pdf_from_document_bytes_preserves_input_suffix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "result.pdf"
    captured: dict[str, object] = {}

    def _fake_convert(
        cls,
        input_path,
        output_path=None,
        libreoffice_path=None,
        timeout=LibreOfficeExecOffice2PDFUtil.DEFAULT_TIMEOUT_SECONDS,
    ):
        captured["input_path"] = Path(input_path)
        captured["output_path"] = output_path
        Path(output_path).write_bytes(b"pdf")
        return Path(output_path)

    monkeypatch.setattr(LibreOfficeExecOffice2PDFUtil, "create_pdf_from_document_file", classmethod(_fake_convert))

    result = LibreOfficeExecOffice2PDFUtil.create_pdf_from_document_bytes(
        input_bytes=b"dummy",
        output_path=str(output_path),
        libreoffice_path="/usr/bin/soffice",
        input_filename="sample.docx",
    )

    assert result == output_path
    assert captured["input_path"] is not None
    assert Path(captured["input_path"]).suffix == ".docx"


def test_create_pdf_from_document_file_rejects_layout_override_for_uno_api(tmp_path: Path) -> None:
    source = tmp_path / "sample.xlsx"
    source.write_bytes(b"dummy")

    with pytest.raises(RuntimeError, match="not supported"):
        LibreOfficeUnoOffice2PDFUtil.create_pdf_from_document_file(
            input_path=str(source),
            output_path=str(source.with_suffix(".pdf")),
            api_url="http://127.0.0.1:2004",
            print_orientation="landscape",
        )


def test_convert_office_files_to_pdf_uses_runtime_method_pywin32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")
    config = _config(method="pywin32")

    def _fake_convert(cls, input_path, output_path=None, *, office_path=None, print_orientation=None, fit_width_pages=None, fit_height_pages=None):
        assert office_path is None
        target = Path(output_path) if output_path is not None else Path(input_path).with_suffix(".pdf")
        target.write_bytes(b"pdf")
        return target

    monkeypatch.setattr(analyze_pdf_mod, "get_runtime_config", lambda: config)
    monkeypatch.setattr(Pywin32Office2PDFUtil, "create_pdf_from_document_file", classmethod(_fake_convert))

    result = asyncio.run(analyze_pdf_mod.convert_office_files_to_pdf([str(source)]))

    assert result == [{"source_path": str(source), "pdf_path": str(source.with_suffix('.pdf'))}]


def test_convert_office_files_to_pdf_uses_runtime_method_uno(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "sample.docx"
    source.write_bytes(b"dummy")
    config = _config(method="libreoffice_uno")

    def _fake_convert(cls, input_path, output_path=None, *, api_url, print_orientation=None, fit_width_pages=None, fit_height_pages=None):
        assert api_url == "http://127.0.0.1:2004"
        target = Path(output_path) if output_path is not None else Path(input_path).with_suffix(".pdf")
        target.write_bytes(b"pdf")
        return target

    monkeypatch.setattr(analyze_pdf_mod, "get_runtime_config", lambda: config)
    monkeypatch.setattr(LibreOfficeUnoOffice2PDFUtil, "create_pdf_from_document_file", classmethod(_fake_convert))

    result = asyncio.run(analyze_pdf_mod.convert_office_files_to_pdf([str(source)]))

    assert result == [{"source_path": str(source), "pdf_path": str(source.with_suffix('.pdf'))}]


def test_kill_libreoffice_by_user_installation_uses_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    matching_process = MagicMock()
    matching_process.info = {"cmdline": ["soffice", "-env:UserInstallation=file:///tmp/profile-a"]}
    non_matching_process = MagicMock()
    non_matching_process.info = {"cmdline": ["python", "worker.py"]}

    monkeypatch.setattr(psutil, "process_iter", lambda attrs: iter([matching_process, non_matching_process]))

    LibreOfficeExecOffice2PDFUtil._kill_libreoffice_by_user_installation("-env:UserInstallation=file:///tmp/profile-a")

    matching_process.kill.assert_called_once_with()
    non_matching_process.kill.assert_not_called()


def test_kill_libreoffice_by_user_installation_ignores_process_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    vanished_process = MagicMock()
    vanished_process.info = {"cmdline": ["soffice", "-env:UserInstallation=file:///tmp/profile-b"]}
    vanished_process.kill.side_effect = psutil.NoSuchProcess(1234)

    monkeypatch.setattr(psutil, "process_iter", lambda attrs: iter([vanished_process]))

    LibreOfficeExecOffice2PDFUtil._kill_libreoffice_by_user_installation("-env:UserInstallation=file:///tmp/profile-b")

    vanished_process.kill.assert_called_once_with()


def test_kill_process_tree_uses_psutil_children(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = MagicMock()
    proc.pid = 4321
    proc.poll.return_value = None

    child_a = MagicMock()
    child_b = MagicMock()
    parent = MagicMock()
    parent.children.return_value = [child_a, child_b]

    monkeypatch.setattr(psutil, "Process", lambda pid: parent)

    LibreOfficeExecOffice2PDFUtil._kill_process_tree(proc)

    parent.children.assert_called_once_with(recursive=True)
    child_b.kill.assert_called_once_with()
    child_a.kill.assert_called_once_with()
    parent.kill.assert_called_once_with()
    proc.kill.assert_not_called()


def test_kill_process_tree_falls_back_to_proc_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = MagicMock()
    proc.pid = 9876
    proc.poll.return_value = None

    monkeypatch.setattr(psutil, "Process", lambda pid: (_ for _ in ()).throw(RuntimeError("boom")))

    LibreOfficeExecOffice2PDFUtil._kill_process_tree(proc)

    proc.kill.assert_called_once_with()