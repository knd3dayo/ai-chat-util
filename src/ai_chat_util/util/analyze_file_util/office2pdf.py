from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, cast

import psutil  # type: ignore[import-not-found]
import requests

from ai_chat_util.core.common.config.runtime import get_runtime_config


class _Office2PDFPyWin32Config(Protocol):
    @property
    def office_path(self) -> str | None: ...


class _Office2PDFLibreOfficeExecConfig(Protocol):
    @property
    def libreoffice_path(self) -> str | None: ...


class _Office2PDFLibreOfficeUnoConfig(Protocol):
    @property
    def api_url(self) -> str: ...


class _Office2PDFConfigProtocol(Protocol):
    @property
    def method(self) -> str: ...

    @property
    def pywin32(self) -> _Office2PDFPyWin32Config: ...

    @property
    def libreoffice_exec(self) -> _Office2PDFLibreOfficeExecConfig: ...

    @property
    def libreoffice_uno(self) -> _Office2PDFLibreOfficeUnoConfig: ...


class _HasOffice2PDFConfig(Protocol):
    @property
    def office2pdf(self) -> _Office2PDFConfigProtocol: ...


PrintOrientation = Literal["portrait", "landscape"]


def _resolve_target_path(
    input_path: str,
    output_path: str,
) -> tuple[Path, Path]:
    source = Path(input_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    source = source.resolve()

    output_candidate = Path(output_path).expanduser()
    if output_candidate.is_dir():
        target = output_candidate / source.with_suffix(".pdf").name
    else:
        target = output_candidate
    target.parent.mkdir(parents=True, exist_ok=True)
    return source, target


def _build_default_output_path(input_path: str) -> str:
    return str(Path(input_path).expanduser().with_suffix(".pdf"))


def _has_print_layout_override(
    print_orientation: PrintOrientation | None,
    fit_width_pages: int | None,
    fit_height_pages: int | None,
) -> bool:
    return print_orientation is not None or fit_width_pages is not None or fit_height_pages is not None


class Pywin32Office2PDFUtil:
    METHOD_NAME = "pywin32"
    _WORD_EXTENSIONS = {".doc", ".docx", ".docm", ".rtf"}
    _EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm", ".xlsb"}
    _POWERPOINT_EXTENSIONS = {".ppt", ".pptx", ".pptm"}

    @classmethod
    def _is_resolvable_binary(cls, candidate: str | Path | None) -> bool:
        if not candidate:
            return False

        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return True

        return shutil.which(str(candidate)) is not None

    @classmethod
    def is_available(cls, office_path: str | Path | None) -> bool:
        if os.name != "nt":
            return False
        if importlib.util.find_spec("win32com.client") is None:
            return False
        if office_path:
            return cls._is_resolvable_binary(office_path)
        return True

    @classmethod
    def _get_office_application_kind(cls, source: Path) -> str:
        ext = source.suffix.lower()
        if ext in cls._WORD_EXTENSIONS:
            return "word"
        if ext in cls._EXCEL_EXTENSIONS:
            return "excel"
        if ext in cls._POWERPOINT_EXTENSIONS:
            return "powerpoint"
        raise RuntimeError(f"Unsupported Office file extension for PDF conversion: {ext or source.name}")

    @classmethod
    def create_pdf_from_document_file(
        cls,
        input_path: str,
        output_path: str,
        *,
        office_path: str | Path | None = None,
        print_orientation: PrintOrientation | None = None,
        fit_width_pages: int | None = None,
        fit_height_pages: int | None = None,
    ) -> Path:
        if _has_print_layout_override(print_orientation, fit_width_pages, fit_height_pages):
            raise RuntimeError("Print layout overrides require office2pdf method 'libreoffice_uno'.")

        if os.name != "nt":
            raise RuntimeError("pywin32 conversion is only supported on Windows.")

        if importlib.util.find_spec("win32com.client") is None:
            raise RuntimeError("pywin32 conversion requires win32com.client to be installed.")

        if office_path and not cls._is_resolvable_binary(office_path):
            raise FileNotFoundError(f"Configured Office executable not found: {office_path}")

        import pythoncom  # type: ignore[import-not-found]
        from win32com.client import DispatchEx  # type: ignore[import-not-found]

        source, target = _resolve_target_path(input_path, output_path)
        app = None
        document = None
        app_kind = cls._get_office_application_kind(source)
        source_str = str(source)
        target_str = str(target)

        pythoncom.CoInitialize()
        try:
            if app_kind == "word":
                app = DispatchEx("Word.Application")
                app.Visible = False
                app.DisplayAlerts = 0
                document = app.Documents.Open(source_str, ReadOnly=True)
                document.ExportAsFixedFormat(OutputFileName=target_str, ExportFormat=17)
            elif app_kind == "excel":
                app = DispatchEx("Excel.Application")
                app.Visible = False
                app.DisplayAlerts = False
                document = app.Workbooks.Open(source_str, ReadOnly=True)
                document.ExportAsFixedFormat(0, target_str)
            elif app_kind == "powerpoint":
                app = DispatchEx("PowerPoint.Application")
                document = app.Presentations.Open(source_str, WithWindow=False)
                document.SaveAs(target_str, 32)
            else:
                raise RuntimeError(f"Unsupported Office application kind: {app_kind}")
        except Exception as exc:
            raise RuntimeError(f"pywin32 failed to convert {source.name} to PDF") from exc
        finally:
            if document is not None:
                try:
                    if app_kind == "powerpoint":
                        document.Close()
                    else:
                        document.Close(False)
                except Exception:
                    pass
            if app is not None:
                try:
                    app.Quit()
                except Exception:
                    pass
            pythoncom.CoUninitialize()

        if not target.exists():
            raise RuntimeError(f"pywin32 conversion did not produce expected PDF: {target}")

        return target.resolve()


class LibreOfficeUnoOffice2PDFUtil:

    METHOD_NAME = "libreoffice_uno"

    @classmethod
    def is_available(cls, api_url: str) -> bool:
        return bool(api_url.strip())

    @classmethod
    def create_pdf_from_document_file(
        cls,
        input_path: str,
        output_path: str,
        *,
        api_url: str,
        print_orientation: PrintOrientation | None = None,
        fit_width_pages: int | None = None,
        fit_height_pages: int | None = None,
    ) -> Path:
        if _has_print_layout_override(print_orientation, fit_width_pages, fit_height_pages):
            raise RuntimeError("Print layout overrides are not supported with office2pdf method 'libreoffice_uno'.")

        source, target = _resolve_target_path(input_path, output_path)
        resolved_api_url = api_url.strip()
        if not resolved_api_url:
            raise RuntimeError("office2pdf.libreoffice_uno.api_url is required.")

        try:
            with source.open("rb") as input_file:
                response = requests.post(
                    resolved_api_url.rstrip("/") + "/convert",
                    files={"file": (source.name, input_file, "application/octet-stream")},
                    data={"convert_to": "pdf"},
                    timeout=600,
                )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to call LibreOffice UNO API at {resolved_api_url}") from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"LibreOffice UNO API conversion failed with status {response.status_code}: {response.text.strip()}"
            )

        target.write_bytes(response.content)

        if not target.exists():
            raise RuntimeError(f"LibreOffice UNO API conversion did not produce expected PDF: {target}")

        return target.resolve()


class LibreOfficeExecOffice2PDFUtil:
    METHOD_NAME = "libreoffice_exec"
    DEFAULT_TIMEOUT_SECONDS = 600

    class _ConversionTimeout(RuntimeError):
        """Internal exception used to distinguish timeout paths."""

    @classmethod
    def _build_user_installation_arg(cls, user_profile_dir: Path) -> str:
        uri = user_profile_dir.resolve().as_uri()
        return f"-env:UserInstallation={uri}"

    @classmethod
    def _kill_process_tree(cls, proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return

        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            try:
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @classmethod
    def _kill_libreoffice_by_user_installation(cls, user_installation_arg: str) -> None:
        if not user_installation_arg:
            return

        marker = user_installation_arg.split("-env:")[-1]
        if not marker:
            return

        try:
            for process in psutil.process_iter(["cmdline"]):
                cmdline = process.info.get("cmdline")
                if not cmdline:
                    continue
                if marker not in " ".join(cmdline):
                    continue
                try:
                    process.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception:
            pass

    @classmethod
    def _resolve_produced_pdf_path(
        cls,
        expected_path: Path,
        output_dir: Path,
        *,
        start_time_epoch: float,
    ) -> Path | None:
        if expected_path.exists():
            return expected_path

        newest: Path | None = None
        newest_mtime = 0.0
        try:
            for candidate in output_dir.glob("*.pdf"):
                try:
                    stat_result = candidate.stat()
                except FileNotFoundError:
                    continue
                if stat_result.st_mtime >= start_time_epoch and stat_result.st_mtime >= newest_mtime:
                    newest_mtime = stat_result.st_mtime
                    newest = candidate
        except Exception:
            return None
        return newest

    @classmethod
    def _run_command_with_timeout_return_proc(
        cls,
        command: list[str],
        timeout: int | None,
    ) -> tuple[subprocess.CompletedProcess[bytes], subprocess.Popen[bytes]]:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            cls._kill_process_tree(proc)
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            raise cls._ConversionTimeout(f"LibreOffice conversion timed out after {timeout}s") from exc

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode,
                command,
                output=stdout,
                stderr=stderr,
            )

        return (subprocess.CompletedProcess(command, proc.returncode, stdout, stderr), proc)

    @classmethod
    def _build_command(
        cls,
        libreoffice_path: str | Path,
        source: Path,
        output_dir: Path,
        extra_args: Iterable[str] | None = None,
    ) -> list[str]:
        command = [
            str(libreoffice_path),
            "--headless",
            "--nologo",
            "--nolockcheck",
        ]
        if extra_args:
            command.extend(extra_args)
        command.extend([
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(source),
        ])
        return command

    @classmethod
    def create_pdf_from_document_file_via_libreoffice_exec(
        cls,
        input_path: str,
        output_path: str,
        libreoffice_path: str | Path,
        timeout: int | None = DEFAULT_TIMEOUT_SECONDS,
    ) -> Path:
        source, target = _resolve_target_path(input_path, output_path)

        output_dir = target.parent.resolve()
        expected_produced_path = output_dir / (source.stem + ".pdf")
        start_epoch = time.time()
        for p in (target, expected_produced_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        with tempfile.TemporaryDirectory() as lo_profile_dirname:
            lo_profile_dir = Path(lo_profile_dirname)
            user_installation_arg = cls._build_user_installation_arg(lo_profile_dir)
            command = cls._build_command(
                libreoffice_path,
                source,
                output_dir,
                extra_args=[user_installation_arg],
            )

            try:
                result, proc = cls._run_command_with_timeout_return_proc(command=command, timeout=timeout)
                produced_candidate = cls._resolve_produced_pdf_path(
                    expected_produced_path,
                    output_dir,
                    start_time_epoch=start_epoch,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
                raise RuntimeError(
                    f"LibreOffice failed to convert {source.name}: {stderr.strip()}"
                ) from exc
            except cls._ConversionTimeout as exc:
                try:
                    if "proc" in locals() and isinstance(locals().get("proc"), subprocess.Popen):
                        cls._kill_process_tree(locals()["proc"])
                except Exception:
                    pass
                cls._kill_libreoffice_by_user_installation(user_installation_arg)
                raise RuntimeError(f"LibreOffice conversion timed out after {timeout}s") from exc
            except FileNotFoundError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Failed to convert {source} to PDF") from exc

        if produced_candidate is None:
            stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
            stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
            raise RuntimeError(
                f"Expected PDF not found at {target}; stdout: {stdout.strip()} stderr: {stderr.strip()}"
            )

        produced_path = produced_candidate
        if produced_path.exists() and produced_path.resolve() != target.resolve():
            produced_path.rename(target)

        if not target.exists():
            stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
            stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
            raise RuntimeError(
                f"Expected PDF not found at {target}; stdout: {stdout.strip()} stderr: {stderr.strip()}"
            )

        return target.resolve()

    @classmethod
    def create_pdf_from_document_bytes(
        cls,
        input_bytes: bytes,
        output_path: str ,
        libreoffice_path: str ,
        timeout: int | None = DEFAULT_TIMEOUT_SECONDS,
        temp_dir: str | Path | None = None,
        input_filename: str | Path | None = None,
    ) -> Path:
        source_suffix = Path(input_filename).suffix if input_filename else ""
        source_name = f"input_document{source_suffix}" if source_suffix else "input_document"

        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            source_path = Path(tmpdirname) / source_name
            source_path.write_bytes(input_bytes)

            return cls.create_pdf_from_document_file(
                input_path=source_path.as_posix(),
                output_path=output_path,
                libreoffice_path=libreoffice_path,
                timeout=timeout,
            )

    @classmethod
    def create_pdf_from_document_file(
        cls,
        input_path: str,
        output_path: str,
        *,
        libreoffice_path: str | Path | None = None,
        timeout: int | None = DEFAULT_TIMEOUT_SECONDS,
    ) -> Path:
        libreoffice_binary = cls.find_libreoffice_binary(explicit_path=libreoffice_path)
        return cls.create_pdf_from_document_file_via_libreoffice_exec(
            input_path=input_path,
            output_path=output_path,
            libreoffice_path=libreoffice_binary,
            timeout=timeout,
        )

    @classmethod
    def try_find_libreoffice_binary(
        cls,
        explicit_path: str | Path | None = None,
        configured_path: str | Path | None = None,
    ) -> str | None:
        candidate = explicit_path or configured_path
        if candidate:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                return str(candidate_path)
            executable = shutil.which(str(candidate))
            if executable:
                return executable
            return None

        for binary in ("soffice", "libreoffice"):
            executable = shutil.which(binary)
            if executable:
                return executable

        return None

    @classmethod
    def find_libreoffice_binary(
        cls,
        explicit_path: str | Path | None = None,
        configured_path: str | Path | None = None,
    ) -> str:
        candidate = explicit_path or configured_path
        resolved = cls.try_find_libreoffice_binary(
            explicit_path=explicit_path,
            configured_path=configured_path,
        )
        if resolved:
            return resolved

        if candidate:
            raise FileNotFoundError(f"LibreOffice binary not found at {candidate}")

        raise RuntimeError(
            "LibreOffice binary not found. Set office2pdf.libreoffice_exec.libreoffice_path in ai-chat-util-config.yml "
            "or ensure LibreOffice is on PATH."
        )

__all__ = [
    "LibreOfficeExecOffice2PDFUtil",
    "LibreOfficeUnoOffice2PDFUtil",
    "Pywin32Office2PDFUtil",
]