from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable

from ai_chat_util.core.common.config.runtime import get_runtime_config


class Office2PDFUtil:
    DEFAULT_TIMEOUT_SECONDS = 600
    _WORD_EXTENSIONS = {".doc", ".docx", ".docm", ".rtf"}
    _EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm", ".xlsb"}
    _POWERPOINT_EXTENSIONS = {".ppt", ".pptx", ".pptm"}

    class _ConversionTimeout(RuntimeError):
        """Internal exception used to distinguish timeout paths."""

    @classmethod
    def _get_effective_office2pdf_config(cls, config: object | None = None):
        effective_config = config if config is not None else get_runtime_config()
        return effective_config.office2pdf

    @classmethod
    def get_selected_method(cls, config: object | None = None) -> str:
        return str(cls._get_effective_office2pdf_config(config).method)

    @classmethod
    def _is_resolvable_binary(cls, candidate: str | Path | None) -> bool:
        if not candidate:
            return False

        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return True

        return shutil.which(str(candidate)) is not None

    @classmethod
    def _is_pywin32_available(cls, office_path: str | Path | None) -> bool:
        if os.name != "nt":
            return False
        if importlib.util.find_spec("win32com.client") is None:
            return False
        if office_path:
            return cls._is_resolvable_binary(office_path)
        return True

    @classmethod
    def _is_libreoffice_uno_available(cls) -> bool:
        return importlib.util.find_spec("uno") is not None

    @classmethod
    def is_conversion_available(
        cls,
        *,
        config: object | None = None,
        libreoffice_path: str | Path | None = None,
        configured_libreoffice_path: str | Path | None = None,
    ) -> bool:
        office2pdf_config = cls._get_effective_office2pdf_config(config)
        method = cls.get_selected_method(config)

        if method == "libreoffice_exec":
            return cls.try_find_libreoffice_binary(
                explicit_path=libreoffice_path,
                configured_path=(
                    configured_libreoffice_path or office2pdf_config.libreoffice_exec.libreoffice_path
                ),
            ) is not None

        if method == "pywin32":
            return cls._is_pywin32_available(office2pdf_config.pywin32.office_path)

        if method == "libreoffice_uno":
            return cls._is_libreoffice_uno_available()

        return False

    @classmethod
    def _resolve_target_path(
        cls,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> tuple[Path, Path]:
        source = Path(input_path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Input file not found: {source}")
        source = source.resolve()

        if output_path is None:
            target = source.with_suffix(".pdf")
        else:
            output_candidate = Path(output_path).expanduser()
            if output_candidate.is_dir():
                target = output_candidate / source.with_suffix(".pdf").name
            else:
                target = output_candidate
        target.parent.mkdir(parents=True, exist_ok=True)
        return source, target

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
    def _create_pdf_via_pywin32(
        cls,
        source: Path,
        target: Path,
        office_path: str | Path | None,
    ) -> Path:
        if os.name != "nt":
            raise RuntimeError("pywin32 conversion is only supported on Windows.")

        if importlib.util.find_spec("win32com.client") is None:
            raise RuntimeError("pywin32 conversion requires win32com.client to be installed.")

        if office_path and not cls._is_resolvable_binary(office_path):
            raise FileNotFoundError(f"Configured Office executable not found: {office_path}")

        import pythoncom  # type: ignore[import-not-found]
        from win32com.client import DispatchEx  # type: ignore[import-not-found]

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

    @classmethod
    def _build_uno_connection_string(cls, connection_string: str | None, host: str, port: int) -> str:
        if connection_string:
            return connection_string.strip()
        return f"uno:socket,host={host},port={port};urp;StarOffice.ComponentContext"

    @classmethod
    def _build_uno_property(cls, name: str, value: object):
        from com.sun.star.beans import PropertyValue  # type: ignore[import-not-found]

        prop = PropertyValue()
        prop.Name = name
        prop.Value = value
        return prop

    @classmethod
    def _detect_uno_pdf_filter(cls, document: object) -> str:
        if document.supportsService("com.sun.star.text.GenericTextDocument"):
            return "writer_pdf_Export"
        if document.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
            return "calc_pdf_Export"
        if document.supportsService("com.sun.star.presentation.PresentationDocument"):
            return "impress_pdf_Export"
        if document.supportsService("com.sun.star.drawing.DrawingDocument"):
            return "draw_pdf_Export"
        return "writer_pdf_Export"

    @classmethod
    def _close_uno_document(cls, document: object) -> None:
        try:
            document.close(True)
            return
        except Exception:
            pass
        try:
            document.dispose()
        except Exception:
            pass

    @classmethod
    def _create_pdf_via_libreoffice_uno(
        cls,
        source: Path,
        target: Path,
        *,
        host: str,
        port: int,
        connection_string: str | None,
    ) -> Path:
        if importlib.util.find_spec("uno") is None:
            raise RuntimeError("LibreOffice UNO conversion requires the 'uno' Python module.")

        import uno  # type: ignore[import-not-found]

        local_context = uno.getComponentContext()
        resolver = local_context.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local_context
        )
        resolved_connection_string = cls._build_uno_connection_string(connection_string, host, port)

        try:
            remote_context = resolver.resolve(resolved_connection_string)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to LibreOffice UNO server using '{resolved_connection_string}'"
            ) from exc

        document = None
        try:
            desktop = remote_context.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", remote_context
            )
            input_url = uno.systemPathToFileUrl(str(source))
            output_url = uno.systemPathToFileUrl(str(target))
            load_props = (cls._build_uno_property("Hidden", True),)
            document = desktop.loadComponentFromURL(input_url, "_blank", 0, load_props)
            export_props = (
                cls._build_uno_property("FilterName", cls._detect_uno_pdf_filter(document)),
            )
            document.storeToURL(output_url, export_props)
        except Exception as exc:
            raise RuntimeError(f"LibreOffice UNO failed to convert {source.name} to PDF") from exc
        finally:
            if document is not None:
                cls._close_uno_document(document)

        if not target.exists():
            raise RuntimeError(f"LibreOffice UNO conversion did not produce expected PDF: {target}")

        return target.resolve()

    @classmethod
    def _raise_unsupported_method(cls, method: str) -> None:
        raise ValueError(f"Unsupported office2pdf method: {method}")

    @classmethod
    def _build_user_installation_arg(cls, user_profile_dir: Path) -> str:
        uri = user_profile_dir.resolve().as_uri()
        return f"-env:UserInstallation={uri}"

    @classmethod
    def _kill_process_tree(cls, proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return

        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return

        try:
            os.killpg(proc.pid, signal.SIGKILL)
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

        if os.name == "nt":
            ps = (
                "$m = "
                + shlex.quote(marker)
                + ";"
                "Get-CimInstance Win32_Process "
                "| Where-Object { $_.CommandLine -and $_.CommandLine -like ('*' + $m + '*') } "
                "| ForEach-Object { "
                "  try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {} "
                "}"
            )
            try:
                subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        ps,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass
            return

        try:
            res = subprocess.run(
                ["ps", "-eo", "pid,args"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            text = res.stdout.decode(errors="ignore")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                pid_s, args = parts
                if marker in args:
                    try:
                        os.kill(int(pid_s), signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            pass

    @classmethod
    def _wait_for_pdf(
        cls,
        expected_path: Path,
        output_dir: Path,
        *,
        timeout_seconds: float | None,
        stable_seconds: float = 1.0,
        poll_interval: float = 0.25,
        start_time_epoch: float | None = None,
    ) -> Path:
        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + timeout_seconds

        last_size: int | None = None
        last_change_t: float | None = None
        start_epoch = start_time_epoch or time.time()

        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("PDF generation wait timed out")

            candidate = expected_path
            if not candidate.exists():
                newest: Path | None = None
                newest_mtime = 0.0
                try:
                    for p in output_dir.glob("*.pdf"):
                        try:
                            st = p.stat()
                        except FileNotFoundError:
                            continue
                        if st.st_mtime >= start_epoch and st.st_mtime >= newest_mtime:
                            newest_mtime = st.st_mtime
                            newest = p
                except Exception:
                    newest = None
                if newest is not None:
                    candidate = newest

            if candidate.exists():
                try:
                    size = candidate.stat().st_size
                except FileNotFoundError:
                    size = None

                if size is not None:
                    now = time.monotonic()
                    if last_size != size:
                        last_size = size
                        last_change_t = now
                    elif last_change_t is not None and (now - last_change_t) >= stable_seconds:
                        return candidate

            time.sleep(poll_interval)

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
        libreoffice_binary: str,
        source: Path,
        output_dir: Path,
        extra_args: Iterable[str] | None = None,
    ) -> list[str]:
        command = [
            libreoffice_binary,
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
    def create_pdf_from_document_bytes(
        cls,
        input_bytes: bytes,
        output_path: str | Path | None = None,
        libreoffice_path: str | Path | None = None,
        configured_libreoffice_path: str | Path | None = None,
        timeout: int | None = DEFAULT_TIMEOUT_SECONDS,
        temp_dir: str | Path | None = None,
        config: object | None = None,
    ) -> Path:
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            source_path = Path(tmpdirname) / "input_document"
            with open(source_path, "wb") as source_file:
                source_file.write(input_bytes)

            return cls.create_pdf_from_document_file(
                input_path=source_path,
                output_path=output_path,
                libreoffice_path=libreoffice_path,
                configured_libreoffice_path=configured_libreoffice_path,
                timeout=timeout,
                config=config,
            )

    @classmethod
    def create_pdf_from_document_file(
        cls,
        input_path: str | Path,
        output_path: str | Path | None = None,
        libreoffice_path: str | Path | None = None,
        configured_libreoffice_path: str | Path | None = None,
        timeout: int | None = DEFAULT_TIMEOUT_SECONDS,
        config: object | None = None,
    ) -> Path:
        source, target = cls._resolve_target_path(input_path, output_path)

        office2pdf_config = cls._get_effective_office2pdf_config(config)
        method = cls.get_selected_method(config)

        if method == "pywin32":
            return cls._create_pdf_via_pywin32(source, target, office2pdf_config.pywin32.office_path)

        if method == "libreoffice_uno":
            return cls._create_pdf_via_libreoffice_uno(
                source,
                target,
                host=office2pdf_config.libreoffice_uno.host,
                port=office2pdf_config.libreoffice_uno.port,
                connection_string=office2pdf_config.libreoffice_uno.connection_string,
            )

        if method != "libreoffice_exec":
            cls._raise_unsupported_method(method)

        libreoffice_binary = cls.find_libreoffice_binary(
            explicit_path=libreoffice_path,
            configured_path=(configured_libreoffice_path or office2pdf_config.libreoffice_exec.libreoffice_path),
        )
        output_dir = target.parent.resolve()
        expected_produced_path = output_dir / (source.stem + ".pdf")
        start_epoch = time.time()
        start_mono = time.monotonic()

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
                libreoffice_binary,
                source,
                output_dir,
                extra_args=[user_installation_arg],
            )

            try:
                result, proc = cls._run_command_with_timeout_return_proc(command=command, timeout=timeout)
                if timeout is None:
                    remaining = None
                else:
                    elapsed = time.monotonic() - start_mono
                    remaining = max(0.0, float(timeout) - elapsed)

                produced_candidate = cls._wait_for_pdf(
                    expected_produced_path,
                    output_dir,
                    timeout_seconds=remaining,
                    start_time_epoch=start_epoch,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
                raise RuntimeError(
                    f"LibreOffice failed to convert {source.name}: {stderr.strip()}"
                ) from exc
            except (TimeoutError, cls._ConversionTimeout) as exc:
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


__all__ = ["Office2PDFUtil"]