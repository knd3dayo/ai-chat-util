from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable
import signal

import os
import shutil
import tempfile
import atexit

class Office2PDFUtil:
    LIBREOFFICE_ENV_VAR = "LIBREOFFICE_PATH"

    @classmethod
    def _build_user_installation_arg(cls, user_profile_dir: Path) -> str:
        """Build `-env:UserInstallation=...` for LibreOffice.

        LibreOffice expects a *file URL* (e.g. file:///... ), not a plain filesystem path.
        """
        # `as_uri()` requires an absolute path
        uri = user_profile_dir.resolve().as_uri()
        return f"-env:UserInstallation={uri}"

    @classmethod
    def _kill_process_tree(cls, proc: subprocess.Popen[bytes]) -> None:
        """Best-effort: kill the process *and its children*.

        LibreOffice may spawn child processes; on timeout we want to clean up the whole
        process tree.

        - Windows: `taskkill /T /F`
        - POSIX: `killpg` (requires `start_new_session=True` on Popen)
        """
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
    def _run_command_with_timeout(
        cls,
        command: list[str],
        timeout: int | None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a command and ensure it's cleaned up on timeout."""
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # POSIX: create a process group so we can killpg on timeout.
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            cls._kill_process_tree(proc)
            # Reap the process if possible.
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            raise RuntimeError(f"LibreOffice conversion timed out after {timeout}s") from exc

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode,
                command,
                output=stdout,
                stderr=stderr,
            )

        return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)

    @classmethod
    def _build_command(
        cls,
        libreoffice_binary: str,
        source: Path,
        output_dir: Path,
        extra_args: Iterable[str] | None = None
    ) -> list[str]:
        """
        Compose the LibreOffice CLI command used for PDF conversion.
        """
        command = [
            libreoffice_binary,
            "--headless",
            "--nologo",
            "--nolockcheck",
        ]
        if extra_args:
            # LibreOffice CLI options should appear before the document path.
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
        timeout: int | None = 120,
        temp_dir: str | Path | None = None,
    ) -> Path:
        tmpdir = tempfile.TemporaryDirectory(dir=temp_dir)
        atexit.register(tmpdir.cleanup)
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            source_path = Path(tmpdirname) / "input_document"
            with open(source_path, "wb") as source_file:
                source_file.write(input_bytes)

            return cls.create_pdf_from_document_file(
                input_path=source_path,
                output_path=output_path,
                libreoffice_path=libreoffice_path,
                timeout=timeout,
            )

    @classmethod
    def create_pdf_from_document_file(
        cls,
        input_path: str | Path,
        output_path: str | Path | None = None,
        libreoffice_path: str | Path | None = None,
        timeout: int | None = 240,
    ) -> Path:
        """
        Convert an Office document to PDF using LibreOffice.

        Args:
            input_path: Path to the Office document to convert.
            output_path: Target PDF path or directory. When omitted, a sibling PDF is created.
            libreoffice_path: Override path to the LibreOffice binary; otherwise use
                ``OFFICE2PDF_LIBREOFFICE`` env var or search PATH.
            timeout: Seconds to wait for LibreOffice. ``None`` disables the timeout.

        Returns:
            The resolved output PDF path.

        Raises:
            FileNotFoundError: When the input or LibreOffice binary cannot be found.
            RuntimeError: When LibreOffice fails to produce a PDF.
        """
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
        
        libreoffice_binary = cls.find_libreoffice_binary(libreoffice_path)
        output_dir = target.parent.resolve()

        # Isolate LibreOffice user profile per conversion to avoid profile locks and
        # lingering state across runs.
        with tempfile.TemporaryDirectory() as lo_profile_dirname:
            lo_profile_dir = Path(lo_profile_dirname)
            extra_args = [cls._build_user_installation_arg(lo_profile_dir)]

            command = cls._build_command(
                libreoffice_binary,
                source,
                output_dir,
                extra_args=extra_args,
            )

            try:
                result = cls._run_command_with_timeout(command=command, timeout=timeout)
            except subprocess.CalledProcessError as exc:  # pragma: no cover - raised paths tested
                stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
                raise RuntimeError(
                    f"LibreOffice failed to convert {source.name}: {stderr.strip()}"
                ) from exc
            except FileNotFoundError:
                raise
            except Exception as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(f"Failed to convert {source} to PDF") from exc

        # LibreOffice names the output after the source stem. Rename if the caller requested a custom
        # filename.
        produced_path = output_dir / (source.stem + ".pdf")
        if produced_path.exists() and produced_path != target:
            produced_path.rename(target)

        if not target.exists():
            stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
            stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
            raise RuntimeError(
                f"Expected PDF not found at {target}; stdout: {stdout.strip()} stderr: {stderr.strip()}"
            )

        return target.resolve()



    @classmethod
    def find_libreoffice_binary(cls, explicit_path: str | Path | None = None) -> str:
        """
        Resolve the LibreOffice executable path.

        Preference order:
        1) explicit path argument
        2) OFFICE2PDF_LIBREOFFICE environment variable
        3) ``soffice`` or ``libreoffice`` on PATH
        """
        candidate = explicit_path or os.getenv(cls.LIBREOFFICE_ENV_VAR)
        if candidate:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                return str(candidate_path)
            executable = shutil.which(str(candidate))
            if executable:
                return executable
            raise FileNotFoundError(f"LibreOffice binary not found at {candidate}")

        for binary in ("soffice", "libreoffice"):
            executable = shutil.which(binary)
            if executable:
                return executable

        raise RuntimeError(
            "LibreOffice binary not found. Set "
            f"{cls.LIBREOFFICE_ENV_VAR} or ensure LibreOffice is on PATH."
        )
