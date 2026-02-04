from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable
import signal
import time
import shlex

import os
import shutil
import tempfile

class Office2PDFUtil:
    LIBREOFFICE_ENV_VAR = "LIBREOFFICE_PATH"
    DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes

    class _ConversionTimeout(RuntimeError):
        """Internal exception used to distinguish timeout paths."""

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
    def _kill_libreoffice_by_user_installation(cls, user_installation_arg: str) -> None:
        """Best-effort: kill LibreOffice processes whose command line contains UserInstallation.

        LibreOffice sometimes returns to foreground quickly and continues conversion in the
        background. In such cases, killing only the originally spawned PID may not stop the
        conversion. We therefore additionally search for processes that were launched with the
        same `-env:UserInstallation=...` value and terminate them.

        `user_installation_arg` is expected to be the exact argument string, e.g.
        "-env:UserInstallation=file:///...".
        """
        if not user_installation_arg:
            return

        # We match by the unique file URI (UserInstallation=file:///...) to avoid killing
        # unrelated LibreOffice instances.
        marker = user_installation_arg.split("-env:")[-1]  # "UserInstallation=..."
        if not marker:
            return

        if os.name == "nt":
            # Use PowerShell to query Win32_Process.CommandLine and stop matching processes.
            # Note: this requires no admin privileges for processes owned by the same user.
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
                # ignore best-effort failures
                pass
            return

        # POSIX fallback: parse `ps` output.
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
                # pid is first token
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
        """Wait until a PDF exists and appears stable.

        Stability definition: file exists and size is unchanged for `stable_seconds`.
        """
        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + timeout_seconds

        last_size: int | None = None
        last_change_t: float | None = None

        # Fallback candidate: when LibreOffice picks a slightly different name.
        # We'll search for the newest pdf modified after conversion start.
        start_epoch = start_time_epoch or time.time()

        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("PDF generation wait timed out")

            candidate = expected_path
            if not candidate.exists():
                # Fallback: find a pdf created/updated after start.
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
                    else:
                        if last_change_t is not None and (now - last_change_t) >= stable_seconds:
                            return candidate

            time.sleep(poll_interval)

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
    def _run_command_with_timeout_return_proc(
        cls,
        command: list[str],
        timeout: int | None,
    ) -> tuple[subprocess.CompletedProcess[bytes], subprocess.Popen[bytes]]:
        """Run a command, returning both CompletedProcess and the Popen instance.

        This is useful when we need the PID for additional clean-up on timeout.
        """
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
        timeout: int | None = DEFAULT_TIMEOUT_SECONDS,
        temp_dir: str | Path | None = None,
    ) -> Path:
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
        timeout: int | None = DEFAULT_TIMEOUT_SECONDS,
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

        expected_produced_path = output_dir / (source.stem + ".pdf")
        start_epoch = time.time()
        start_mono = time.monotonic()

        # Avoid false positives when a PDF from a previous run already exists.
        for p in (target, expected_produced_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                # best-effort: don't fail conversion just because cleanup couldn't happen
                pass

        # Isolate LibreOffice user profile per conversion to avoid profile locks and
        # lingering state across runs.
        with tempfile.TemporaryDirectory() as lo_profile_dirname:
            lo_profile_dir = Path(lo_profile_dirname)
            user_installation_arg = cls._build_user_installation_arg(lo_profile_dir)
            extra_args = [user_installation_arg]

            command = cls._build_command(
                libreoffice_binary,
                source,
                output_dir,
                extra_args=extra_args,
            )

            try:
                # Step 1: wait for the soffice command itself (it might return quickly).
                result, proc = cls._run_command_with_timeout_return_proc(command=command, timeout=timeout)

                # Step 2: wait for the PDF to be produced (some environments convert in background).
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
            except subprocess.CalledProcessError as exc:  # pragma: no cover - raised paths tested
                stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
                raise RuntimeError(
                    f"LibreOffice failed to convert {source.name}: {stderr.strip()}"
                ) from exc
            except (TimeoutError, cls._ConversionTimeout) as exc:
                # Ensure we stop any lingering background LibreOffice processes.
                try:
                    # We don't always have `proc` in scope if an exception happened before spawn.
                    if "proc" in locals() and isinstance(locals().get("proc"), subprocess.Popen):
                        cls._kill_process_tree(locals()["proc"])  # type: ignore[index]
                except Exception:
                    pass
                cls._kill_libreoffice_by_user_installation(user_installation_arg)
                raise RuntimeError(f"LibreOffice conversion timed out after {timeout}s") from exc
            except FileNotFoundError:
                raise
            except Exception as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(f"Failed to convert {source} to PDF") from exc

        # LibreOffice names the output after the source stem. Rename if the caller requested a custom
        # filename.
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
