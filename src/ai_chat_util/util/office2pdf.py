from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import os
import shutil
from pathlib import Path

class Office2PDFUtil:
    LIBREOFFICE_ENV_VAR = "LIBREOFFICE_PATH"

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
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(source),
        ]
        if extra_args:
            command.extend(extra_args)
        return command

    @classmethod
    def create_pdf_from_document(
        cls,
        input_path: str | Path,
        output_path: str | Path | None = None,
        libreoffice_path: str | Path | None = None,
        timeout: int | None = 120,
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

        command = cls._build_command(libreoffice_binary, source, output_dir)

        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
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
