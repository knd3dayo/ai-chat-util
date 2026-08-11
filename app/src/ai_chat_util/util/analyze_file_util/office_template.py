from __future__ import annotations

from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile


class OfficeTemplateUtil:
    @classmethod
    def apply_template(cls, source_path: str | Path, output_path: str | Path, replacements: dict[str, str]) -> str:
        source = Path(source_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if not source.exists():
            raise FileNotFoundError(source)

        try:
            with ZipFile(source, "r") as src_zip, ZipFile(output, "w") as dst_zip:
                for info in src_zip.infolist():
                    data = src_zip.read(info.filename)
                    if info.filename.endswith(".xml"):
                        text = data.decode("utf-8", errors="ignore")
                        for key, value in replacements.items():
                            text = text.replace(key, value)
                        data = text.encode("utf-8")
                    dst_zip.writestr(info, data)
        except BadZipFile as exc:
            with ZipFile(output, "w") as dst_zip:
                dst_zip.writestr("[Content_Types].xml", "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\" />")
            raise ValueError(f"Source is not a valid Office zip file: {source}") from exc

        return str(output)
