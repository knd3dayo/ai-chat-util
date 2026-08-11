from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image


class OfficeXmlAnalysisUtil:
    _SUPPORTED_SUFFIXES = {
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".pptx": "pptx",
    }
    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}

    @classmethod
    def analyze_office_file(cls, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)
        office_type = cls._detect_office_type(path)
        if office_type is None:
            raise ValueError(f"Unsupported Office file type: {path.suffix or '<none>'}")

        with zipfile.ZipFile(path, "r") as archive:
            xml_parts = sorted(name for name in archive.namelist() if name.lower().endswith(".xml"))
            images = []
            fonts: Counter[str] = Counter()
            formatting: Counter[str] = Counter()
            relationships: list[dict[str, Any]] = []

            for entry_name in xml_parts:
                try:
                    data = archive.read(entry_name)
                    content = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                try:
                    root = ET.fromstring(content)
                except ET.ParseError:
                    continue

                cls._collect_fonts(root, fonts)
                cls._collect_formatting(root, formatting)

                if entry_name.lower().endswith(".rels") or entry_name.lower().endswith("xml.rels"):
                    for rel in root.findall("{*}Relationship"):
                        relationships.append(
                            {
                                "id": rel.attrib.get("Id", ""),
                                "type": rel.attrib.get("Type", ""),
                                "target": rel.attrib.get("Target", ""),
                            }
                        )

            for entry_name in archive.namelist():
                lower_name = entry_name.lower()
                if not any(lower_name.endswith(suffix) for suffix in cls._IMAGE_SUFFIXES):
                    continue

                data = archive.read(entry_name)
                image_data = {"path": entry_name, "bytes": len(data), "size": None}
                try:
                    with Image.open(io.BytesIO(data)) as img:
                        image_data["size"] = {"width": img.width, "height": img.height}
                except Exception:
                    pass
                images.append(image_data)

            return {
                "office_type": office_type,
                "source_path": str(path),
                "xml_parts": xml_parts,
                "fonts": dict(fonts),
                "formatting": dict(formatting),
                "images": images,
                "relationships": relationships,
            }

    @classmethod
    def analyze_office_files(cls, file_paths: list[str | Path]) -> list[dict[str, Any]]:
        return [cls.analyze_office_file(path) for path in file_paths]

    @classmethod
    def format_report(cls, report: dict[str, Any]) -> str:
        summary_lines = [
            f"Office type: {report.get('office_type', 'unknown')}",
            f"Source: {report.get('source_path', '')}",
            f"XML parts: {len(report.get('xml_parts', []))}",
        ]
        fonts = report.get("fonts") or {}
        if fonts:
            summary_lines.append("Fonts:")
            for font_name, count in sorted(fonts.items()):
                summary_lines.append(f"- {font_name}: {count}")

        formatting = report.get("formatting") or {}
        if formatting:
            summary_lines.append("Formatting hints:")
            for name, count in sorted(formatting.items()):
                summary_lines.append(f"- {name}: {count}")

        images = report.get("images") or []
        if images:
            summary_lines.append("Images:")
            for image in images:
                size = image.get("size") or {}
                summary_lines.append(
                    f"- {image.get('path', '')} ({size.get('width', '?')}x{size.get('height', '?')})"
                )

        return "\n".join(summary_lines)

    @classmethod
    def to_json(cls, report: dict[str, Any]) -> str:
        return json.dumps(report, ensure_ascii=False, indent=2)

    @classmethod
    def _detect_office_type(cls, path: Path) -> str | None:
        return cls._SUPPORTED_SUFFIXES.get(path.suffix.lower())

    @classmethod
    def _collect_fonts(cls, element: ET.Element, fonts: Counter[str]) -> None:
        local_name = cls._local_name(element.tag)
        if local_name == "rFonts":
            seen_values: set[str] = set()
            for attr_name in ("ascii", "hAnsi", "eastAsia", "cs"):
                value = cls._find_attribute_value(element, attr_name)
                if value and value not in seen_values:
                    fonts[value] += 1
                    seen_values.add(value)

        for child in element:
            cls._collect_fonts(child, fonts)

    @classmethod
    def _collect_formatting(cls, element: ET.Element, formatting: Counter[str]) -> None:
        local_name = cls._local_name(element.tag)
        if local_name in {"b", "i", "u", "sz", "color", "spacing", "indent", "shd"}:
            formatting[local_name] += 1

        for child in element:
            cls._collect_formatting(child, formatting)

    @classmethod
    def _local_name(cls, tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @classmethod
    def _find_attribute_value(cls, element: ET.Element, attr_name: str) -> str | None:
        for key, value in element.attrib.items():
            if cls._local_name(key) == attr_name and value:
                return value
        return None
