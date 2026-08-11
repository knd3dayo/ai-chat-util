from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MarkdownBlock:
    type: str
    text: str = ""
    level: int = 0
    ordered: bool = False
    rows: list[list[str]] = field(default_factory=list)
    image_path: str | None = None
    code: str = ""


class MarkdownParser:
    @classmethod
    def parse(cls, markdown_text: str) -> list[MarkdownBlock]:
        blocks: list[MarkdownBlock] = []
        lines = markdown_text.splitlines()
        current_paragraph: list[str] = []
        code_lines: list[str] = []
        in_code_block = False
        table_rows: list[list[str]] = []

        def flush_paragraph() -> None:
            if current_paragraph:
                blocks.append(MarkdownBlock(type="paragraph", text="\n".join(current_paragraph).strip()))
                current_paragraph.clear()

        def flush_code_block() -> None:
            if code_lines:
                blocks.append(MarkdownBlock(type="code_block", code="\n".join(code_lines)))
                code_lines.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_code_block:
                    code_lines.append("")
                else:
                    flush_paragraph()
                continue

            if stripped.startswith("```"):
                if in_code_block:
                    flush_code_block()
                    in_code_block = False
                else:
                    flush_paragraph()
                    in_code_block = True
                continue

            if in_code_block:
                code_lines.append(stripped)
                continue

            if stripped.startswith("# "):
                flush_paragraph()
                blocks.append(MarkdownBlock(type="heading", text=stripped[2:], level=1))
            elif stripped.startswith("## "):
                flush_paragraph()
                blocks.append(MarkdownBlock(type="heading", text=stripped[3:], level=2))
            elif stripped.startswith("### "):
                flush_paragraph()
                blocks.append(MarkdownBlock(type="heading", text=stripped[4:], level=3))
            elif stripped.startswith("- "):
                flush_paragraph()
                blocks.append(MarkdownBlock(type="list_item", text=stripped[2:]))
            elif stripped.startswith("|") and "|" in stripped[1:]:
                flush_paragraph()
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                table_rows.append(cells)
            elif stripped.startswith("![") and "](" in stripped:
                flush_paragraph()
                label, path = stripped[2:].split("](", 1)
                path = path.rstrip(")")
                blocks.append(MarkdownBlock(type="image", image_path=path))
            else:
                current_paragraph.append(stripped)

        flush_paragraph()
        flush_code_block()
        if table_rows:
            blocks.append(MarkdownBlock(type="table", rows=table_rows))
        return blocks
