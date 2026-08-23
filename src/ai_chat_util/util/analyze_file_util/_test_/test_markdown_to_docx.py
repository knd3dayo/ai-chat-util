from pathlib import Path

from ai_chat_util.util.analyze_file_util.markdown_to_docx import MarkdownToDocxUtil


def test_markdown_to_docx_creates_document(tmp_path: Path) -> None:
    output_path = tmp_path / "sample.docx"
    result = MarkdownToDocxUtil.convert_markdown_to_docx(
        "# Title\n\nThis is a paragraph.\n\n- bullet one\n- bullet two",
        output_path,
    )

    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
