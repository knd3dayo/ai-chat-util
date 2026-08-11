from pathlib import Path

from ai_chat_util.util.analyze_file_util.markdown_to_pptx import MarkdownToPptxUtil
from ai_chat_util.util.analyze_file_util.markdown_to_excel import MarkdownToExcelUtil


def test_markdown_to_pptx_and_excel_create_files(tmp_path: Path) -> None:
    pptx_path = tmp_path / "sample.pptx"
    excel_path = tmp_path / "sample.xlsx"

    pptx_output = MarkdownToPptxUtil.convert_markdown_to_pptx("# Title\n\nBody text", pptx_path)
    excel_output = MarkdownToExcelUtil.convert_markdown_to_excel("- item 1\n- item 2", excel_path)

    assert pptx_output == str(pptx_path)
    assert excel_output == str(excel_path)
    assert pptx_path.exists()
    assert excel_path.exists()
    assert pptx_path.stat().st_size > 0
    assert excel_path.stat().st_size > 0
