from pathlib import Path

from ai_chat_util.util.analyze_file_util.markdown_to_docx import MarkdownToDocxUtil
from ai_chat_util.util.analyze_file_util.markdown_to_pptx import MarkdownToPptxUtil
from ai_chat_util.util.analyze_file_util.markdown_to_excel import MarkdownToExcelUtil
from ai_chat_util.util.analyze_file_util.office_template import OfficeTemplateUtil


def test_markdown_office_extensions_create_outputs(tmp_path: Path) -> None:
    markdown = "# Heading\n\nParagraph text\n\n- bullet\n\n| col1 | col2 |\n|---|---|\n| a | b |"

    docx_path = tmp_path / "demo.docx"
    pptx_path = tmp_path / "demo.pptx"
    excel_path = tmp_path / "demo.xlsx"
    template_path = tmp_path / "template.docx"

    MarkdownToDocxUtil.convert_markdown_to_docx(markdown, docx_path)
    MarkdownToPptxUtil.convert_markdown_to_pptx(markdown, pptx_path)
    MarkdownToExcelUtil.convert_markdown_to_excel(markdown, excel_path)

    import zipfile

    with zipfile.ZipFile(template_path, "w") as archive:
        archive.writestr("word/document.xml", "<root>PLACEHOLDER</root>")
    output_path = tmp_path / "templated.docx"
    OfficeTemplateUtil.apply_template(template_path, output_path, {"PLACEHOLDER": "VALUE"})

    assert docx_path.exists()
    assert pptx_path.exists()
    assert excel_path.exists()
    assert output_path.exists()
