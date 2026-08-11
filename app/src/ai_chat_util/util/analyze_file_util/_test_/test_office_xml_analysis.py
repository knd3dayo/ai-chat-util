import io
import zipfile
from pathlib import Path

from ai_chat_util.util.analyze_file_util.office_xml_analysis import OfficeXmlAnalysisUtil


def _make_png_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGBA", (1, 1), (255, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_analyze_office_xml_report_extracts_fonts_and_images(tmp_path: Path) -> None:
    office_path = tmp_path / "sample.docx"
    with zipfile.ZipFile(office_path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
            <w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">
              <w:body>
                <w:p>
                  <w:r>
                    <w:rPr>
                      <w:rFonts w:ascii=\"Arial\" w:hAnsi=\"Arial\" />
                      <w:sz w:val=\"24\" />
                      <w:b />
                    </w:rPr>
                    <w:t>Hello</w:t>
                  </w:r>
                </w:p>
              </w:body>
            </w:document>
            """,
        )
        archive.writestr(
            "word/styles.xml",
            """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
            <w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">
              <w:style w:type=\"paragraph\">
                <w:pPr>
                  <w:spacing w:before=\"120\" w:after=\"60\" />
                </w:pPr>
              </w:style>
            </w:styles>
            """,
        )
        archive.writestr("word/media/image1.png", _make_png_bytes())

    report = OfficeXmlAnalysisUtil.analyze_office_file(office_path)

    assert report["office_type"] == "docx"
    assert "word/document.xml" in report["xml_parts"]
    assert "word/styles.xml" in report["xml_parts"]
    assert "Arial" in report["fonts"]
    assert report["fonts"]["Arial"] == 1
    assert report["images"][0]["path"] == "word/media/image1.png"
    assert report["images"][0]["size"] == {"width": 1, "height": 1}
