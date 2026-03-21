import fitz  # PyMuPDF
import base64
import json
from fitz import Document

def _extract_content(doc: Document) -> list[dict]:
    results = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # --- テキスト抽出 ---
        text = page.get_text("text")
        if isinstance(text, str):
            text = text.strip()
        else:
            text = ""

        if text:
            results.append({
                "type": "text",
                "text": text
            })

        # --- ページ全体を画像化 ---
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        results.append({
            "type": "image",
            "bytes": img_bytes
        })

    return results

def extract_content_from_bytes(pdf_bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return _extract_content(doc)

def extract_content_from_file(pdf_path) -> list[dict]:
    fitz.open()
    doc = fitz.open(pdf_path)
    return _extract_content(doc)

if __name__ == "__main__":
    import sys
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    content = extract_content_from_file(pdf_file)

    # JSONとして保存
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

    print("抽出完了！output.json に保存しました。")
