import fitz  # PyMuPDF
import base64
import json

def extract_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)
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

if __name__ == "__main__":
    import sys
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    content = extract_pdf_content(pdf_file)

    # JSONとして保存
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

    print("抽出完了！output.json に保存しました。")
