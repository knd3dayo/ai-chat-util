from pdfminer.high_level import extract_text


class PDFUtil:
    @classmethod
    def extract_text_from_pdf(cls, filename: str) -> str:
        """
        PDFファイルからテキストを抽出する。
        Args:
            filename (str): PDFファイルのパス
        Returns:
            str: 抽出されたテキスト
        """
        text = extract_text(filename)
        return text

