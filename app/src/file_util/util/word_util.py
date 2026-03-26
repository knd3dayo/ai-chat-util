import docx
from io import StringIO

class WordUtil:
    @classmethod
    def extract_text_from_docx(cls, filename):
        """
        指定された.docxファイルからテキストを抽出します。

        Args:
            filename (str): .docxファイルのパス
        Returns:
            str: 抽出されたテキスト
        """
        # 出力用のストリームを作成
        output = StringIO()
        doc = docx.Document(filename)
        for para in doc.paragraphs:
            output.write(para.text)
            output.write("\n")
            
        return output.getvalue()