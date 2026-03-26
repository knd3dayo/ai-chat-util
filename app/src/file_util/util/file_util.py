import base64
import os
from file_util.util.excel_util import ExcelUtil
from file_util.util.ppt_util import PPTUtil
from file_util.util.word_util import WordUtil
from file_util.util.text_util import TextUtil
from file_util.util.pdf_util import PDFUtil

from file_util.model import FileUtilDocument

import file_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class FileUtil:
    """ファイル操作のユーティリティクラス"""

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """テキストをサニタイズする

        複数の改行や空白を1つにまとめて、テキストを整形します。

        Args:
            text: サニタイズ対象のテキスト

        Returns:
            サニタイズされたテキスト。入力が空の場合は空文字列
        """
        # textが空の場合は空の文字列を返す
        if not text or len(text) == 0:
            return ""
        import re
        # 1. 複数の改行を1つの改行に変換
        text = re.sub(r'\n+', '\n', text)
        # 2. 複数のスペースを1つのスペースに変換
        text = re.sub(r' +', ' ', text)

        return text

    @classmethod
    async def extract_text_from_file_async(cls, filename) -> str:
        """ファイルからテキストを非同期で抽出する

        対応形式: テキストファイル、PDF、Excel、Word、PowerPoint

        Args:
            filename: 抽出対象のファイルパス

        Returns:
            str: 抽出されたテキスト。サニタイズ済み。非対応形式の場合は空文字列
        """
        document_type = FileUtilDocument.from_file(document_path=filename)
        encoding = document_type.encoding
        mime_type = document_type.mime_type
        
        if mime_type is None:
            return ""
        logger.debug(mime_type)
        result = None        

        if document_type.is_text():
            # テキストファイルの場合
            result = await TextUtil.process_text_async(filename, mime_type, encoding)

        # application/pdf
        elif document_type.is_pdf():
            result = PDFUtil.extract_text_from_pdf(filename)
            
        # application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
        elif document_type.is_excel():
            result = ExcelUtil.extract_text_from_sheet(filename)
            
        # application/vnd.openxmlformats-officedocument.wordprocessingml.document
        elif document_type.is_word():
            result = WordUtil.extract_text_from_docx(filename)
            
        # application/vnd.openxmlformats-officedocument.presentationml.presentation
        elif document_type.is_ppt():
            result = PPTUtil.extract_text_from_pptx(filename)

        else:
            logger.error("Unsupported file type: " + mime_type)

        return cls.sanitize_text(result if result is not None else "")

    @classmethod
    async def extract_base64_to_text(cls, extension: str, base64_data: str) -> str:

        # サイズが0の場合は空文字を返す
        if not base64_data or len(base64_data) == 0:
            return ""

        # base64からバイナリデータに変換
        base64_data_bytes = base64.b64decode(base64_data)

        # 拡張子の指定。extensionがNoneまたは空の場合は設定しない.空でない場合は"."を先頭に付与
        suffix = ""
        if extension is not None and extension != "":
            suffix = "." + extension
        # base64データから一時ファイルを生成
        import aiofiles.tempfile
        async with aiofiles.tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=suffix) as temp:
            await temp.write(base64_data_bytes)
            await temp.close()
            # 一時ファイルからテキストを抽出
            temp_path = temp.name if isinstance(temp.name, str) else str(temp.name)
            text = await FileUtil.extract_text_from_file_async(temp_path)
            # 一時ファイルを削除
            os.remove(temp_path)
            return text

        return text
