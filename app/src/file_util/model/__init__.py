from magika import Magika
from magika.types import MagikaResult 
from chardet.universaldetector import UniversalDetector
from pathlib import Path

from pydantic import BaseModel, Field, PrivateAttr
import file_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

from enum import StrEnum

class FileUtilDocumentType(StrEnum):
    TEXT = "text"
    PDF = "pdf"
    EXCEL = "excel"
    WORD = "word"
    PPT = "ppt"
    IMAGE = "image"
    UNSUPPORTED = "unsupported"


class FileUtilDocument(BaseModel):
    
    data: bytes = Field(..., description="Document data as bytes")
    identifier: str = Field(..., description="Identifier of the document")
    __mime_type: str = PrivateAttr("")
    __encoding: str | None = PrivateAttr(None)

    @property
    def mime_type(self) -> str:
        """MIMEタイプを取得する"""
        return self.__mime_type
    @property
    def encoding(self) -> str | None:
        """エンコーディングを取得する"""
        return self.__encoding
    
    def __init__(self, **data):
        super().__init__(**data)
        mime_type, encoding = self.identify_data_type(self.data)
        self.__mime_type = mime_type if mime_type else ""
        self.__encoding = encoding

    @classmethod
    def from_file(cls, document_path: str) -> "FileUtilDocument":
        """ファイルパスからDocumentTypeインスタンスを作成する

        Args:
            document_path: ドキュメントのファイルパス

        Returns:
            DocumentType: 作成されたDocumentTypeインスタンス
        """
        # ファイルのバイト列を取得
        with open(document_path, "rb") as f:
            byte_data = f.read()
        
        return cls(data=byte_data, identifier=document_path)

    @classmethod
    def identify_data_type(cls, data: bytes) -> tuple[str | None, str | None]:
        """バイト列のMIMEタイプとエンコーディングを判定する

        Args:
            data: 判定対象のバイト列

        Returns:
            tuple[str | None, str | None]:
                MIMEタイプ文字列とエンコーディング文字列のタプル。
                判定失敗時は(None, None)
        """
        m = Magika()
        try:
            res: MagikaResult = m.identify_bytes(data) # type: ignore
            encoding = None
            if res.dl.is_text:
                encoding = cls.get_encoding_from_bytes(data)

        except Exception as e:
            logger.debug(e)
            return None, None

        return res.output.mime_type , encoding

    @classmethod
    def identify_file_type(cls, filename) -> tuple[str | None, str | None]:
        """ファイルのMIMEタイプとエンコーディングを判定する

        Args:
            filename: 判定対象のファイルパス

        Returns:
            tuple[str | None, str | None]:
                MIMEタイプ文字列とエンコーディング文字列のタプル。
                判定失敗時は(None, None)
        """
        m = Magika()
        # ファイルの種類を判定
        path = Path(filename)
        try:
            res: MagikaResult = m.identify_path(path) # type: ignore
            encoding = None
            if res.dl.is_text:
                encoding = cls.get_encoding(filename)

        except Exception as e:
            logger.debug(e)
            return None, None

        return res.output.mime_type , encoding

    @classmethod
    def get_encoding(cls, filename) -> str | None:
        """ファイルのエンコーディングを判定する

        ファイルの先頭8192バイトを読み込んで、エンコーディングを判定します。

        Args:
            filename: 判定対象のファイルパス

        Returns:
            str | None: エンコーディング文字列。判定失敗時はNone
        """
        # ファイルのbyte列を取得
        # アクセスできない場合は例外をキャッチ
        with open(filename, "rb") as f:
            # 1KB読み込む
            byte_data = f.read(8192)
            # エンコーディング判定
            encoding = cls.get_encoding_from_bytes(byte_data)
            return encoding
    
    @classmethod
    def get_encoding_from_bytes(cls, byte_data: bytes) -> str | None:
        """バイト列からエンコーディングを判定する

        Args:
            byte_data: 判定対象のバイト列

        Returns:
            str | None: エンコーディング文字列。判定失敗時はNone
        """
        detector = UniversalDetector()
        detector.feed(byte_data)
        detector.close()
        encoding = detector.result['encoding']  
        return encoding

    def get_document_type(self) -> FileUtilDocumentType:
        """Determine the document type based on its MIME type.

        Returns:
            DocumentTypeEnum:
                The determined document type.
        """
        if self.is_text():
            return FileUtilDocumentType.TEXT
        elif self.is_pdf():
            return FileUtilDocumentType.PDF
        elif self.is_excel():
            return FileUtilDocumentType.EXCEL
        elif self.is_word():
            return FileUtilDocumentType.WORD
        elif self.is_ppt():
            return FileUtilDocumentType.PPT
        elif self.is_image():
            return FileUtilDocumentType.IMAGE
        else:
            return FileUtilDocumentType.UNSUPPORTED

    def is_text(self) -> bool:
        """Check if the document type is a text type based on its MIME type."""
        return self.mime_type.startswith("text/")
    
    def is_pdf(self) -> bool:
        """Check if the document type is a PDF type based on its MIME type."""
        return self.mime_type == "application/pdf"
    
    def is_excel(self) -> bool:
        """Check if the document type is an Excel type based on its MIME type."""
        return self.mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    def is_word(self) -> bool:
        """Check if the document type is a Word type based on its MIME type."""
        return self.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    
    def is_ppt(self) -> bool:
        """Check if the document type is a PowerPoint type based on its MIME type."""
        return self.mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    
    def is_image(self) -> bool:
        """Check if the document type is an image type based on its MIME type."""
        return self.mime_type.startswith("image/")
    
    def is_office_document(self) -> bool:
        """Check if the document type is any Office document type based on its MIME type."""
        return self.is_excel() or self.is_word() or self.is_ppt()
    
    def is_unsupported(self) -> bool:
        """Check if the document type is unsupported based on its MIME type."""
        return not (self.is_text() or self.is_pdf() or self.is_office_document() or self.is_image())
    
