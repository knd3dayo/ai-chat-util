from __future__ import annotations

from typing import Optional
import os
import uuid
import tempfile
import atexit
import base64
from abc import ABC, abstractmethod

from ai_chat_util.common.config.runtime import AiChatUtilConfig, get_runtime_config
from ai_chat_util.common.model.ai_chatl_util_models import (
    ChatHistory, ChatRequestContext, ChatMessage,
    ChatContent, WebRequestModel
)
from file_util.model import FileUtilDocument
from file_util.util.office2pdf import Office2PDFUtil
from file_util.util.downloader import DownLoader
from file_util.util import pdf_util

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class LLMMessageContentFactoryBase(ABC):

    def _get_effective_config(self) -> AiChatUtilConfig:
        config = self.get_config()
        if config is not None:
            return config
        return get_runtime_config()

    def _get_network_download_options(self) -> tuple[bool, str | None]:
        config = self._get_effective_config()
        return config.network.requests_verify, config.network.ca_bundle

    def _get_configured_libreoffice_path(self) -> str | None:
        return self._get_effective_config().office2pdf.libreoffice_path

    def is_text_content(self, content: ChatContent) -> bool:
        return content.params.get("type") == "text"

    def is_image_content(self, content: ChatContent) -> bool:
        return content.params.get("type") == "image_url"

    def is_file_content(self, content: ChatContent) -> bool:
        return content.params.get("type") == "file"

    def get_user_role_name(self) -> str:
        return "user"

    def get_assistant_role_name(self) -> str:
        return "assistant"

    def get_system_role_name(self) -> str:
        return "system"

    def create_user_message(self, chat_content_list: list[ChatContent]) -> ChatMessage:
        return ChatMessage(
            role=self.get_user_role_name(),
            content=chat_content_list
        )

    def create_assistant_message(self, chat_content_list: list[ChatContent]) -> ChatMessage:
        return ChatMessage(
            role=self.get_assistant_role_name(),
            content=chat_content_list
        )

    def create_system_message(self, chat_content_list: list[ChatContent]) -> ChatMessage:
        return ChatMessage(
            role=self.get_system_role_name(),
            content=chat_content_list
        )

    def create_text_content(self, text: str) -> "ChatContent":
        params = {"type": "text", "text": text}
        return ChatContent(params=params)

    def create_image_content(self, file_data: FileUtilDocument, detail: str) -> list["ChatContent"]:
        return self._create_image_content_(file_data, detail)

    def create_image_content_from_file(self, file_path: str, detail: str) -> list["ChatContent"]:
        return self._create_image_content_(FileUtilDocument.from_file(file_path), detail)

    def create_image_content_from_url(self, file_url: WebRequestModel, detail: str) -> list["ChatContent"]:
        tmpdir = tempfile.TemporaryDirectory()
        atexit.register(tmpdir.cleanup)

        requests_verify, ca_bundle = self._get_network_download_options()
        file_paths = DownLoader.download_files(
            [file_url],
            tmpdir.name,
            requests_verify=requests_verify,
            ca_bundle=ca_bundle,
        )
        return self._create_image_content_(FileUtilDocument.from_file(file_paths[0]), detail)

    async def create_image_content_from_url_async(self, file_url: WebRequestModel, detail: str) -> list["ChatContent"]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            requests_verify, ca_bundle = self._get_network_download_options()
            file_paths = await DownLoader.download_files_async(
                [file_url],
                tmp_dir,
                requests_verify=requests_verify,
                ca_bundle=ca_bundle,
            )
            return self._create_image_content_(FileUtilDocument.from_file(file_paths[0]), detail)

    def create_pdf_content(self, document_type: FileUtilDocument, detail: str = "auto") -> list["ChatContent"]:
        config = self.get_config()
        if not config:
            raise ValueError("LLMClientの設定が取得できませんでした。")

        use_custom = config.features.use_custom_pdf_analyzer
        if use_custom:
            return self._create_custom_pdf_content_(document_type, detail=detail)
        else:
            return self._create_pdf_content_(document_type, detail=detail)

    def create_pdf_content_from_file(self, file_path: str, detail: str = "auto") -> list["ChatContent"]:
            return self.create_pdf_content(FileUtilDocument.from_file(file_path), detail=detail)

    def create_pdf_content_from_url(self, file_url: str, detail: str = "auto") -> list["ChatContent"]:
        tmpdir = tempfile.TemporaryDirectory()
        atexit.register(tmpdir.cleanup)

        requests_verify, ca_bundle = self._get_network_download_options()
        file_paths = DownLoader.download_files(
            [WebRequestModel(url=file_url)],
            tmpdir.name,
            requests_verify=requests_verify,
            ca_bundle=ca_bundle,
        )
        return self.create_pdf_content_from_file(file_paths[0], detail=detail)

    def _create_custom_pdf_content_from_file(self, file_path: str, detail: str = "auto") -> list["ChatContent"]:
        '''
        PDFファイルのバイトデータから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''
        with open(file_path, "rb") as pdf_file:
            document_type = FileUtilDocument(data=pdf_file.read(), identifier=file_path)
        return self._create_custom_pdf_content_(document_type, detail=detail)

    def _create_custom_pdf_content_(self, document_type: FileUtilDocument, detail: str = "auto") -> list["ChatContent"]:
        '''
        PDFファイルのバイトデータから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''
        page_info_content = self.create_text_content(text=f"PDFファイル: {document_type.identifier} の内容を以下に示します。")
        pdf_contents = [page_info_content]
        # PDFからテキストと画像を抽出
        pdf_elements = pdf_util.extract_content_from_bytes(document_type.data)
        for element in pdf_elements:
            if element["type"] == "text":
                text_content = self.create_text_content(text=element["text"])
                pdf_contents.append(text_content)
            elif element["type"] == "image":
                document_type = FileUtilDocument(data=element["bytes"], identifier=document_type.identifier)
                image_content = self._create_image_content_(document_type, detail)
                pdf_contents.extend(image_content)

        return pdf_contents

    def create_office_content(self, document_type: FileUtilDocument, detail: str) -> list["ChatContent"]:
        '''
        複数のOfficeドキュメントとプロンプトからドキュメント解析を行う。各ドキュメントのテキスト抽出、各ドキュメントの説明、プロンプト応答を生成して返す
        '''
        # Officeドキュメントを一時的にPDFに変換する
        temp_dir = tempfile.TemporaryDirectory()
        atexit.register(temp_dir.cleanup)
        temp_file_path = os.path.join(temp_dir.name, f"{os.path.basename(document_type.identifier)}_{uuid.uuid4()}.pdf")
        Office2PDFUtil.create_pdf_from_document_bytes(
            input_bytes=document_type.data,
            output_path=temp_file_path,
            configured_libreoffice_path=self._get_configured_libreoffice_path(),
        )
        # 元ファイルからPDFに変換した旨の説明を追加
        explanation_text = f"""
            {temp_file_path}は、元のOfficeドキュメント: {document_type.identifier} をPDFに変換したものです。
            ユーザーにどのファイルを元にしたのかを伝えるため、回答を行う際には、元のファイル名を使用してください。
            """
        explanation_content = self.create_text_content(text=explanation_text)
        pdf_contents = [explanation_content]

        pdf_contents.extend(self.create_pdf_content_from_file(temp_file_path, detail=detail))

        return pdf_contents

    def create_office_content_from_file(
            self, file_path: str, detail: str = "auto"
            ) -> list["ChatContent"]:
        '''
        複数のOfficeドキュメントとプロンプトからドキュメント解析を行う。各ドキュメントのテキスト抽出、各ドキュメントの説明、プロンプト応答を生成して返す
        '''
        with open(file_path, "rb") as office_file:
            document_type = FileUtilDocument(data=office_file.read(), identifier=file_path)
        return self.create_office_content(document_type, detail=detail)

    def create_office_content_from_url(self, file_url: str, detail: str = "auto") -> list["ChatContent"]:
        '''
        複数のOfficeドキュメントとプロンプトからドキュメント解析を行う。各ドキュメントのテキスト抽出、各ドキュメントの説明、プロンプト応答を生成して返す
        '''
        tmpdir = tempfile.TemporaryDirectory()
        atexit.register(tmpdir.cleanup)

        requests_verify, ca_bundle = self._get_network_download_options()
        file_paths = DownLoader.download_files(
            [WebRequestModel(url=file_url)],
            tmpdir.name,
            requests_verify=requests_verify,
            ca_bundle=ca_bundle,
        )

        office_contents = []
        for file_path in file_paths:
            content = self.create_office_content_from_file(
                file_path, detail=detail)
            office_contents.extend(content)

        return office_contents

    def create_multi_format_content(
            self, document_type: FileUtilDocument, detail: str = "auto"
            ) -> list["ChatContent"]:
        '''
        複数形式ファイルから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''

        if document_type.is_text():
            text = document_type.data.decode('utf-8')
            return [self.create_text_content(text)]

        if document_type.is_image():
            return self.create_image_content(document_type, detail)

        if document_type.is_pdf():
            return self.create_pdf_content(document_type, detail=detail)

        if document_type.is_office_document():
            return self.create_office_content(document_type, detail=detail)

        raise ValueError(f"Unsupported document type for file: {document_type.identifier}")

    def create_multi_format_contents_from_file(
            self, file_path: str, detail: str = "auto"
            ) -> list["ChatContent"]:
        '''
        複数形式ファイルから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''

        document_type = FileUtilDocument.from_file(document_path=file_path)

        if document_type.is_text():
            with open(file_path, "r", encoding="utf-8") as text_file:
                text_data = text_file.read()
            return [self.create_text_content(text_data)]

        if document_type.is_image():
            return self.create_image_content_from_file(file_path, detail)

        if document_type.is_pdf():
            return self.create_pdf_content_from_file(file_path, detail=detail)

        if document_type.is_office_document():
            return self.create_office_content_from_file(file_path, detail=detail)

        raise ValueError(f"Unsupported document type for file: {file_path}")

    def create_multi_format_contents_from_url(
            self, file_url: str, detail: str = "auto"
            ) -> list["ChatContent"]:
        '''
        複数形式ファイルから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''
        tmpdir = tempfile.TemporaryDirectory()
        atexit.register(tmpdir.cleanup)
        requests_verify, ca_bundle = self._get_network_download_options()
        file_paths = DownLoader.download_files(
            [WebRequestModel(url=file_url)],
            tmpdir.name,
            requests_verify=requests_verify,
            ca_bundle=ca_bundle,
        )

        return self.create_multi_format_contents_from_file(
            file_paths[0], detail=detail
        )

    # 最後のsystem, assistant以降のユーザーメッセージリストを取得するユーティリティ関数
    # 最後のユーザーメッセージリストとそれ以前のメッセージリストのタプルを返す
    def __get_last_user_messages__(
        self, chat_history: ChatHistory
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        last_user_messages: list[ChatMessage] = []
        previous_messages: list[ChatMessage] = []
        for message in reversed(chat_history.messages):
            if message.role == self.get_user_role_name():
                last_user_messages.insert(0, message)
            else:
                previous_messages.insert(0, message)
                if message.role in [self.get_system_role_name(), self.get_assistant_role_name()]:
                    break
        return last_user_messages, previous_messages

    def __preprocess_text_message__(
            self,
            chat_message_list: list[ChatMessage],
            request_context: ChatRequestContext
        ) -> list[ChatMessage]:
        '''
        request_contextの内容に従い、メッセージの前処理を実施する
        * ChatMessageのcontentのうち、typeがtextの要素を抽出し、
            * split_modeがnone以外の場合、split_message_lengthで指定された文字数を超える場合は分割する
            * split_modeがnone以外の場合、prompt_template_textを各分割メッセージの前に付与する.
              prompt_template_textが空文字列の場合は例外をスローする
        Args:
            chat_message_list (list[ChatMessage]): 前処理対象のChatMessageのリスト
            request_context (ChatRequestContext): 前処理の設定情報
        Returns:
            list[ChatMessage]: 前処理後のChatMessageのリスト

        '''
        def __insert_prompt_template__(
            chat_message_list: list[ChatMessage],
            request_context: ChatRequestContext
        ) -> list[ChatMessage]:
            result_chat_message_list: list[ChatMessage] = []
            for chat_message in chat_message_list:
                if request_context.prompt_template_text:
                    prompt_template_content = self.create_text_content(request_context.prompt_template_text)
                    chat_message.content.insert(0, prompt_template_content)
                result_chat_message_list.append(chat_message)
            return result_chat_message_list

        if request_context.split_mode == ChatRequestContext.split_mode_name_none:
            return __insert_prompt_template__(chat_message_list, request_context)

        if not request_context.prompt_template_text:
            raise ValueError("prompt_template_text must be set when split_mode is not 'None'")

        split_message_length = request_context.split_message_length
        if split_message_length <= 0:
            # 分割しない設定の場合はそのまま返す
            return __insert_prompt_template__(chat_message_list, request_context)

        # textタイプのcontentを抽出する
        text_type_contents = [
            content for chat_message in chat_message_list for content in chat_message.content if self.is_text_content(content)
            ]
        if len(text_type_contents) == 0:
            return __insert_prompt_template__(chat_message_list, request_context)

        # text以外のcontentを抽出する
        non_text_contents = [
            content for chat_message in chat_message_list for content in chat_message.content if not self.is_text_content(content)
        ]

        text_result_chat_message_list: list[ChatMessage] = []
        # textを結合
        combined_text = "\n".join([text_content.params.get("text", "") for text_content in text_type_contents])
        # 文字数で分割する
        for i in range(0, len(combined_text), split_message_length):
            split_text = combined_text[i:i + split_message_length]
            split_contents = [self.create_text_content(f"{request_context.prompt_template_text}\n{split_text}")]
            for split_content in split_contents:
                chat_message = ChatMessage(
                    role=self.get_user_role_name(),
                    content=[split_content]
                )
                # textタイプ以外のcontentを追加する
                for non_text_content in non_text_contents:
                    chat_message.content.append(non_text_content)

                text_result_chat_message_list.append(chat_message)

        return text_result_chat_message_list

    def __preprocess_image_urls__(
        self,
        chat_message_list: list[ChatMessage],
        request_context: ChatRequestContext
    ) -> list[ChatMessage]:
        '''
        request_contextの内容に従い、画像URLの前処理を実施する
        * split_modeがnone以外の場合、
          ChatMessageのcontentのうち、typeがimage_urlの要素を抽出し、
          max_images_per_requestで指定された画像数を超える場合は分割する
        Args:
            chat_message_list (list[ChatMessage]): 前処理対象のChatMessageのリスト
            request_context (ChatRequestContext): 前処理の設定情報
        Returns:
            list[ChatMessage]: 前処理後のChatMessageのリスト
        '''
        if request_context.split_mode == ChatRequestContext.split_mode_name_none:
            return chat_message_list

        max_images = request_context.max_images_per_request
        if max_images <= 0:
            # 分割しない設定の場合はそのまま返す
            return chat_message_list

        result_chat_message_list: list[ChatMessage] = []

        for chat_message in chat_message_list:
            image_url_contents = [
                content for content in chat_message.content if self.is_image_content(content)
            ]

            # 画像が無い、または分割不要ならそのまま
            if len(image_url_contents) == 0 or len(image_url_contents) <= max_images:
                result_chat_message_list.append(chat_message)
                continue

            # 分割時は、テキスト＋その他（画像以外）を各分割メッセージに維持する
            text_contents = [
                content for content in chat_message.content if self.is_text_content(content)
            ]
            other_contents = [
                content
                for content in chat_message.content
                if (not self.is_text_content(content)) and (not self.is_image_content(content))
            ]
            base_contents = text_contents + other_contents

            for i in range(0, len(image_url_contents), max_images):
                split_image_url_contents: list[ChatContent] = image_url_contents[i: i + max_images]
                split_contents = base_contents + split_image_url_contents

                split_chat_message = ChatMessage(role=chat_message.role, content=split_contents)
                result_chat_message_list.append(split_chat_message)

        return result_chat_message_list

    @abstractmethod
    def _create_image_content_(self, document_type: FileUtilDocument, detail: str) -> list[ChatContent]:
        pass

    @abstractmethod
    def _create_pdf_content_(self, document_type: FileUtilDocument, detail: str) -> list[ChatContent]:
        pass

    @abstractmethod
    def get_config(self) -> AiChatUtilConfig | None:
        pass


class LLMMessageContentFactory(LLMMessageContentFactoryBase):

    def __init__(self, config: Optional[AiChatUtilConfig] = None):
        self.config = config

    def get_config(self) -> AiChatUtilConfig | None:
        return self.config

    def _create_image_content_(self, document_type: FileUtilDocument, detail: str) -> list[ChatContent]:
        base64_image = base64.b64encode(document_type.data).decode('utf-8')
        image_url = f"data:image/png;base64,{base64_image}"
        params = {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
        return [ChatContent(params=params)]

    def _create_pdf_content_(self, document_type: FileUtilDocument, detail: str) -> list[ChatContent]:
        base64_file = base64.b64encode(document_type.data).decode('utf-8')
        file_url = f"data:application/pdf;base64,{base64_file}"
        params = {"type": "file", "file": {"file_data": file_url, "filename": document_type.identifier}}
        return [ChatContent(params=params)]