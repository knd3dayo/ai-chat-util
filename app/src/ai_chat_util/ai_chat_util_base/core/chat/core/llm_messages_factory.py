from __future__ import annotations

from typing import Optional
from io import BytesIO
import os
import uuid
import tempfile
import atexit
import base64
from abc import ABC, abstractmethod

from docx import Document as WordDocument
from openpyxl import load_workbook
from pptx import Presentation

from ai_chat_util.ai_chat_util_base.core.common.config.runtime import AiChatUtilConfig, get_runtime_config
from ai_chat_util.ai_chat_util_base.core.chat.model import (
    ChatHistory, ChatRequestContext, ChatMessage,
    ChatContent
)
from ai_chat_util.ai_chat_util_base.util.analyze_file_util import pdf_util

import ai_chat_util.ai_chat_util_base.core.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class LLMMessageContentFactoryBase(ABC):

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

    def create_image_content(self, identifier: str, data: bytes, detail: str) -> list["ChatContent"]:
        return self._create_image_content_(identifier, data, detail)

    def create_pdf_content(self, identifier: str, data: bytes, detail: str = "auto") -> list["ChatContent"]:
        config = self.get_config()
        if not config:
            raise ValueError("LLMClientの設定が取得できませんでした。")

        use_custom = config.features.use_custom_pdf_analyzer
        if use_custom:
            return self._create_custom_pdf_content_(identifier, data, detail=detail)
        else:
            return self._create_pdf_content_(identifier, data, detail=detail)

    def _create_custom_pdf_content_(self, identifier: str, data: bytes, detail: str = "auto") -> list["ChatContent"]:
        '''
        PDFファイルのバイトデータから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''
        page_info_content = self.create_text_content(text=f"PDFファイル: {identifier} の内容を以下に示します。")
        pdf_contents = [page_info_content]
        # PDFからテキストと画像を抽出
        pdf_elements = pdf_util.extract_content_from_bytes(data)
        for element in pdf_elements:
            if element["type"] == "text":
                text_content = self.create_text_content(text=element["text"])
                pdf_contents.append(text_content)
            elif element["type"] == "image":
                image_content = self._create_image_content_(identifier, element["bytes"], detail)
                pdf_contents.extend(image_content)

        return pdf_contents

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
    def _create_image_content_(self, identifier: str, data: bytes, detail: str) -> list[ChatContent]:
        pass

    @abstractmethod
    def _create_pdf_content_(self, identifier: str, data: bytes, detail: str) -> list[ChatContent]:
        pass

    @abstractmethod
    def get_config(self) -> AiChatUtilConfig | None:
        pass


class LLMMessageContentFactory(LLMMessageContentFactoryBase):

    def __init__(self, config: Optional[AiChatUtilConfig] = None):
        self.config = config

    def get_config(self) -> AiChatUtilConfig | None:
        return self.config

    def _create_image_content_(self, identifier: str, data: bytes, detail: str) -> list[ChatContent]:
        base64_image = base64.b64encode(data).decode('utf-8')
        image_url = f"data:image/png;base64,{base64_image}"
        identifier_params = {"type": "text", "text": f"Image Identifier: {identifier}"}
        image_params = {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
        return [ChatContent(params=identifier_params), ChatContent(params=image_params)]
    

    def _create_pdf_content_(self, identifier: str, data: bytes, detail: str) -> list[ChatContent]:
        base64_file = base64.b64encode(data).decode('utf-8')
        file_url = f"data:application/pdf;base64,{base64_file}"
        params = {"type": "file", "file": {"file_data": file_url, "filename": identifier}}
        return [ChatContent(params=params)]