from __future__ import annotations

import time

from ai_chat_util.ai_chat_util_base.chat import AbstractChatClient
from ai_chat_util.ai_chat_util_base.ai_chatl_util_models import (
    ChatContent,
    ChatHistory,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    WebRequestModel,
)
from ai_chat_util.ai_chat_util_base.file_util.model import FileUtilDocument

import ai_chat_util.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


class LLMClientUtil:
    @classmethod
    async def analyze_image_files(
        cls,
        llm_client: AbstractChatClient,
        file_list: list[str],
        prompt: str,
        detail: str,
    ) -> ChatResponse:
        started = time.perf_counter()
        logger.info(
            "IMAGE_ANALYZE_START images=%d detail=%s prompt_len=%d",
            len(file_list or []),
            detail,
            len((prompt or "").strip()),
        )
        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        image_content_list: list[ChatContent] = []
        encode_started = time.perf_counter()
        total_bytes = 0
        for image_path in file_list:
            doc = FileUtilDocument.from_file(document_path=image_path)
            try:
                total_bytes += len(doc.data or b"")
            except Exception:
                pass
            image_contents = llm_client.get_message_factory()._create_image_content_(doc, detail)
            image_content_list.extend(image_contents)
        logger.info(
            "IMAGE_ENCODE_END images=%d total_bytes=%d elapsed_ms=%d",
            len(file_list or []),
            total_bytes,
            int((time.perf_counter() - encode_started) * 1000),
        )

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_started = time.perf_counter()
        logger.info("IMAGE_CHAT_START")
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        logger.info(
            "IMAGE_CHAT_END elapsed_ms=%d total_elapsed_ms=%d",
            int((time.perf_counter() - chat_started) * 1000),
            int((time.perf_counter() - started) * 1000),
        )
        return chat_response

    @classmethod
    async def analyze_image_urls(
        cls,
        llm_client: AbstractChatClient,
        image_url_list: list[WebRequestModel],
        prompt: str,
        detail: str,
    ) -> ChatResponse:
        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        image_content_list: list[ChatContent] = []
        for image_url in image_url_list:
            image_contents = await llm_client.get_message_factory().create_image_content_from_url_async(
                image_url, detail
            )
            image_content_list.extend(image_contents)

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        return chat_response

    @classmethod
    async def analyze_pdf_files(
        cls,
        llm_client: AbstractChatClient,
        file_list: list[str],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        pdf_content_list = []
        config = llm_client.get_config()
        if not config:
            raise ValueError("LLMClientの設定が取得できませんでした。")

        for file_path in file_list:
            if config.features.use_custom_pdf_analyzer:
                logger.info(f"Using custom PDF analyzer for file: {file_path}")
                pdf_content = llm_client.get_message_factory()._create_custom_pdf_content_from_file(file_path, detail)
            else:
                logger.info(f"Using standard PDF analyzer for file: {file_path}")
                pdf_content = llm_client.get_message_factory().create_pdf_content_from_file(file_path)
            pdf_content_list.extend(pdf_content)

        chat_message = ChatMessage(role="user", content=[prompt_content] + pdf_content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        return chat_response

    @classmethod
    async def analyze_office_files(
        cls,
        llm_client: AbstractChatClient,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        office_contents: list[ChatContent] = []
        for file_path in file_path_list:
            pdf_content = llm_client.get_message_factory().create_office_content_from_file(
                file_path, detail=detail
            )
            office_contents.extend(pdf_content)

        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)

        chat_message = ChatMessage(role="user", content=[prompt_content] + office_contents)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        response: ChatResponse = await llm_client.chat(chat_request)
        return response

    @classmethod
    async def analyze_documents_data(
        cls,
        llm_client: AbstractChatClient,
        document_type_list: list[FileUtilDocument],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        content_list = []
        for document_type in document_type_list:
            contents = llm_client.get_message_factory().create_multi_format_content(
                document_type, detail=detail
            )
            content_list.extend(contents)

        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        chat_message = ChatMessage(role="user", content=[prompt_content] + content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        return chat_response

    @classmethod
    async def analyze_files(
        cls,
        llm_client: AbstractChatClient,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        content_list = []
        skipped_files: list[str] = []
        for file_path in file_path_list:
            try:
                contents = llm_client.get_message_factory().create_multi_format_contents_from_file(
                    file_path, detail=detail
                )
            except ValueError as exc:
                if "Unsupported document type" not in str(exc):
                    raise
                skipped_files.append(file_path)
                logger.info("FILE_ANALYZE_SKIP unsupported=%s", file_path)
                continue
            content_list.extend(contents)

        if not content_list:
            raise ValueError(
                "No supported files were found for analyze_files. "
                f"skipped={len(skipped_files)}"
            )

        if skipped_files:
            logger.info("FILE_ANALYZE_SKIPPED_COUNT count=%d", len(skipped_files))

        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        chat_message = ChatMessage(role="user", content=[prompt_content] + content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        return chat_response