from typing import Optional, Any, cast
import os
import uuid
import asyncio
import time
import tempfile
import atexit
import requests
import base64
from abc import ABC, abstractmethod

from ..config.runtime import get_runtime_config, AiChatUtilConfig
from ..model.models import (
    ChatHistory, ChatResponse, ChatRequestContext, ChatMessage, 
    ChatContent, WebRequestModel, ChatRequest
)
from ..util.office2pdf import Office2PDFUtil
from file_util.model import FileUtilDocument
from .mcp_client import MCPClient

import litellm

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LLMClientUtil:

    @classmethod
    def is_text_content(cls, content: ChatContent) -> bool:
        return content.params.get("type") == "text"

    @classmethod
    def is_image_content(cls, content: ChatContent) -> bool:
        return content.params.get("type") == "image_url"
    
    @classmethod
    def is_file_content(cls, content: ChatContent) -> bool:
        return content.params.get("type") == "file"

    @classmethod
    def download_files(cls, urls: list[WebRequestModel], download_dir: str) -> list[str]:
        """
        Download files from the given URLs to the specified directory.
        Returns a list of file paths where the files are saved.
        """
        cfg = get_runtime_config()

        verify_enabled = cfg.network.requests_verify
        ca_bundle = cfg.network.ca_bundle
        # requestsのverifyには bool | str(=CA bundle path) を渡せる
        verify: bool | str
        if ca_bundle:
            verify = ca_bundle
        else:
            verify = verify_enabled

        def get_file_name_from_url(url: str) -> str:
            """
            URLからファイル名を抽出する。
            例: https://example.com/path/to/file.txt -> file.txt
            """
            from urllib.parse import urlparse
            import os
            parsed_url = urlparse(url)
            return os.path.basename(parsed_url.path)
        
        file_paths = []
        for item in urls:
            # タイムアウト無しだと接続/読み取りで無限待ちになり得る
            res = requests.get(url=item.url, headers=item.headers, verify=verify, timeout=(10, 60))
            res.raise_for_status()
            
            file_path = os.path.join(download_dir, get_file_name_from_url(item.url))
            with open(file_path, "wb") as f:
                f.write(res.content)
            file_paths.append(file_path)
        return file_paths

    @classmethod
    async def download_files_async(cls, urls: list[WebRequestModel], download_dir: str) -> list[str]:
        """Download files asynchronously (for use inside async flows).

        This avoids blocking the event loop (sync requests.get) and ensures
        HTTP resources are closed via `async with`.
        """
        cfg = get_runtime_config()

        verify_enabled = cfg.network.requests_verify
        ca_bundle = cfg.network.ca_bundle
        verify: bool | str
        if ca_bundle:
            verify = ca_bundle
        else:
            verify = verify_enabled

        def get_file_name_from_url(url: str) -> str:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            return os.path.basename(parsed_url.path)

        try:
            import httpx
        except Exception as e:
            raise RuntimeError("httpx が見つかりません。依存関係を確認してください。") from e

        timeout = httpx.Timeout(60.0, connect=10.0)

        file_paths: list[str] = []
        async with httpx.AsyncClient(verify=verify, timeout=timeout, follow_redirects=True) as client:
            for item in urls:
                resp = await client.get(item.url, headers=item.headers)
                resp.raise_for_status()

                file_path = os.path.join(download_dir, get_file_name_from_url(item.url))
                with open(file_path, "wb") as f:
                    f.write(resp.content)
                file_paths.append(file_path)

        return file_paths

    @classmethod
    async def run_mcp_chat_completion(
        cls, llm_config: AiChatUtilConfig, chat_request: ChatRequest, default_timeout_seconds, **kwargs
    ) -> ChatResponse:
        messages = chat_request.chat_history.messages
        mcp_client = MCPClient(llm_config)
        # タイムアウトが未指定だと、ネットワーク待ちで無限に止まることがある
        kwargs.setdefault("timeout", default_timeout_seconds)

        # LiteLLM/OpenAI 側の timeout とは別に、アプリ側でも強制タイムアウトを掛けて
        # 予期せぬ接続待ち等で“体感ハング”しないようにする。
        hard_timeout: float = default_timeout_seconds
        timeout_kw = kwargs.get("timeout")
        if isinstance(timeout_kw, (int, float)) and float(timeout_kw) > 0:
            hard_timeout = float(timeout_kw)

        # async関数内で同期I/Oを呼ぶとイベントループがブロックされるため、acompletionを使う
        # NOTE: messages には画像base64等が入るため、ここでは内容をログ出力しない（巨大化防止）
        logger.debug(
            "LLM completion request: provider=%s model=%s messages=%d timeout=%s",
            llm_config.llm.provider,
            llm_config.llm.completion_model,
            len(messages),
            kwargs.get("timeout"),
        )
        try:
            response = await asyncio.wait_for(
                mcp_client.chat(chat_request),
                timeout=hard_timeout,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                "LLM呼び出しがタイムアウトしました。"
                f" timeout={hard_timeout}s provider={llm_config.llm.provider} model={llm_config.llm.completion_model}.\n"
                "対処: config.yml の llm.timeout_seconds を増やすか、CLIの --loglevel/--logfile でログを確認してください。"
            ) from e
        logger.debug("LLM completion response type: %s", type(response))

        if isinstance(response, ChatResponse):
            return response

        raise TypeError(f"Unexpected MCP response type: {type(response)!r}")

    @classmethod
    async def run_litellm_chat_completion(
        cls, llm_config: AiChatUtilConfig, chat_request: ChatRequest, default_timeout_seconds, **kwargs
    ) -> ChatResponse:
        messages = chat_request.chat_history.messages
        message_dict_list: list[dict[str, Any]] = [msg.model_dump() for msg in messages]
        params = {}
        # api_key の解決/未設定エラーは設定ロード時(runtime)に行う。
        provider = (llm_config.llm.provider or "").lower()
        api_key = llm_config.llm.api_key
        params["api_key"] = api_key
        params["model"] = f"{llm_config.llm.provider}/{llm_config.llm.completion_model}"
        params["messages"] = message_dict_list
        if llm_config.llm.base_url:
            params["base_url"] = llm_config.llm.base_url
        if llm_config.llm.api_version:
            params["api_version"] = llm_config.llm.api_version

        # タイムアウトが未指定だと、ネットワーク待ちで無限に止まることがある
        kwargs.setdefault("timeout", default_timeout_seconds)

        # LiteLLM/OpenAI 側の timeout とは別に、アプリ側でも強制タイムアウトを掛けて
        # 予期せぬ接続待ち等で“体感ハング”しないようにする。
        hard_timeout: float = default_timeout_seconds
        timeout_kw = kwargs.get("timeout")
        if isinstance(timeout_kw, (int, float)) and float(timeout_kw) > 0:
            hard_timeout = float(timeout_kw)

        # async関数内で同期I/Oを呼ぶとイベントループがブロックされるため、acompletionを使う
        # NOTE: messages には画像base64等が入るため、ここでは内容をログ出力しない（巨大化防止）
        logger.debug(
            "LLM completion request: provider=%s model=%s messages=%d timeout=%s",
            provider,
            params.get("model"),
            len(message_dict_list),
            kwargs.get("timeout"),
        )
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    **params,
                    **kwargs
                ),
                timeout=hard_timeout,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                "LLM呼び出しがタイムアウトしました。"
                f" timeout={hard_timeout}s provider={provider} model={params.get('model')}.\n"
                "対処: config.yml の llm.timeout_seconds を増やすか、CLIの --loglevel/--logfile でログを確認してください。"
            ) from e
        logger.debug("LLM completion response type: %s", type(response))

        if isinstance(response, litellm.ModelResponse):
            # NOTE: litellm.ModelResponse は実行時に usage が載りますが、型定義上は
            # 属性として見えないことがあるため dict-style access を使う。
            usage = response.get("usage") or {}
            output_tokens = int(usage.get("completion_tokens", 0) or 0)
            input_tokens = int(usage.get("prompt_tokens", 0) or 0)

            choices = cast(list[Any], response.get("choices") or [])
            output = ""
            if choices:
                first_choice = cast(Any, choices[0])
                # OpenAI互換の {"message": {"content": "..."}} を優先して読む
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                else:
                    message = getattr(first_choice, "message", None)

                if isinstance(message, dict):
                    output = message.get("content") or ""
                else:
                    output = getattr(message, "content", "") or ""

            # choicesが空 or contentが空の場合は、明示的に失敗させて原因をユーザーに見せる。
            # （"何も出力されない" 体験を避ける）
            if not choices:
                err = response.get("error")
                if err:
                    raise RuntimeError(f"LLM応答にエラーが含まれています: {err}")
                response_dict = cast(dict[str, Any], response)
                raise RuntimeError(
                    "LLM応答の choices が空でした。"
                    f" response_keys={list(response_dict.keys())}"
                )
            if not str(output).strip():
                err = response.get("error")
                if err:
                    raise RuntimeError(f"LLM応答にエラーが含まれています: {err}")
                response_dict = cast(dict[str, Any], response)
                raise RuntimeError(
                    "LLM応答の content が空でした。"
                    f" response_keys={list(response_dict.keys())}"
                )

            return ChatResponse(
                messages=[ChatMessage(role="assistant", content=[ChatContent(params={"type": "text", "text": output})])],
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        raise TypeError(f"Unexpected response type: {type(response)!r}")

class LLMClientBase(ABC):

    
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

    @abstractmethod
    def create(
        self, llm_config: AiChatUtilConfig | None = None
        ) -> "LLMClientBase":
        pass

    @abstractmethod
    def _create_image_content_(self, document_type: FileUtilDocument, detail: str) -> list[ChatContent]:
        pass

    @abstractmethod
    def _create_pdf_content_(self, document_type: FileUtilDocument, detail: str) -> list[ChatContent]:
        pass

    @abstractmethod
    def get_config(self) -> AiChatUtilConfig | None:
        pass

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

        file_paths = LLMClientUtil.download_files([file_url], tmpdir.name)
        return self._create_image_content_(FileUtilDocument.from_file(file_paths[0]), detail)

    async def create_image_content_from_url_async(self, file_url: WebRequestModel, detail: str) -> list["ChatContent"]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_paths = await LLMClientUtil.download_files_async([file_url], tmp_dir)
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

        file_paths = LLMClientUtil.download_files([WebRequestModel(url=file_url)], tmpdir.name)
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
        import ai_chat_util.util.pdf_util as pdf_util

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
            output_path=temp_file_path
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

        file_paths = LLMClientUtil.download_files([WebRequestModel(url=file_url)], tmpdir.name)

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
        file_paths = LLMClientUtil.download_files([WebRequestModel(url=file_url)], tmpdir.name)

        return self.create_multi_format_contents_from_file(
            file_paths[0], detail=detail
        )


    async def analyze_image_files(self, file_list: list[str], prompt: str, detail: str) -> ChatResponse:
        '''
        複数の画像とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        started = time.perf_counter()
        logger.info(
            "IMAGE_ANALYZE_START images=%d detail=%s prompt_len=%d",
            len(file_list or []),
            detail,
            len((prompt or "").strip()),
        )
        prompt_content = self.create_text_content(text=prompt)
        image_content_list: list[ChatContent] = []
        encode_started = time.perf_counter()
        total_bytes = 0
        for image_path in file_list:
            doc = FileUtilDocument.from_file(document_path=image_path)
            try:
                total_bytes += len(doc.data or b"")
            except Exception:
                pass
            image_contents = self.create_image_content(doc, detail)
            image_content_list.extend(image_contents)
        logger.info(
            "IMAGE_ENCODE_END images=%d total_bytes=%d elapsed_ms=%d",
            len(file_list or []),
            total_bytes,
            int((time.perf_counter() - encode_started) * 1000),
        )

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None)
        chat_started = time.perf_counter()
        logger.info("IMAGE_CHAT_START")
        chat_response: ChatResponse = await self.chat(chat_request)
        logger.info(
            "IMAGE_CHAT_END elapsed_ms=%d total_elapsed_ms=%d",
            int((time.perf_counter() - chat_started) * 1000),
            int((time.perf_counter() - started) * 1000),
        )
        return chat_response
    
    async def analyze_image_urls(self, image_url_list: list[WebRequestModel], prompt: str, detail: str) -> ChatResponse:
        '''
        複数の画像URLとプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        prompt_content = self.create_text_content(text=prompt)
        image_content_list: list[ChatContent] = []
        for image_url in image_url_list:
            image_contents = await self.create_image_content_from_url_async(image_url, detail)
            image_content_list.extend(image_contents)

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(
                messages=[chat_message]), chat_request_context=None)
        chat_response: ChatResponse = await self.chat(chat_request)
        return chat_response
    

    async def analyze_pdf_files(
            self, file_list: list[str], prompt: str, detail: str = "auto"
            ) -> ChatResponse:
        '''
        複数の画像とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        prompt_content = self.create_text_content(text=prompt)
        pdf_content_list = []
        config = self.get_config()
        if not config:
            raise ValueError("LLMClientの設定が取得できませんでした。")

        for file_path in file_list:
            if config.features.use_custom_pdf_analyzer:
                logger.info(f"Using custom PDF analyzer for file: {file_path}")
                pdf_content = self._create_custom_pdf_content_from_file(file_path, detail)
            else:
                logger.info(f"Using standard PDF analyzer for file: {file_path}")
                pdf_content = self.create_pdf_content_from_file(file_path)
            pdf_content_list.extend(pdf_content)

        chat_message = ChatMessage(role="user", content=[prompt_content] + pdf_content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(
                messages=[chat_message]), chat_request_context=None)
        chat_response: ChatResponse = await self.chat(chat_request)
        return chat_response

    async def analyze_office_files(
            self, file_path_list: list[str], prompt: str, detail: str = "auto"
            ) -> ChatResponse:
        '''
        複数のOfficeドキュメントとプロンプトからドキュメント解析を行う。
        各ドキュメントのテキスト抽出、各ドキュメントの説明、プロンプト応答を生成して返す
        '''
        office_contents: list[ChatContent] = []
        for file_path in file_path_list:
            # Officeドキュメントを一時的にPDFに変換する
            pdf_content = self.create_office_content_from_file(
                file_path, detail=detail)

            office_contents.extend(pdf_content)

        prompt_content = self.create_text_content(text=prompt)

        chat_message = ChatMessage(role="user", content=[prompt_content] + office_contents)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None)
        response: ChatResponse = await self.chat(chat_request)
        return response

    async def analyze_documents_data(
            self, document_type_list: list[FileUtilDocument], prompt: str, detail: str = "auto"
            ) -> ChatResponse:
        '''
        複数の形式のドキュメントとプロンプトからドキュメント解析を行う。
        各ドキュメントのテキスト抽出、各ドキュメントの説明、プロンプト応答を生成して返す
        '''
        content_list = []
        for document_type in document_type_list:
            contents = self.create_multi_format_content(
                document_type, detail=detail)
            content_list.extend(contents)

        prompt_content = self.create_text_content(text=prompt)
        chat_message = ChatMessage(role="user", content=[prompt_content] + content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None)
        chat_response: ChatResponse = await self.chat(chat_request)
        return chat_response


    async def analyze_files(
            self, file_path_list: list[str], prompt: str, detail: str = "auto"
            ) -> ChatResponse:
        '''
        複数の形式のドキュメントとプロンプトからドキュメント解析を行う。
        各ドキュメントのテキスト抽出、各ドキュメントの説明、プロンプト応答を生成して返す
        '''
        content_list = []
        for file_path in file_path_list:
            contents = self.create_multi_format_contents_from_file(
                file_path, detail=detail)
            content_list.extend(contents)

        prompt_content = self.create_text_content(text=prompt)
        chat_message = ChatMessage(role="user", content=[prompt_content] + content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None)
        chat_response: ChatResponse = await self.chat(chat_request)
        return chat_response


    async def simple_chat(self, prompt: str,) -> str:
        '''
        簡易的なChatCompletionを実行する.
        引数として渡されたプロンプトをChatMessageに変換し、LLMに対してChatCompletionを実行する.
        その後、文字列を返す.
        Args:
            prompt (str): プロンプト文字列
        Returns:
            CompletionResponse: LLMからの応答
        '''
        chat_message = ChatMessage(
            role=self.get_user_role_name(),
            content=[self.create_text_content(prompt)]
        )
        chat_history = ChatHistory(messages=[chat_message])
        
        response = await self.chat(ChatRequest(chat_history=chat_history, chat_request_context=None))
        return response.output

    async def chat(
            self, chat_request: ChatRequest, **kwargs
            ) -> ChatResponse:
        '''
        LLMに対してChatCompletionを実行する.
        引数として渡されたChatMessageの前処理を実施した上で、LLMに対してChatCompletionを実行する.
        その後、後処理を実施し、CompletionResponseを返す.
        chat_messageがNoneの場合は、chat_historyから最後のユーザーメッセージを取得して処理を実施する.

        Args:
            chat_request (ChatRequest): チャットリクエスト
        Returns:
            ChatResponse: LLMからの応答
        '''
        if chat_request is None:
            raise ValueError("chat_request must be provided")
        if chat_request.chat_request_context:
            return await self.__chat_with_request_context__(
                chat_request, chat_request.chat_request_context, **kwargs
            )
        else:
            return await self.__normal_chat__(
                chat_request, **kwargs
            )   

    @abstractmethod
    async def __chat_with_request_context__(
        self, chat_request: ChatRequest, chat_request_context: ChatRequestContext, **kwargs
    ) -> ChatResponse:
        pass

    @abstractmethod
    async def __normal_chat__(
        self, chat_request: ChatRequest, **kwargs
    ) -> ChatResponse:
        pass

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
            content for chat_message in chat_message_list for content in chat_message.content if LLMClientUtil.is_text_content(content)
            ]
        if len(text_type_contents) == 0:
            return __insert_prompt_template__(chat_message_list, request_context)

        # text以外のcontentを抽出する
        non_text_contents = [
            content for chat_message in chat_message_list for content in chat_message.content if not LLMClientUtil.is_text_content(content)
        ]

        text_result_chat_message_list: list[ChatMessage] = []        
        # textを結合
        combined_text = "\n".join([text_content.params.get("text", "") for text_content in text_type_contents] )
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
                content for content in chat_message.content if LLMClientUtil.is_image_content(content)
            ]

            # 画像が無い、または分割不要ならそのまま
            if len(image_url_contents) == 0 or len(image_url_contents) <= max_images:
                result_chat_message_list.append(chat_message)
                continue

            # 分割時は、テキスト＋その他（画像以外）を各分割メッセージに維持する
            text_contents = [
                content for content in chat_message.content if LLMClientUtil.is_text_content(content)
            ]
            other_contents = [
                content
                for content in chat_message.content
                if (not LLMClientUtil.is_text_content(content)) and (not LLMClientUtil.is_image_content(content))
            ]
            base_contents = text_contents + other_contents

            for i in range(0, len(image_url_contents), max_images):
                split_image_url_contents: list[ChatContent] = image_url_contents[i : i + max_images]
                split_contents = base_contents + split_image_url_contents

                split_chat_message = ChatMessage(role=chat_message.role, content=split_contents)
                result_chat_message_list.append(split_chat_message)

        return result_chat_message_list

    async def __postprocess_messages__(
        self,
        chat_responses: list[ChatResponse],
        request_context: ChatRequestContext
    ) -> ChatResponse:
        '''
        request_contextの内容に従い、メッセージの後処理を実施する
        * split_modeがsplit_and_summarizeの場合、
            ChatMessageのcontentのうち、typeがtextの要素を抽出し、
            summarize_prompt_textを用いて要約を実施する
            summarize_prompt_textが空文字列の場合は例外をスローする
        Args:
            chat_responses (list[CompletionResponse]): 後処理対象のCompletionResponseリスト
            request_context (ChatRequestContext): 後処理の設定情報
        Returns:
            ChatMessage: 後処理後のChatMessage
        '''
        if request_context.split_mode != ChatRequestContext.split_mode_name_split_and_summarize:
            # chat_responsesのサイズが1の場合はそのまま返す
            if len(chat_responses) == 1:
                return chat_responses[0]
            
            # split_modeがsplit_and_summarize以外の場合は、各テキストの冒頭に[answer_part_i]を付与して結合する
            result_text = ""
            for i, chat_response in enumerate(chat_responses):
                result_text += f"[answer_part_{i+1}]\n" + chat_response.output + "\n"
            return ChatResponse(
                messages=[
                    ChatMessage(role="assistant", content=[
                        ChatContent(params={"type": "text", "text": result_text.strip()})
                        ]
                        )]
                )
        
        if not request_context.summarize_prompt_text:
            raise ValueError("summarize_prompt_text must be set when split_mode is 'split_and_summarize'")
        # split_modeがsplit_and_summarizeの場合は要約を実施する
        summmarize_request_text = request_context.summarize_prompt_text + "\n"
        for chat_response in chat_responses:
            summmarize_request_text += chat_response.output + "\n"

        # request_contextはsplit_modeをnoneに設定して要約を実施する
        request_context = ChatRequestContext(
            split_mode=ChatRequestContext.split_mode_name_none,
            summarize_prompt_text=request_context.summarize_prompt_text
        )
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(
                messages=[]
                ), chat_request_context=request_context)
        client = self.create(self.get_config())
        text_content = client.create_text_content(summmarize_request_text)
        message = ChatMessage(
            role=self.get_user_role_name(),
            content=[text_content]
        )
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[message], ), chat_request_context=None)
        summarize_response = await client.chat(chat_request)
        return summarize_response


class LLMClient(LLMClientBase):

    llm_config: AiChatUtilConfig
    concurrency_limit: int = 16

    default_timeout_seconds: float = 300.0
    
    def __init__(self, llm_config: AiChatUtilConfig, use_mcp: bool = False):

        self.llm_config = llm_config
        self.use_mcp = use_mcp

        # config.yml の non-secret 設定を既定値として採用
        try:
            self.default_timeout_seconds = float(getattr(self.llm_config.llm, "timeout_seconds", 60.0) or 60.0)
        except (TypeError, ValueError):
            raise ValueError(
                f"llm.timeout_seconds は数値である必要があります: {getattr(self.llm_config.llm, 'timeout_seconds', None)!r}"
            )

    def get_config(self) -> AiChatUtilConfig:
        return self.llm_config

    def create(
        self, llm_config: AiChatUtilConfig | None = None, use_mcp: bool = False
        ) -> "LLMClient":
        if llm_config is None:
            llm_config = get_runtime_config()
        return LLMClient(llm_config, use_mcp=use_mcp)

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


    async def _chat_completion_(self, chat_request: ChatRequest, **kwargs) -> ChatResponse:
        if self.use_mcp:
            return await LLMClientUtil.run_mcp_chat_completion(
                self.llm_config, 
                chat_request, 
                self.default_timeout_seconds, 
                **kwargs
            )
        else:
            return await LLMClientUtil.run_litellm_chat_completion(
                    self.llm_config, 
                    chat_request, 
                    self.default_timeout_seconds, 
                    **kwargs
            )

    async def __normal_chat__(self, chat_request: ChatRequest, **kwargs) -> ChatResponse:
        '''
        LLMに対してChatCompletionを実行する.
        引数として渡されたChatMessageをそのままLLMに対してChatCompletionを実行する.
        その後、CompletionResponseを返す.
        chat_messageがNoneの場合は、chat_historyから最後のユーザーメッセージを取得して処理を実施する.

        Args:
            chat_request (ChatRequest): チャットリクエスト
        Returns:
            ChatResponse: LLMからの応答
        '''
        # IMPORTANT:
        # chat() から渡される chat_message_list は self.chat_request.chat_history.messages と同一オブジェクトのことがある。
        # そのリストを反復しつつ add_message() で同じリストに追記すると、リストが伸び続けて無限ループする。
        # ここでは「既存履歴をそのまま使う」か「渡されたリストのコピーを履歴として採用」し、重複追加はしない。
        chat_response =  await self._chat_completion_(chat_request, **kwargs)
        text_content = self.create_text_content(chat_response.output)
        chat_request.chat_history.add_message(ChatMessage(
            role=self.get_assistant_role_name(),
            content=[text_content]
        ))
        return chat_response

    async def __chat_with_request_context__(self, chat_request: ChatRequest, request_context: ChatRequestContext, **kwargs) -> ChatResponse:
        '''
        LLMに対してChatCompletionを実行する.
        引数として渡されたChatRequestの前処理を実施した上で、LLMに対してChatCompletionを実行する.
        その後、後処理を実施し、ChatResponseを返す.
        chat_requestがNoneの場合は、chat_historyから最後のユーザーメッセージを取得して処理を実施する.

        Args:
            chat_request (ChatRequest): チャットリクエスト
        Returns:
            ChatResponse: LLMからの応答
        '''
        if not chat_request:
            raise ValueError("chat_request must be provided")

        # 前処理を実行
        last_user_messages, previous_messages = self.__get_last_user_messages__(chat_request.chat_history)
        preprocessed_messages = self.__preprocess_text_message__(last_user_messages, request_context)
        preprocessed_messages = self.__preprocess_image_urls__(preprocessed_messages, request_context)

        # LLMに対してChatCompletionを実行. messageごとにasyncioのタスクを作成して実行する
        async def __process_message__(message_num: int, message: ChatMessage, previous_messages: list[ChatMessage]) -> tuple[int, ChatResponse]:
            client = self.create(self.llm_config)
            chat_request: ChatRequest = ChatRequest(chat_history=ChatHistory(messages=previous_messages), chat_request_context=request_context)
            
            chat_request.chat_history.add_message(message)
            chat_response =  await client._chat_completion_(chat_request, **kwargs)
            return (message_num, chat_response)
            
        chat_response_tuples: list[tuple[int, ChatResponse]] = []

        sem = asyncio.Semaphore(self.concurrency_limit)

        async def __run_one__(i: int, message: ChatMessage) -> tuple[int, ChatResponse]:
            async with sem:
                return await __process_message__(i, message, previous_messages)

        tasks = [asyncio.create_task(__run_one__(i, message)) for i, message in enumerate(preprocessed_messages)]
        chat_response_tuples = await asyncio.gather(*tasks)

        # message_numでソートしてCompletionResponseのリストを作成
        chat_response_tuples.sort(key=lambda x: x[0])
        chat_responses = [t[1] for t in chat_response_tuples]

        # 後処理を実行
        postprocessed_response = await self.__postprocess_messages__(chat_responses, request_context)

        # chat_historyにpreprocessed_messageとpostprocessed_responseを追加する
        for preprocessed_message in preprocessed_messages:
            
            chat_request.chat_history.add_message(preprocessed_message)

        text_content = self.create_text_content(postprocessed_response.output)
        response_message = ChatMessage(
            role=self.get_assistant_role_name(),
            content=[text_content]
        )
        chat_request.chat_history.add_message(response_message)

        return postprocessed_response

