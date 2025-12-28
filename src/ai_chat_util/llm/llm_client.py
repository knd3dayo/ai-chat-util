# 抽象クラス
from abc import ABC, abstractmethod
import os
import uuid
import copy
import tiktoken
import asyncio
import base64
import tempfile
import atexit
import requests

from openai import AsyncOpenAI, AsyncAzureOpenAI

from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.model import ChatHistory, ChatResponse, ChatRequestContext, ChatMessage, ChatContent, RequestModel
from ai_chat_util.util.office2pdf import Office2PDFUtil
from file_util.model import DocumentType

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LLMClient(ABC):


    llm_config: LLMConfig = LLMConfig()
    chat_history: ChatHistory = ChatHistory()

    concurrency_limit: int = 16
    
    @abstractmethod
    def create(
        cls, llm_config: LLMConfig = LLMConfig(), 
        chat_history: ChatHistory = ChatHistory(), 
        request_context: ChatRequestContext = ChatRequestContext()
    ) -> "LLMClient":
        pass

    @abstractmethod
    def get_user_role_name(self) -> str:
        pass

    @abstractmethod
    def get_assistant_role_name(self) -> str:
        pass

    @abstractmethod
    def get_system_role_name(self) -> str:
        pass

    @abstractmethod
    async def _chat_completion_(self, **kwargs) ->  ChatResponse:
        pass

    @abstractmethod
    def _create_image_content_(cls, image_data: bytes, detail: str) -> "ChatContent":
        pass
    
    @abstractmethod
    def _create_pdf_content_(self, file_data: bytes, filename: str) -> "ChatContent":
        pass

    @abstractmethod
    def _is_text_content_(self, content: ChatContent) -> bool:
        pass

    @abstractmethod
    def _is_image_content_(self, content: ChatContent) -> bool:
        pass

    @abstractmethod    
    def _is_file_content_(self, content: ChatContent) -> bool:
        pass

    @classmethod
    def get_token_count(cls, model: str, input_text: str) -> int:
        # completion_modelに対応するencoderを取得する
        # 暫定処理 
        # "gpt-4.1-": "o200k_base",  # e.g., gpt-4.1-nano, gpt-4.1-mini
        # "gpt-4.5-": "o200k_base", # e.g., gpt-4.5-preview
        if model.startswith("gpt-41") or model.startswith("gpt-4.1") or model.startswith("gpt-4.5"):
            encoder = tiktoken.get_encoding("o200k_base")
        else:
            encoder = tiktoken.encoding_for_model(model)
        # token数を取得する
        return len(encoder.encode(input_text))

    @classmethod
    def download_files(cls, urls: list[RequestModel], download_dir: str) -> list[str]:
        """
        Download files from the given URLs to the specified directory.
        Returns a list of file paths where the files are saved.
        """
        file_paths = []
        for item in urls:
            res = requests.get(url=item.url, headers=item.headers)
            file_path = os.path.join(download_dir, os.path.basename(item.url))
            with open(file_path, "wb") as f:
                f.write(res.content)
            file_paths.append(file_path)
        return file_paths


    def create_text_content(self, text: str) -> "ChatContent":
        params = {"type": "text", "text": text}
        return ChatContent(params=params)


    def create_image_content_from_file(self, file_path: str, detail: str) -> "ChatContent":
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
        return self._create_image_content_(image_data, detail)

    def create_image_content_from_url(self, file_url: RequestModel, detail: str) -> "ChatContent":
        tmpdir = tempfile.TemporaryDirectory()
        atexit.register(tmpdir.cleanup)

        file_paths = self.download_files([file_url], tmpdir.name)
        with open(file_paths[0], "rb") as image_file:
            image_data = image_file.read()
        return self._create_image_content_(image_data, detail)
    
    def create_pdf_content_from_file(self, file_path: str) -> list["ChatContent"]:
        with open(file_path, "rb") as pdf_file:
            file_data = pdf_file.read()
        return [self._create_pdf_content_(file_data, file_path)]

    def create_pdf_content_from_url(self, file_url: str, filename: str) -> "ChatContent":
        tmpdir = tempfile.TemporaryDirectory()
        atexit.register(tmpdir.cleanup)

        file_paths = self.download_files([RequestModel(url=file_url)], tmpdir.name)
        with open(file_paths[0], "rb") as pdf_file:
            file_data = pdf_file.read()
        return self._create_pdf_content_(file_data, filename)

    def create_custom_pdf_contents_from_file(self, file_path: str, detail: str = "auto") -> list["ChatContent"]:
        '''
        PDFファイルのバイトデータから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''
        import ai_chat_util.util.pdf_util as pdf_util

        page_info_content = self.create_text_content(text=f"PDFファイル: {file_path} の内容を以下に示します。")
        pdf_contents = [page_info_content]
        # PDFからテキストと画像を抽出
        pdf_elements = pdf_util.extract_pdf_content(file_path)
        for element in pdf_elements:
            if element["type"] == "text":
                text_content = self.create_text_content(text=element["text"])
                pdf_contents.append(text_content)
            elif element["type"] == "image":
                image_content = self._create_image_content_(element["bytes"], detail)
                pdf_contents.append(image_content)

        return pdf_contents
    
    def create_custom_pdf_contents_from_url(self, file_url: str, filename: str, detail: str = "auto") -> list["ChatContent"]:
        '''
        PDFファイルのバイトデータから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''
        import ai_chat_util.util.pdf_util as pdf_util

        page_info_content = self.create_text_content(text=f"PDFファイル: {filename} の内容を以下に示します。")
        pdf_contents = [page_info_content]

        tmpdir = tempfile.TemporaryDirectory()
        atexit.register(tmpdir.cleanup)

        file_paths = self.download_files([RequestModel(url=file_url)], tmpdir.name)

        for file_path in file_paths:
            pdf_content = self.create_custom_pdf_contents_from_file(file_path, detail)
            pdf_contents.extend(pdf_content)

        return pdf_contents

    def create_office_content_from_file(
            self, file_path: str, use_custom_pdf_analyzer: bool = False, detail: str = "auto"
            ) -> list["ChatContent"]:
        '''
        複数のOfficeドキュメントとプロンプトからドキュメント解析を行う。各ドキュメントのテキスト抽出、各ドキュメントの説明、プロンプト応答を生成して返す
        '''
        # Officeドキュメントを一時的にPDFに変換する
        temp_dir = tempfile.TemporaryDirectory()
        atexit.register(temp_dir.cleanup)
        temp_file_path = os.path.join(temp_dir.name, f"{os.path.basename(file_path)}_{uuid.uuid4()}.pdf")
        Office2PDFUtil.create_pdf_from_document(
            input_path=file_path,
            output_path=temp_file_path
        )
        # 元ファイルからPDFに変換した旨の説明を追加
        explanation_text = f"""
            {temp_file_path}は、元のOfficeドキュメント: {file_path} をPDFに変換したものです。
            ユーザーにどのファイルを元にしたのかを伝えるため、回答を行う際には、元のファイル名を使用してください。
            """
        explanation_content = self.create_text_content(text=explanation_text)
        pdf_contents = [explanation_content]

        if use_custom_pdf_analyzer:
            pdf_contents.extend(self.create_custom_pdf_contents_from_file(temp_file_path, detail))
        else:
            pdf_contents.extend(self.create_pdf_content_from_file(temp_file_path))

        return pdf_contents

    def create_office_content_from_url(self, file_url: str, use_custom_pdf_analyzer: bool = False, detail: str = "auto") -> list["ChatContent"]:
        '''
        複数のOfficeドキュメントとプロンプトからドキュメント解析を行う。各ドキュメントのテキスト抽出、各ドキュメントの説明、プロンプト応答を生成して返す
        '''
        tmpdir = tempfile.TemporaryDirectory()
        atexit.register(tmpdir.cleanup)

        file_paths = self.download_files([RequestModel(url=file_url)], tmpdir.name)

        office_contents = []
        for file_path in file_paths:
            content = self.create_office_content_from_file(
                file_path, use_custom_pdf_analyzer=use_custom_pdf_analyzer, detail=detail)
            office_contents.extend(content)

        return office_contents

    def create_multi_format_contents_from_file(
            self, file_path: str, use_custom_pdf_analyzer: bool = False, detail: str = "auto"
            ) -> list["ChatContent"]:
        '''
        複数形式ファイルから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''
            
        document_type = DocumentType(document_path=file_path)

        if document_type.is_text():
            with open(file_path, "r", encoding="utf-8") as text_file:
                text_data = text_file.read()
            return [self.create_text_content(text_data)]

        if document_type.is_image():
            return [self.create_image_content_from_file(file_path, detail)]

        if document_type.is_pdf():
            if use_custom_pdf_analyzer:
                return self.create_custom_pdf_contents_from_file(file_path, detail)
            else:
                return self.create_pdf_content_from_file(file_path)

        if document_type.is_office_document():
            return self.create_office_content_from_file(
                file_path, use_custom_pdf_analyzer=use_custom_pdf_analyzer, detail=detail)
        
        raise ValueError(f"Unsupported document type for file: {file_path}")    

    def create_multi_format_contents_from_url(
            self, file_url: str, use_custom_pdf_analyzer: bool = False, detail: str = "auto"
            ) -> list["ChatContent"]:
        '''
        複数形式ファイルから、テキスト抽出と画像抽出を行い、ChatContentのリストを生成して返す
        '''
        tmpdir = tempfile.TemporaryDirectory()
        atexit.register(tmpdir.cleanup)
        file_paths = self.download_files([RequestModel(url=file_url)], tmpdir.name)

        return self.create_multi_format_contents_from_file(
            file_paths[0], use_custom_pdf_analyzer=use_custom_pdf_analyzer, detail=detail
        )


    async def analyze_image_files(self, file_list: list[str], prompt: str, detail: str) -> ChatResponse:
        '''
        複数の画像とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        prompt_content = self.create_text_content(text=prompt)
        image_content_list = []
        for image_path in file_list:
            image_content = self.create_image_content_from_file(image_path, detail)
            image_content_list.append(image_content)

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_response: ChatResponse = await self.chat([chat_message],  request_context=None)
        return chat_response
    
    async def analyze_image_urls(self, image_url_list: list[RequestModel], prompt: str, detail: str) -> ChatResponse:
        '''
        複数の画像URLとプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        prompt_content = self.create_text_content(text=prompt)
        image_content_list = []
        for image_url in image_url_list:
            image_content = self.create_image_content_from_url(image_url, detail)
            image_content_list.append(image_content)

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_response: ChatResponse = await self.chat([chat_message],  request_context=None)
        return chat_response
    

    async def analyze_pdf_files(
            self, file_list: list[str], prompt: str, detail: str = "auto"
            ) -> ChatResponse:
        '''
        複数の画像とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        prompt_content = self.create_text_content(text=prompt)
        pdf_content_list = []
        for file_path in file_list:
            if self.llm_config.use_custom_pdf_analyzer:
                logger.info(f"Using custom PDF analyzer for file: {file_path}")
                pdf_content = self.create_custom_pdf_contents_from_file(file_path, detail)
            else:
                logger.info(f"Using standard PDF analyzer for file: {file_path}")
                pdf_content = self.create_pdf_content_from_file(file_path)
            pdf_content_list.extend(pdf_content)

        chat_message = ChatMessage(role="user", content=[prompt_content] + pdf_content_list)
        chat_response: ChatResponse = await self.chat([chat_message],  request_context=None)
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
                file_path, use_custom_pdf_analyzer=self.llm_config.use_custom_pdf_analyzer, detail=detail)

            office_contents.extend(pdf_content)

        prompt_content = self.create_text_content(text=prompt)

        chat_message = ChatMessage(role="user", content=[prompt_content] + office_contents)
        response: ChatResponse = await self.chat([chat_message],  request_context=None)
        return response

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
                file_path, use_custom_pdf_analyzer=self.llm_config.use_custom_pdf_analyzer, detail=detail)
            content_list.extend(contents)

        prompt_content = self.create_text_content(text=prompt)
        chat_message = ChatMessage(role="user", content=[prompt_content] + content_list)
        chat_response: ChatResponse = await self.chat([chat_message],  request_context=None)
        return chat_response

    async def simple_chat(self, prompt: str) -> str:
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
        response = await self.chat([chat_message], request_context=None)
        return response.output

    async def chat(
            self, chat_message_list: list[ChatMessage] = [], request_context: ChatRequestContext|None = None, **kwargs
            ) -> ChatResponse:
        '''
        LLMに対してChatCompletionを実行する.
        引数として渡されたChatMessageの前処理を実施した上で、LLMに対してChatCompletionを実行する.
        その後、後処理を実施し、CompletionResponseを返す.
        chat_messageがNoneの場合は、chat_historyから最後のユーザーメッセージを取得して処理を実施する.

        Args:
            chat_message (ChatMessage): チャットメッセージ
        Returns:
            CompletionResponse: LLMからの応答
        '''
        if request_context:
            return await self.__chat_with_request_context__(
                chat_message_list, request_context, **kwargs
            )
        else:
            return await self.__normal_chat__(
                chat_message_list, **kwargs
            )   

    async def __normal_chat__(self, chat_message_list: list[ChatMessage] = [], **kwargs) -> ChatResponse:
        '''
        LLMに対してChatCompletionを実行する.
        引数として渡されたChatMessageをそのままLLMに対してChatCompletionを実行する.
        その後、CompletionResponseを返す.
        chat_messageがNoneの場合は、chat_historyから最後のユーザーメッセージを取得して処理を実施する.

        Args:
            chat_message (ChatMessage): チャットメッセージ
        Returns:
            CompletionResponse: LLMからの応答
        '''
        if len(chat_message_list) == 0:
            chat_messages = self.chat_history.get_last_role_messages(self.get_user_role_name())
            if len(chat_messages) == 0:
                raise ValueError("No chat messages to process.")
        else:
            chat_messages = chat_message_list

        for chat_message in chat_messages:
            self.chat_history.add_message(chat_message)
        chat_response =  await self._chat_completion_(**kwargs)
        text_content = self.create_text_content(chat_response.output)
        self.chat_history.add_message(ChatMessage(
            role=self.get_assistant_role_name(),
            content=[text_content]
        ))
        return chat_response

    async def __chat_with_request_context__(self, chat_message_list: list[ChatMessage], request_context: ChatRequestContext, **kwargs) -> ChatResponse:
        '''
        LLMに対してChatCompletionを実行する.
        引数として渡されたChatMessageの前処理を実施した上で、LLMに対してChatCompletionを実行する.
        その後、後処理を実施し、CompletionResponseを返す.
        chat_messageがNoneの場合は、chat_historyから最後のユーザーメッセージを取得して処理を実施する.

        Args:
            chat_message (ChatMessage): チャットメッセージ
        Returns:
            CompletionResponse: LLMからの応答
        '''
        if len(chat_message_list) == 0:
            chat_messages = self.chat_history.get_last_role_messages(self.get_user_role_name())
            if len(chat_messages) == 0:
                raise ValueError("No chat messages to process.")
        else:
            chat_messages = chat_message_list

        # 前処理を実行
        preprocessed_messages: list[ChatMessage] = chat_messages
        preprocessed_messages = self.__preprocess_text_message__(preprocessed_messages, request_context)
        preprocessed_messages = self.__preprocess_image_urls__(preprocessed_messages, request_context)

        # LLMに対してChatCompletionを実行. messageごとにasyncioのタスクを作成して実行する
        async def __process_message__(message_num: int, message: ChatMessage) -> tuple[int, ChatResponse]:
            client = self.create(
                self.llm_config, chat_history=copy.deepcopy(self.chat_history), request_context=request_context)
            
            client.chat_history.add_message(message)
            chat_response =  await client._chat_completion_(**kwargs)
            return (message_num, chat_response)
            
        chat_response_tuples: list[tuple[int, ChatResponse]] = []

        sem = asyncio.Semaphore(self.concurrency_limit)

        async def __run_one__(i: int, message: ChatMessage) -> tuple[int, ChatResponse]:
            async with sem:
                return await __process_message__(i, message)

        tasks = [asyncio.create_task(__run_one__(i, message)) for i, message in enumerate(preprocessed_messages)]
        chat_response_tuples = await asyncio.gather(*tasks)

        # message_numでソートしてCompletionResponseのリストを作成
        chat_response_tuples.sort(key=lambda x: x[0])
        chat_responses = [t[1] for t in chat_response_tuples]

        for preprocessed_message in preprocessed_messages:
            # 
            client = self.create(
                self.llm_config, chat_history=copy.deepcopy(self.chat_history), request_context=request_context)
            
            client.chat_history.add_message(preprocessed_message)
            chat_response =  await client._chat_completion_(**kwargs)
            chat_responses.append(chat_response)

        # 後処理を実行
        postprocessed_response = await self.__postprocess_messages__(chat_responses, request_context)

        # chat_historyにpreprocessed_messageとpostprocessed_responseを追加する
        for preprocessed_message in preprocessed_messages:
            self.chat_history.add_message(preprocessed_message)

        text_content = self.create_text_content(postprocessed_response.output)
        response_message = ChatMessage(
            role=self.get_assistant_role_name(),
            content=[text_content]
        )
        self.chat_history.add_message(response_message)

        return postprocessed_response
    
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
            content for chat_message in chat_message_list for content in chat_message.content if self._is_text_content_(content)
            ]
        if len(text_type_contents) == 0:
            return __insert_prompt_template__(chat_message_list, request_context)

        # text以外のcontentを抽出する
        non_text_contents = [
            content for chat_message in chat_message_list for content in chat_message.content if self._is_text_content_(content) == False
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

        # messageごとに処理を実施
        for chat_message in chat_message_list:
            image_url_contents = [
                content for content in chat_message.content if self._is_image_content_(content)
            ]
            if len(image_url_contents) == 0:
                # chat_messageをそのまま追加
                result_chat_message_list.append(chat_message)
                continue

            # textタイプのcontentを抽出する
            text_contents = [
                content for content in chat_message.content if self._is_text_content_(content)
            ]
            for i in range(0, len(image_url_contents), max_images):
                split_image_url_contents: list[ChatContent] = image_url_contents[i:i + max_images]
                split_contents = text_contents + split_image_url_contents
                
                split_chat_message = ChatMessage(
                    role=chat_message.role,
                    content=split_contents
                )
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
            return ChatResponse(output=result_text.strip())
        
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

        client = self.create(self.llm_config, request_context=request_context)
        text_content = client.create_text_content(summmarize_request_text)
        message = ChatMessage(
            role=self.get_user_role_name(),
            content=[text_content]
        )
        summarize_response = await client.chat([message])
        return summarize_response

