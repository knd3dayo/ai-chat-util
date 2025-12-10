
from typing import Sequence
# 抽象クラス
from abc import ABC, abstractmethod
import os
import copy
import tiktoken
import asyncio
import base64
import tempfile

from openai import AsyncOpenAI, AsyncAzureOpenAI

from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.model import ChatHistory, ChatResponse, ChatRequestContext, ChatMessage, ChatContent
from ai_chat_util.util.office2pdf import Office2PDFUtil

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LLMClient(ABC):

    llm_config: LLMConfig = LLMConfig()
    chat_history: ChatHistory = ChatHistory()

    @abstractmethod
    async def _chat_completion_(self, **kwargs) ->  ChatResponse:
        pass

    @abstractmethod
    def _create_image_content_from_url_(cls, image_url: str) -> "ChatContent":
        pass

    @abstractmethod
    def _create_image_content_from_bytes_(cls, image_data: bytes) -> "ChatContent":
        pass
    
    @abstractmethod
    def _create_pdf_content_from_url_(self, file_url: str, filename: str) -> "ChatContent":
        pass
    
    @abstractmethod
    def _create_pdf_content_from_bytes_(self, file_data: bytes, filename: str) -> "ChatContent":
        pass

    @classmethod
    def create_text_content(cls, text: str) -> "ChatContent":
        params = {"type": "text", "text": text}
        return ChatContent(params=params)
    
    def create_image_content_from_url(self, image_url: str) -> "ChatContent":
        return self._create_image_content_from_url_(image_url)

    def create_image_content_from_bytes(self, image_data: bytes) -> "ChatContent":
        return self._create_image_content_from_bytes_(image_data)

    def create_image_content_from_file(self, file_path: str) -> "ChatContent":
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
        return self.create_image_content_from_bytes(image_data)
   
    def create_pdf_content_from_url(self, file_url: str, filename: str) -> "ChatContent":
        return self._create_pdf_content_from_url_(file_url, filename)

    def create_pdf_content_from_bytes(self, file_data: bytes, filename: str) -> "ChatContent":
        return self._create_pdf_content_from_bytes_(file_data, filename)

    def create_pdf_content_from_file(self, file_path: str) -> "ChatContent":
        with open(file_path, "rb") as pdf_file:
            file_data = pdf_file.read()
        filename = os.path.basename(file_path)
        return self.create_pdf_content_from_bytes(file_data, filename)

    @classmethod
    def create_llm_client(
        cls, llm_config: LLMConfig, 
        chat_history: ChatHistory = ChatHistory(), 
        request_context: ChatRequestContext = ChatRequestContext()
    ) -> 'LLMClient':
        if llm_config.llm_provider == "azure_openai":
            return AzureOpenAIClient(llm_config, chat_history, request_context)
        else:
            return OpenAIClient(llm_config, chat_history, request_context)

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

    async def analyze_image_files(self, image_path_list: list[str], prompt: str) -> str:
        '''
        複数の画像とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        prompt_content = self.create_text_content(text=prompt)
        image_content_list = []
        for image_path in image_path_list:
            image_content = self.create_image_content_from_file(image_path)
            image_content_list.append(image_content)

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_response: ChatResponse = await self.chat([chat_message],  request_context=None)
        return chat_response.output


    async def analyze_pdf_files(self, pdf_path_list: list[str], prompt: str) -> str:
        '''
        複数の画像とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        prompt_content = self.create_text_content(text=prompt)
        pdf_content_list = []
        for pdf_path in pdf_path_list:
            pdf_content = self.create_pdf_content_from_file(pdf_path)
            pdf_content_list.append(pdf_content)

        chat_message = ChatMessage(role="user", content=[prompt_content] + pdf_content_list)
        chat_response: ChatResponse = await self.chat([chat_message],  request_context=None)
        return chat_response.output

    async def analyze_office_document_files(self, file_path_list: list[str], prompt: str) -> str:
        '''
        複数のOfficeドキュメントとプロンプトからドキュメント解析を行う。各ドキュメントのテキスト抽出、各ドキュメントの説明、プロンプト応答を生成して返す
        '''
        prompt_content = self.create_text_content(text=prompt)
        pdf_content_list = []
        for file_path in file_path_list:
            # Officeドキュメントを一時的にPDFに変換する
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdf_path = os.path.join(temp_dir, "temp_converted.pdf")
                Office2PDFUtil.create_pdf_from_document(
                    input_path=file_path,
                    output_path=temp_pdf_path
                )
                pdf_content = self.create_pdf_content_from_file(temp_pdf_path)
                pdf_content_list.append(pdf_content)

        chat_message = ChatMessage(role="user", content=[prompt_content] + pdf_content_list)
        chat_response: ChatResponse = await self.chat([chat_message],  request_context=None)
        return chat_response.output

    async def simple_image_analysis(self, image_path: str, prompt: str) -> ChatResponse:
        '''
        簡易的な画像解析を実行する.
        引数として渡された画像ファイルパスとプロンプトをChatMessageに変換し、LLMに対してChatCompletionを実行する.
        その後、CompletionResponseを返す.

        Args:
            image_path (str): 画像ファイルのパス
            prompt (str): プロンプト文字列
        Returns:
            CompletionResponse: LLMからの応答
        '''
        prompt_content = self.create_text_content(text=prompt)
        image_content = self.create_image_content_from_file(image_path)
        chat_message = ChatMessage(
            role=ChatHistory.user_role_name,
            content=[prompt_content, image_content]
        )
        return await self.chat([chat_message], request_context=None)

    async def simple_pdf_analysis(self, pdf_path: str, prompt: str) -> ChatResponse:
        '''
        簡易的なPDF解析を実行する.
        引数として渡されたPDFファイルパスとプロンプトをChatMessageに変換し、LLMに対してChatCompletionを実行する.
        その後、CompletionResponseを返す.

        Args:
            pdf_path (str): PDFファイルのパス
            prompt (str): プロンプト文字列
        Returns:
            CompletionResponse: LLMからの応答
        '''
        prompt_content = self.create_text_content(text=prompt)
        pdf_content = self.create_pdf_content_from_file(pdf_path)
        chat_message = ChatMessage(
            role=ChatHistory.user_role_name,
            content=[prompt_content, pdf_content]
        )
        return await self.chat([chat_message], request_context=None)

    async def simple_office_document_analysis(self, file_path: str, prompt: str) -> ChatResponse:
        '''
        簡易的なOfficeドキュメント解析を実行する.
        引数として渡されたOfficeドキュメントファイルパスとプロンプトをChatMessageに変換し、LLMに対してChatCompletionを実行する.
        その後、CompletionResponseを返す.

        Args:
            file_path (str): Officeドキュメントファイルのパス
            prompt (str): プロンプト文字列
        Returns:
            CompletionResponse: LLMからの応答
        '''
        prompt_content = self.create_text_content(text=prompt)
        # Officeドキュメントを一時的にPDFに変換する
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = os.path.join(temp_dir, "temp_converted.pdf")
            Office2PDFUtil.create_pdf_from_document(
                input_path=file_path,
                output_path=temp_pdf_path
            )
            pdf_content = self.create_pdf_content_from_file(temp_pdf_path)
            chat_message = ChatMessage(
                role=ChatHistory.user_role_name,
                content=[prompt_content, pdf_content]
            )
            return await self.chat([chat_message], request_context=None)

    async def simple_chat(self, prompt: str) -> ChatResponse:
        '''
        簡易的なChatCompletionを実行する.
        引数として渡されたプロンプトをChatMessageに変換し、LLMに対してChatCompletionを実行する.
        その後、CompletionResponseを返す.

        Args:
            prompt (str): プロンプト文字列
        Returns:
            CompletionResponse: LLMからの応答
        '''
        chat_message = ChatMessage(
            role=ChatHistory.user_role_name,
            content=[self.create_text_content(prompt)]
        )
        return await self.chat([chat_message], request_context=None)

    async def chat(self, chat_message_list: list[ChatMessage] = [], request_context: ChatRequestContext|None = None, **kwargs) -> ChatResponse:
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
            chat_messages = self.chat_history.get_last_user_messages()
            if len(chat_messages) == 0:
                raise ValueError("No chat messages to process.")
        else:
            chat_messages = chat_message_list

        for chat_message in chat_messages:
            self.chat_history.add_message(chat_message)
        chat_response =  await self._chat_completion_(**kwargs)
        text_content = self.create_text_content(chat_response.output)
        self.chat_history.add_message(ChatMessage(
            role=ChatHistory.assistant_role_name,
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
            chat_messages = self.chat_history.get_last_user_messages()
            if len(chat_messages) == 0:
                raise ValueError("No chat messages to process.")
        else:
            chat_messages = chat_message_list

        # 前処理を実行
        preprocessed_messages: list[ChatMessage] = chat_messages
        preprocessed_messages = self.__preprocess_text_message(preprocessed_messages, request_context)
        preprocessed_messages = self.__preprocess_image_urls(preprocessed_messages, request_context)

        # LLMに対してChatCompletionを実行. messageごとにasyncioのタスクを作成して実行する
        async def __process_message(message_num: int, message: ChatMessage) -> tuple[int, ChatResponse]:
            client = LLMClient.create_llm_client(
                self.llm_config, chat_history=copy.deepcopy(self.chat_history), request_context=request_context)
            
            client.chat_history.add_message(message)
            chat_response =  await client._chat_completion_(**kwargs)
            return (message_num, chat_response)
            
        chat_response_tuples: list[tuple[int, ChatResponse]] = []

        tasks = [__process_message(i, message) for i, message in enumerate(preprocessed_messages)]
        async with asyncio.Semaphore(16):
            chat_response_tuples = await asyncio.gather(*tasks)

        # message_numでソートしてCompletionResponseのリストを作成
        chat_response_tuples.sort(key=lambda x: x[0])
        chat_responses = [t[1] for t in chat_response_tuples]

        for preprocessed_message in preprocessed_messages:
            # 
            client = LLMClient.create_llm_client(
                self.llm_config, chat_history=copy.deepcopy(self.chat_history), request_context=request_context)
            
            client.chat_history.add_message(preprocessed_message)
            chat_response =  await client._chat_completion_(**kwargs)
            chat_responses.append(chat_response)

        # 後処理を実行
        postprocessed_response = await self.__postprocess_messages(chat_responses, request_context)

        # chat_historyにpreprocessed_messageとpostprocessed_responseを追加する
        for preprocessed_message in preprocessed_messages:
            self.chat_history.add_message(preprocessed_message)

        text_content = self.create_text_content(postprocessed_response.output)
        response_message = ChatMessage(
            role=ChatHistory.assistant_role_name,
            content=[text_content]
        )
        self.chat_history.add_message(response_message)

        return postprocessed_response
    
    def __preprocess_text_message(
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
        def __insert_prompt_template(
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
            return __insert_prompt_template(chat_message_list, request_context)

        if not request_context.prompt_template_text:
            raise ValueError("prompt_template_text must be set when split_mode is not 'None'")

        split_message_length = request_context.split_message_length
        if split_message_length <= 0:
            # 分割しない設定の場合はそのまま返す
            return __insert_prompt_template(chat_message_list, request_context)


        # textタイプのcontentを抽出する
        text_type_contents = [ 
            content for chat_message in chat_message_list for content in chat_message.content if content.params.get("type") == "text"
            ]
        if len(text_type_contents) == 0:
            return __insert_prompt_template(chat_message_list, request_context)

        # text以外のcontentを抽出する
        non_text_contents = [
            content for chat_message in chat_message_list for content in chat_message.content if content.params.get("type") != "text"
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
                    role=ChatHistory.user_role_name,
                    content=[split_content]
                )
                # textタイプ以外のcontentを追加する
                for non_text_content in non_text_contents:
                    chat_message.content.append(non_text_content)

                text_result_chat_message_list.append(chat_message)

        return text_result_chat_message_list        

    def __preprocess_image_urls(
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
                content for content in chat_message.content if content.params.get("type") == "image_url"
            ]
            if len(image_url_contents) == 0:
                # chat_messageをそのまま追加
                result_chat_message_list.append(chat_message)
                continue

            # image_urlタイプのcontentを抽出する
            image_urls = [content.params.get("image_url") for content in image_url_contents if content.params.get("image_url")]

            # textタイプのcontentを抽出する
            text_contents = [
                content for content in chat_message.content if content.params.get("type") == "text" and content.params.get("text")
            ]
            for i in range(0, len(image_urls), max_images):
                split_image_urls: list[str|None] = image_urls[i:i + max_images]
                split_contents = text_contents + [self.create_image_content_from_url(url) for url in split_image_urls if url]
                
                split_chat_message = ChatMessage(
                    role=chat_message.role,
                    content=split_contents
                )
                result_chat_message_list.append(split_chat_message)

        return result_chat_message_list

    async def __postprocess_messages(
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

        client = LLMClient.create_llm_client(self.llm_config, request_context=request_context)
        text_content = client.create_text_content(summmarize_request_text)
        message = ChatMessage(
            role=ChatHistory.user_role_name,
            content=[text_content]
        )
        summarize_response = await client.chat([message])
        return summarize_response


class OpenAIClient(LLMClient):
    def __init__(self, llm_config: LLMConfig, chat_history: ChatHistory = ChatHistory(), request_context: ChatRequestContext = ChatRequestContext()):
        if llm_config.base_url:
            self.client = AsyncOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        else:
            self.client = AsyncOpenAI(api_key=llm_config.api_key)

        self.model = llm_config.completion_model

        self.chat_history = chat_history
        self.request_context = request_context

    async def _chat_completion_(self,  **kwargs) -> ChatResponse:
        message_dict_list = [msg.model_dump() for msg in self.chat_history.messages]
        response = await self.client.chat.completions.create(
            model=self.chat_history.model,
            messages=message_dict_list,
            **kwargs
        )
        return ChatResponse(output=response.choices[0].message.content or "")

    def _create_image_content_from_url_(self, image_url: str) -> "ChatContent":
        params = {"type": "image_url", "image_url": {"url": image_url}}
        return ChatContent(params=params)
    
    def _create_image_content_from_bytes_(self, image_data: bytes) -> "ChatContent":
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/png;base64,{base64_image}"
        return self._create_image_content_from_url_(image_url)

    def _create_pdf_content_from_url_(self, file_url: str, filename: str) -> "ChatContent":
        params = {"type": "file", "file": {"file_data": file_url, "filename": filename}}
        return ChatContent(params=params)
    
    def _create_pdf_content_from_bytes_(self, file_data: bytes, filename: str) -> ChatContent:
        base64_file = base64.b64encode(file_data).decode('utf-8')
        file_url = f"data:application/pdf;base64,{base64_file}"    
        return self._create_pdf_content_from_url_(file_url, filename)


class AzureOpenAIClient(OpenAIClient):
    def __init__(self, llm_config: LLMConfig, chat_history: ChatHistory = ChatHistory(), request_context: ChatRequestContext = ChatRequestContext()):
        if llm_config.base_url:
            self.client = AsyncAzureOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        elif llm_config.api_version and llm_config.endpoint:
            self.client = AsyncAzureOpenAI(api_key=llm_config.api_key, azure_endpoint=llm_config.endpoint, api_version=llm_config.api_version)
        else:
            raise ValueError("Either base_url or both api_version and endpoint must be provided.")

        self.model = llm_config.completion_model

        self.chat_history = chat_history
        self.request_context = request_context

    async def _chat_completion_(self, **kwargs) -> ChatResponse:
        
        message_dict_list = [msg.model_dump() for msg in self.chat_history.messages]
        response = await self.client.chat.completions.create(
            model=self.chat_history.model,
            messages=message_dict_list,
            **kwargs
        )
        return ChatResponse(output=response.choices[0].message.content or "")

    # Azureの場合は、CompletionでのPDF処理が未対応の模様。PDFファイルからテキストと画像を抽出して、
    # 生成AIに渡すという独自実装を行う。
    async def analyze_pdf_files(self, pdf_path_list: list[str], prompt: str) -> str:
        import ai_chat_util.util.pdf_util as pdf_util
        prompt_content = self.create_text_content(text=prompt)
        pdf_messages = []
        for pdf_path in pdf_path_list:
            # PDFからテキストと画像を抽出
            pdf_elements = pdf_util.extract_pdf_content(pdf_path)
            pdf_contents = []
            for element in pdf_elements:
                if element["type"] == "text":
                    text_content = self.create_text_content(text=element["text"])
                    pdf_contents.append(text_content)
                elif element["type"] == "image_url":
                    image_content = self.create_image_content_from_url(element["image_url"])
                    pdf_contents.append(image_content)

            page_info_content = self.create_text_content(text=f"PDFファイル: {os.path.basename(pdf_path)} の内容を以下に示します。")
            pdf_messages.append(ChatMessage(
                role=ChatHistory.user_role_name,
                content=[page_info_content] + pdf_contents
            ))
        chat_message = ChatMessage(role="user", content=[prompt_content])
        response: ChatResponse = await self.chat([chat_message] + pdf_messages,  request_context=None)
        return response.output


if __name__ == "__main__":
    import sys
    # 画像ファイルを指定して、画像分析を行う
    async def main():
        llm_config = LLMConfig()
        llm_client = LLMClient.create_llm_client(llm_config)

        image_paths = sys.argv[1:]
        prompt = "上記のPDFに関して、内容を説明してください。また、画像に含まれるテキストを抽出してください。"

        response_text = await llm_client.analyze_pdf_files(image_paths, prompt)
        print("=== Response ===")
        print(response_text)


    asyncio.run(main())
