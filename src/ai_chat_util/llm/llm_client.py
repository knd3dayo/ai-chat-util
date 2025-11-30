# 抽象クラス

from abc import ABC, abstractmethod
import copy
import tiktoken

from openai import AsyncOpenAI, AsyncAzureOpenAI

from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.model import CompletionRequest, CompletionResponse, ChatRequestContext, ChatMessage, ChatContent

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LLMClient(ABC):

    llm_config: LLMConfig = LLMConfig()
    chat_history: CompletionRequest = CompletionRequest()
    request_context: ChatRequestContext = ChatRequestContext()

    @abstractmethod
    async def _chat_completion(self, **kwargs) ->  CompletionResponse:
        pass

    @classmethod
    def create_llm_client(
        cls, llm_config: LLMConfig, 
        chat_history: CompletionRequest = CompletionRequest(), 
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

    async def run_chat(self, chat_message: ChatMessage |None = None) -> CompletionResponse:
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
        if not chat_message:
            chat_messages = self.chat_history.get_last_user_messages()
            if len(chat_messages) == 0:
                raise ValueError("No chat messages to process.")
        else:
            chat_messages = [chat_message]

             # 前処理を実行
        preprocessed_messages: list[ChatMessage] = chat_messages
        preprocessed_messages = self.__preprocess_text_message(preprocessed_messages, self.request_context)
        preprocessed_messages = self.__preprocess_image_urls(preprocessed_messages, self.request_context)

        # LLMに対してChatCompletionを実行. messageごとにasyncioのタスクを作成して実行する
        async def __process_message(message_num: int, message: ChatMessage) -> tuple[int, CompletionResponse]:
            client = LLMClient.create_llm_client(
                self.llm_config, chat_history=copy.deepcopy(self.chat_history), request_context=self.request_context)
            
            client.chat_history.add_message(message)
            chat_response =  await client._chat_completion()
            return (message_num, chat_response)
            
        chat_response_tuples: list[tuple[int, CompletionResponse]] = []

        tasks = [__process_message(i, message) for i, message in enumerate(preprocessed_messages)]
        async with asyncio.Semaphore(16):
            chat_response_tuples = await asyncio.gather(*tasks)

        # message_numでソートしてCompletionResponseのリストを作成
        chat_response_tuples.sort(key=lambda x: x[0])
        chat_responses = [t[1] for t in chat_response_tuples]

        for preprocessed_message in preprocessed_messages:
            # 
            client = LLMClient.create_llm_client(
                self.llm_config, chat_history=copy.deepcopy(self.chat_history), request_context=self.request_context)
            
            client.chat_history.add_message(preprocessed_message)
            chat_response =  await client._chat_completion()
            chat_responses.append(chat_response)

        # 後処理を実行
        postprocessed_response = await self.__postprocess_messages(chat_responses, self.request_context)

        # chat_historyにpreprocessed_messageとpostprocessed_responseを追加する
        for preprocessed_message in preprocessed_messages:
            self.chat_history.add_message(preprocessed_message)
        response_message = ChatMessage(
            role=CompletionRequest.assistant_role_name,
            content=[ChatContent(type="text", text=postprocessed_response.output)]
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
                    prompt_template_content = ChatContent(
                        type = "text",
                        text = request_context.prompt_template_text
                    )
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
            content for chat_message in chat_message_list for content in chat_message.content if content.type == "text"
            ]
        if len(text_type_contents) == 0:
            return __insert_prompt_template(chat_message_list, request_context)

        # text以外のcontentを抽出する
        non_text_contents = [
            content for chat_message in chat_message_list for content in chat_message.content if content.type != "text"
        ]

        text_result_chat_message_list: list[ChatMessage] = []        
        # textを結合
        combined_text = "\n".join([content.text for content in text_type_contents if content.text])
        # 文字数で分割する
        for i in range(0, len(combined_text), split_message_length):
            split_text = combined_text[i:i + split_message_length]
            split_contents = [ChatContent(type="text", text=f"{request_context.prompt_template_text}\n{split_text}")]
            for split_content in split_contents:
                chat_message = ChatMessage(
                    role=CompletionRequest.user_role_name,
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
                content for content in chat_message.content if content.type == "image_url"
            ]
            if len(image_url_contents) == 0:
                # chat_messageをそのまま追加
                result_chat_message_list.append(chat_message)
                continue

            # image_urlタイプのcontentを抽出する
            image_urls = [content.image_url for content in image_url_contents if content.image_url]

            # textタイプのcontentを抽出する
            text_contents = [
                content for content in chat_message.content if content.type == "text" and content.text
            ]
            for i in range(0, len(image_urls), max_images):
                split_image_urls = image_urls[i:i + max_images]
                
                split_contents = text_contents + [ChatContent(type="image_url", image_url=url) for url in split_image_urls]
                
                split_chat_message = ChatMessage(
                    role=chat_message.role,
                    content=split_contents
                )
                result_chat_message_list.append(split_chat_message)

        return result_chat_message_list

    async def __postprocess_messages(
        self,
        chat_responses: list[CompletionResponse],
        request_context: ChatRequestContext
    ) -> CompletionResponse:
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
            return CompletionResponse(output=result_text.strip())
        
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
        message = ChatMessage(
            role=CompletionRequest.user_role_name,
            content=[ChatContent(type="text", text=summmarize_request_text)]
        )
        summarize_response = await client.run_chat(message)
        return summarize_response

class AzureOpenAIClient(LLMClient):
    def __init__(self, llm_config: LLMConfig, chat_history: CompletionRequest = CompletionRequest(), request_context: ChatRequestContext = ChatRequestContext()):
        if llm_config.base_url:
            self.client = AsyncAzureOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        elif llm_config.api_version and llm_config.endpoint:
            self.client = AsyncAzureOpenAI(api_key=llm_config.api_key, azure_endpoint=llm_config.endpoint, api_version=llm_config.api_version)
        else:
            raise ValueError("Either base_url or both api_version and endpoint must be provided.")

        self.model = llm_config.completion_model

        self.chat_history = chat_history
        self.request_context = request_context

    async def _chat_completion(self, **kwargs) -> CompletionResponse:
        
        response = await self.client.chat.completions.create(
            model=self.chat_history.model,
            messages=self.chat_history.messages,
        )
        return CompletionResponse(output=response.choices[0].message.content or "")


class OpenAIClient(LLMClient):
    def __init__(self, llm_config: LLMConfig, chat_history: CompletionRequest = CompletionRequest(), request_context: ChatRequestContext = ChatRequestContext()):
        if llm_config.base_url:
            self.client = AsyncOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        else:
            self.client = AsyncOpenAI(api_key=llm_config.api_key)

        self.model = llm_config.completion_model

        self.chat_history = chat_history
        self.request_context = request_context

    async def _chat_completion(self,  **kwargs) -> CompletionResponse:
        response = await self.client.chat.completions.create(
            model=self.chat_history.model,
            messages=self.chat_history.messages,
        )
        return CompletionResponse(output=response.choices[0].message.content or "")

if __name__ == "__main__":
    import sys
    input_text = "こんにちは、AIチャットユーティリティのテストコードです。"
    promt_template_text = "以下の文章に基づいて回答してください。"
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            input_text = f.read()
        promt_template_text = "以下の文章を要約してください。"

    async def main():
        # テストコード
        client = LLMClient.create_llm_client(llm_config=LLMConfig())
        request_context = ChatRequestContext()
        request_context.split_mode = ChatRequestContext.split_mode_name_split_and_summarize
        request_context.split_message_length = 1000
        request_context.prompt_template_text = promt_template_text
        client.request_context = request_context

        message = ChatMessage(
            role=CompletionRequest.user_role_name,
            content=[ChatContent(type="text", text=input_text)]
        )
        response = await client.run_chat(message)
        print(response.output)\



    import asyncio
    asyncio.run(main())
