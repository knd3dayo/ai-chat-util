from __future__ import annotations

from abc import ABC, abstractmethod

from ai_chat_util.common.config.runtime import AiChatUtilConfig
from ai_chat_util.ai_chat_util_base.chat.model import (
    ChatContent,
    ChatHistory,
    ChatMessage,
    ChatRequest,
    ChatRequestContext,
    ChatResponse,
)

from .llm_messages_factory import LLMMessageContentFactoryBase
from .abstract_chat_client import AbstractChatClient


class ChatClientBase(AbstractChatClient, ABC):

    @abstractmethod
    def get_config(self) -> AiChatUtilConfig | None:
        pass

    @abstractmethod
    def get_message_factory(self) -> LLMMessageContentFactoryBase:
        pass

    @abstractmethod
    def create(
        self, llm_config: AiChatUtilConfig | None = None
    ) -> "ChatClientBase":
        pass

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
            role=self.get_message_factory().get_user_role_name(),
            content=[self.get_message_factory().create_text_content(prompt)]
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

        # NOTE:
        # ChatRequest.chat_request_context はモデル定義上デフォルト値が入るため、常に truthy になり得る。
        # 分割/テンプレ/画像分割などの前処理が必要なときだけ request_context 経路を使い、
        # それ以外は通常経路へフォールバックする。
        ctx = chat_request.chat_request_context
        needs_context = False
        if ctx is not None:
            try:
                split_mode = getattr(ctx, "split_mode", ChatRequestContext.split_mode_name_none)
                max_images = int(getattr(ctx, "max_images_per_request", 0) or 0)
                prompt_template = str(getattr(ctx, "prompt_template_text", "") or "")

                needs_context = (
                    split_mode != ChatRequestContext.split_mode_name_none
                    or max_images > 0
                    or bool(prompt_template.strip())
                )
            except Exception:
                # If ctx is malformed, be conservative and use the context path.
                needs_context = True

        if needs_context and ctx is not None:
            return await self.__chat_with_request_context__(
                chat_request, ctx, **kwargs
            )
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
        text_content = client.get_message_factory().create_text_content(summmarize_request_text)
        message = ChatMessage(
            role=client.get_message_factory().get_user_role_name(),
            content=[text_content]
        )
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[message], ), chat_request_context=None)
        summarize_response = await client.chat(chat_request)
        return summarize_response