from abc import ABC, abstractmethod

from ai_chat_util_base.model.ai_chatl_util_models import (
    ChatResponse, ChatRequest
)
from .llm_client import LLMMessageContentFactoryBase
from ai_chat_util_base.config.runtime import AiChatUtilConfig

class AbstractLLMClient(ABC):

    @abstractmethod
    async def simple_chat(self, prompt: str,) -> str:
        '''
        簡易的なChatCompletionを実行する.
        引数として渡されたプロンプトをChatMessageに変換し、LLMに対してChatCompletionを実行する.
        その後、文字列を返す.
        Args:
            prompt (str): プロンプト文字列
        Returns:
            str: LLMからの応答文字列
        ''' 
        pass

    @abstractmethod
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

    @abstractmethod
    def get_message_factory(self) -> LLMMessageContentFactoryBase:
        '''
        LLMClientが使用するChatMessageFactoryを返す.
        '''
        pass

    @abstractmethod
    def get_config(self) -> AiChatUtilConfig | None:
        '''
        LLMClientの設定を返す.
        '''
        pass
