from ai_chat_util.ai_chat_util_base.chat.core import AbstractChatClient, create_llm_client
from ai_chat_util.ai_chat_util_base.common.config.runtime import AiChatUtilConfig
from .batch_client_base import BatchClientBase


class BatchClient(BatchClientBase):
    def _create_client(self, llm_config: AiChatUtilConfig | None = None) -> AbstractChatClient:
        return create_llm_client(llm_config)


