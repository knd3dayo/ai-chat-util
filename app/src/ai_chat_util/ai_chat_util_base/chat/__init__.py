from ai_chat_util.common.config.runtime import AiChatUtilConfig, get_runtime_config

from .abstract_chat_client import AbstractChatClient
from .chat_client_base import ChatClientBase
from .chat_client import LLMClient
from .llm_message_content_factory import LLMMessageContentFactory, LLMMessageContentFactoryBase


def create_llm_client(
    llm_config: AiChatUtilConfig | None = None,
) -> AbstractChatClient:
    if llm_config is None:
        llm_config = get_runtime_config()
    return LLMClient(llm_config)

__all__ = [
    "AbstractChatClient",
    "ChatClientBase",
    "LLMClient",
    "LLMMessageContentFactory",
    "LLMMessageContentFactoryBase",
    "create_llm_client",
]