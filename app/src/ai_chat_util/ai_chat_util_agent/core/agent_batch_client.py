from ai_chat_util.ai_chat_util_base.core.chat.core import AbstractChatClient
from ai_chat_util.ai_chat_util_base.batch import BatchClientBase
from ai_chat_util.ai_chat_util_base.core.common.config.runtime import AiChatUtilConfig

from .agent_client_factory import AgentFactory


class MCPBatchClient(BatchClientBase):
    def _create_client(self, llm_config: AiChatUtilConfig | None = None) -> AbstractChatClient:
        return AgentFactory.create_mcp_client(llm_config)


class DeepAgentBatchClient(BatchClientBase):
    def _create_client(self, llm_config: AiChatUtilConfig | None = None) -> AbstractChatClient:
        return AgentFactory.create_deepagent_client(llm_config)