from ai_chat_util.core.chat import AbstractChatClient
from ai_chat_util.core.chat.batch_client_base import BatchClientBase
from ai_chat_util.core.common.config.runtime import AiChatUtilConfig

from .agent_client_factory import AgentFactory


class MCPBatchClient(BatchClientBase):
    def _create_client(self, llm_config: AiChatUtilConfig | None = None) -> AbstractChatClient:
        return AgentFactory.create_mcp_client(llm_config)


class DeepAgentBatchClient(BatchClientBase):
    def _create_client(self, llm_config: AiChatUtilConfig | None = None) -> AbstractChatClient:
        return AgentFactory.create_deepagent_client(llm_config)