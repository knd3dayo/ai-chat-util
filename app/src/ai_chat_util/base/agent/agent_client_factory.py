from ai_chat_util.common.config.runtime import AiChatUtilConfig, get_runtime_config
from ai_chat_util.base.chat import AbstractChatClient

from .agent_client import DeepAgentMCPClient, AgentClient


class AgentFactory:
	@classmethod
	def create_mcp_client(
		cls, llm_config: AiChatUtilConfig | None = None,
	) -> AbstractChatClient:
		if llm_config is None:
			llm_config = get_runtime_config()
		return AgentClient(llm_config)

	@classmethod
	def create_deepagent_client(
		cls, llm_config: AiChatUtilConfig | None = None,
	) -> AbstractChatClient:
		if llm_config is None:
			llm_config = get_runtime_config()
		return DeepAgentMCPClient(llm_config)
