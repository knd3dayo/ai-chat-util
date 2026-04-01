from ai_chat_util.common.config.runtime import AiChatUtilConfig, get_runtime_config
from ai_chat_util.base.llm.abstract_llm_client import AbstractLLMClient

from .mcp_client import DeepAgentMCPClient, MCPClient


class AgentFactory:
	@classmethod
	def create_mcp_client(
		cls, llm_config: AiChatUtilConfig | None = None,
	) -> AbstractLLMClient:
		if llm_config is None:
			llm_config = get_runtime_config()
		return MCPClient(llm_config)

	@classmethod
	def create_deepagent_client(
		cls, llm_config: AiChatUtilConfig | None = None,
	) -> AbstractLLMClient:
		if llm_config is None:
			llm_config = get_runtime_config()
		return DeepAgentMCPClient(llm_config)
