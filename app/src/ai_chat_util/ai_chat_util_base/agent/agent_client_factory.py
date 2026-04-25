from ai_chat_util.common.config.runtime import AiChatUtilConfig, get_runtime_config
from ai_chat_util.ai_chat_util_base.ai_chatl_util_models import ChatRequestContext
from ai_chat_util.ai_chat_util_base.chat import AbstractChatClient

from .agent_client import CodingAgentMCPClient, DeepAgentMCPClient, AgentClient


class AgentFactory:
	@classmethod
	def create_mcp_client(
		cls, llm_config: AiChatUtilConfig | None = None, default_request_context: ChatRequestContext | None = None,
	) -> AbstractChatClient:
		if llm_config is None:
			llm_config = get_runtime_config()
		return AgentClient(llm_config, default_request_context=default_request_context)

	@classmethod
	def create_deepagent_client(
		cls, llm_config: AiChatUtilConfig | None = None,
	) -> AbstractChatClient:
		if llm_config is None:
			llm_config = get_runtime_config()
		return DeepAgentMCPClient(llm_config)

	@classmethod
	def create_codingagent_client(
		cls, llm_config: AiChatUtilConfig | None = None,
	) -> AbstractChatClient:
		if llm_config is None:
			llm_config = get_runtime_config()
		return CodingAgentMCPClient(llm_config)
