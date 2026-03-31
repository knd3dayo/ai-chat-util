from ai_chat_util_base.config.runtime import get_runtime_config, AiChatUtilConfig
from ai_chat_util.base.llm.llm_client import LLMClient
from .llm_mcp_client import MCPClient
from ai_chat_util.base.llm.hitl_client import StdIOHITLClient, HITLClientBase
from .abstract_llm_client import AbstractLLMClient 

class LLMFactory:
    @classmethod
    def create_llm_client(
        cls, llm_config: AiChatUtilConfig | None = None,
    ) -> AbstractLLMClient:
        if llm_config is None:
            llm_config = get_runtime_config()
        return LLMClient(llm_config)

    @classmethod
    def create_mcp_client(
        cls, llm_config: AiChatUtilConfig | None = None,
    ) -> AbstractLLMClient:
        if llm_config is None:
            llm_config = get_runtime_config()
        return MCPClient(llm_config)
        
    @classmethod
    def create_stdio_hitl_client(
        cls, llm_client: AbstractLLMClient, runtime_config: AiChatUtilConfig | None = None, trace_id: str | None = None
    ) -> HITLClientBase:
        if runtime_config is None:
            runtime_config = get_runtime_config()
        return StdIOHITLClient(llm_client, runtime_config, trace_id=trace_id)



