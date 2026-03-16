from ai_chat_util_base.config.ai_chat_util_runtime import get_runtime_config, AiChatUtilConfig
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.hitl_client import StdIOHITLClient, HITLClientBase
 
class LLMFactory:
    @classmethod
    def create_llm_client(
        cls, llm_config: AiChatUtilConfig | None = None, 
        use_mcp: bool = False
    ) -> LLMClient:
        if llm_config is None:
            llm_config = get_runtime_config()
        return LLMClient(llm_config, use_mcp=use_mcp)
        
    @classmethod
    def create_stdio_hitl_client(
        cls, llm_client: LLMClient, runtime_config: AiChatUtilConfig | None = None, trace_id: str | None = None
    ) -> HITLClientBase:
        if runtime_config is None:
            runtime_config = get_runtime_config()
        return StdIOHITLClient(llm_client, runtime_config, trace_id=trace_id)



