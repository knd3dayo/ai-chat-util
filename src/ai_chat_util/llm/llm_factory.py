from ai_chat_util.config.runtime import get_runtime_config, AiChatUtilConfig
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.model.models import ChatHistory, ChatRequestContext, ChatRequest

class LLMFactory:
    @classmethod
    def create_llm_client(
        cls, llm_config: AiChatUtilConfig | None = None, 
        use_mcp: bool = False
    ) -> LLMClient:
        if llm_config is None:
            llm_config = get_runtime_config()
        return LLMClient(llm_config, use_mcp=use_mcp)
        




