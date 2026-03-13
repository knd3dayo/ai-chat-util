from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.llm_client import  LLMClient, LiteLLMClient
from ai_chat_util.llm.model import ChatHistory, ChatRequestContext, ChatRequest

class LLMFactory:
    @classmethod
    def create_llm_client(
        cls, llm_config: LLMConfig | None = None, 
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[]), 
            chat_request_context=None)
            
    ) -> LLMClient:
        if llm_config is None:
            llm_config = LLMConfig()
        return LiteLLMClient(llm_config, chat_request)
        

