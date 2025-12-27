from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.llm_client import  LLMClient
from ai_chat_util.llm.clients.openai import OpenAIClient, AzureOpenAIClient
from ai_chat_util.llm.model import ChatHistory, ChatRequestContext

class LLMFactory:
    @classmethod
    def create_llm_client(
        cls, llm_config: LLMConfig = LLMConfig(), 
        chat_history: ChatHistory = ChatHistory(), 
        request_context: ChatRequestContext = ChatRequestContext()
    ) -> LLMClient:
        if llm_config.llm_provider == "azure_openai":
            return AzureOpenAIClient(llm_config, chat_history, request_context)
        else:
            return OpenAIClient(llm_config, chat_history, request_context)

