from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.llm_client import  LLMClient
from ai_chat_util.llm.clients.openai import OpenAIClient, AzureOpenAIClient
from ai_chat_util.llm.model import ChatHistory, ChatRequestContext, ChatRequest
from ai_chat_util.llm.clients.anthropic import AnthropicClient

class LLMFactory:
    @classmethod
    def create_llm_client(
        cls, llm_config: LLMConfig = LLMConfig(), 
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(
                model=LLMConfig().completion_model, messages=[]), 
            chat_request_context=None)
            
    ) -> LLMClient:
        if llm_config.llm_provider == "azure_openai":
            return AzureOpenAIClient(llm_config, chat_request)
        elif llm_config.llm_provider == "openai":
            return OpenAIClient(llm_config, chat_request)
        elif llm_config.llm_provider == "anthropic":
            return AnthropicClient(llm_config, chat_request)
        
        raise ValueError(f"Unsupported LLM provider: {llm_config.llm_provider}")
        

