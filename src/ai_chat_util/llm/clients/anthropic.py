import base64
from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.model import ChatHistory, ChatResponse, ChatRequestContext, ChatMessage, ChatContent, RequestModel

from anthropic import AsyncAnthropic

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class AnthropicClient(LLMClient):
    def __init__(self, llm_config: LLMConfig, chat_history: ChatHistory = ChatHistory(), request_context: ChatRequestContext = ChatRequestContext()):

        self.client = AsyncAnthropic(api_key=llm_config.api_key)
        self.model = llm_config.completion_model
        self.chat_history = chat_history
        self.request_context = request_context
    
    def create(
        self, llm_config: LLMConfig = LLMConfig(), 
        chat_history: ChatHistory = ChatHistory(), 
        request_context: ChatRequestContext = ChatRequestContext()
    ) -> "LLMClient":
        return AnthropicClient(llm_config, chat_history, request_context)  
    
    
    def get_user_role_name(self) -> str:
        return "user"

    def get_assistant_role_name(self) -> str:
        return "assistant"

    def get_system_role_name(self) -> str:
        return "system"

    async def _chat_completion_(self,  **kwargs) -> ChatResponse:

        message_dict_list = [msg.model_dump() for msg in self.chat_history.messages]

        # Invoke the model with the request.
        logger.debug(f"Anthropic Chat Completion Request: {message_dict_list}, Model: {self.model}, Kwargs: {kwargs}")
        response = await self.client.messages.create(
            max_tokens=4196,
            model=self.model,
            messages=message_dict_list,
            **kwargs
        )

        return ChatResponse(
            output=response.content[0].text or "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
    
    def _is_text_content_(self, content: ChatContent) -> bool:
        return content.params.get("type") == "text"

    def _is_image_content_(self, content: ChatContent) -> bool:
        return content.params.get("type") == "image"
    
    def _is_file_content_(self, content: ChatContent) -> bool:
        return content.params.get("type") == "file"
    
    def _create_image_content_(self, image_data: bytes, detail: str) -> "ChatContent":
        base64_image = base64.b64encode(image_data).decode('utf-8')
        media_type = "image/png"
        params = {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_image}}
        return ChatContent(params=params)
    
    def _create_pdf_content_(self, file_data: bytes, filename: str) -> ChatContent:
        base64_file = base64.b64encode(file_data).decode('utf-8')
        media_type = "application/pdf" 
        params = {"type": "document", "source": {"type": "base64", "media_type": media_type, "data": base64_file}}
        return ChatContent(params=params)
