import base64

from openai import AsyncOpenAI, AsyncAzureOpenAI

from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.model import ChatHistory, ChatResponse, ChatRequestContext, ChatMessage, ChatContent, RequestModel, ChatRequest

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self, llm_config: LLMConfig, chat_request: ChatRequest = ChatRequest()):
        if llm_config.base_url:
            self.client = AsyncOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        else:
            self.client = AsyncOpenAI(api_key=llm_config.api_key)

        self.model = llm_config.completion_model

        self.chat_request = chat_request

    def create(
        self, llm_config: LLMConfig = LLMConfig(), 
        chat_request: ChatRequest = ChatRequest()
    ) -> "LLMClient":
        return OpenAIClient(llm_config, chat_request)

    def get_user_role_name(self) -> str:
        return "user"

    def get_assistant_role_name(self) -> str:
        return "assistant"

    def get_system_role_name(self) -> str:
        return "system"

    async def _chat_completion_(self,  **kwargs) -> ChatResponse:
        messages = self.chat_request.chat_history.messages
        message_dict_list = [msg.model_dump() for msg in messages]
        response = await self.client.chat.completions.create(
            model=self.chat_request.chat_history.model,
            messages=message_dict_list,
            **kwargs
        )
        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)
        return ChatResponse(
            output=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens
            )

    def _is_text_content_(self, content: ChatContent) -> bool:
        return content.params.get("type") == "text"

    def _is_image_content_(self, content: ChatContent) -> bool:
        return content.params.get("type") == "image_url"
    
    def _is_file_content_(self, content: ChatContent) -> bool:
        return content.params.get("type") == "file"
    
    def _create_image_content_(self, image_data: bytes, detail: str) -> "ChatContent":
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/png;base64,{base64_image}"
        params = {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
        return ChatContent(params=params)
    
    def _create_pdf_content_(self, file_data: bytes, filename: str) -> ChatContent:
        base64_file = base64.b64encode(file_data).decode('utf-8')
        file_url = f"data:application/pdf;base64,{base64_file}"    
        params = {"type": "file", "file": {"file_data": file_url, "filename": filename}}
        return ChatContent(params=params)


class AzureOpenAIClient(OpenAIClient):
    def __init__(self, llm_config: LLMConfig, chat_reqquest: ChatRequest = ChatRequest()):
        if llm_config.base_url:
            self.client = AsyncAzureOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        elif llm_config.api_version and llm_config.endpoint:
            self.client = AsyncAzureOpenAI(api_key=llm_config.api_key, azure_endpoint=llm_config.endpoint, api_version=llm_config.api_version)
        else:
            raise ValueError("Either base_url or both api_version and endpoint must be provided.")

        self.model = llm_config.completion_model


    def create(
        self, llm_config: LLMConfig = LLMConfig(), 
        chat_request: ChatRequest = ChatRequest()
    ) -> "LLMClient":
        return AzureOpenAIClient(llm_config, chat_request)

    async def _chat_completion_(self, **kwargs) -> ChatResponse:
        
        message_dict_list = [msg.model_dump() for msg in self.chat_request.chat_history.messages]
        response = await self.client.chat.completions.create(
            model=self.chat_request.chat_history.model,
            messages=message_dict_list,
            **kwargs
        )
        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)
        return ChatResponse(
            output=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens
            )
