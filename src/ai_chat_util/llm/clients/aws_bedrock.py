import base64
import json

from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.model import (
    ChatHistory, ChatResponse, ChatContent, ChatRequest
)
from file_util.model import DocumentType

import boto3

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class AWSBedrockClient(LLMClient):
    def __init__(self, llm_config: LLMConfig, chat_request: ChatRequest | None = None):
        if chat_request is None:
            chat_request = ChatRequest(
                chat_history=ChatHistory(
                    model=llm_config.completion_model, messages=[]),
                chat_request_context=None
            )
        params = {"service_name": 'bedrock-runtime', "region_name": llm_config.region_name}

        # アクセスキーとシークレットキーが設定されている場合のパラメーター
        if llm_config.api_key and llm_config.api_secret:
            params['aws_access_key_id'] = llm_config.api_key
            params['aws_secret_access_key'] = llm_config.api_secret

        self.client = boto3.client(
            **params
        )

        self.model = llm_config.completion_model
        self.chat_request = chat_request

    def create(
        self, llm_config: LLMConfig = LLMConfig(), 
        chat_request: ChatRequest | None = None
    ) -> "LLMClient":
        return AWSBedrockClient(llm_config, chat_request)

    def get_user_role_name(self) -> str:
        return "user"

    def get_assistant_role_name(self) -> str:
        return "assistant"

    def get_system_role_name(self) -> str:
        return "system"

    async def _chat_completion_(self,  **kwargs) -> ChatResponse:
        messages = self.chat_request.chat_history.messages
        message_dict_list = [msg.model_dump() for msg in messages]


        native_request = {
            "messages": message_dict_list
        }

        # Convert the native request to JSON.
        request = json.dumps(native_request)
        # Invoke the model with the request.
        response = self.client.invoke_model(modelId=self.model, body=request)
        # Decode the response body.
        model_response = json.loads(response["body"].read())

        # Extract and print the response text.
        response_text = model_response["content"][0]["text"]
        input_tokens = model_response.get("usage", {}).get("inputTokens", 0)
        output_tokens = model_response.get("usage", {}).get("outputTokens", 0)
    
        return ChatResponse(
            output=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens
            )

    def _is_text_content_(self, content: ChatContent) -> bool:
        return content.params.get("type") == "text"

    def _is_image_content_(self, content: ChatContent) -> bool:
        return content.params.get("type") == "image_url"
    
    def _is_file_content_(self, content: ChatContent) -> bool:
        return content.params.get("type") == "file"
    
    def _create_image_content_(self, file_data: DocumentType, detail: str) -> list[ChatContent]:
        base64_image = base64.b64encode(file_data.data).decode('utf-8')
        image_url = f"data:image/png;base64,{base64_image}"
        params = {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
        return [ChatContent(params=params)]
    
    def _create_pdf_content_(self, file_data: DocumentType, file_name: str) -> list[ChatContent]:
        base64_file = base64.b64encode(file_data.data).decode('utf-8')
        file_url = f"data:application/pdf;base64,{base64_file}"    
        params = {"type": "file", "file": {"file_data": file_url, "filename": file_name}}
        return [ChatContent(params=params)]