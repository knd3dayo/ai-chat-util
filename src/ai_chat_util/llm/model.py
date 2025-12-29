import base64, os
# 抽象クラス
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal
from pydantic import BaseModel, Field
from typing import Optional, Any
from typing import ClassVar, Optional, Any
from pydantic import BaseModel, Field

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class RequestModel(BaseModel):
    url: str
    headers: dict[str, Any] = {}

    

class ChatContent(BaseModel):
    params: dict[str, Any] = Field(default={}, description="Parameters of the chat content.")
    def model_dump(self, *args, **kwargs):
            base = super().model_dump(*args, **kwargs)
            # paramsを展開
            return {**{k: v for k, v in base.items() if k != "params"}, **self.params}

class ChatMessage(BaseModel):
    role: str = Field(default="user", description="The role of the message sender (e.g., 'user', 'assistant').")
    content: list[ChatContent] = Field(default=[], description="The content of the message, which can be text or other types.")

    # model_dump をオーバーライドして content を展開する
    def model_dump(self, *args, **kwargs):
        base = super().model_dump(*args, **kwargs)
        # baseからcontentを除去
        del base["content"]

        # contentを展開
        return {**{k: v for k, v in base.items() if k != "content"}, **{"content": [c.model_dump() for c in self.content]}}

    def get_last_user_content(self) -> Optional[ChatContent]:
        """
        Get the last ChatContent in the content list.
        
        Returns:
            Optional[ChatContent]: The last chat content or None if no content exists.
        """
        if self.content:
            last_content = self.content[-1]
            logger.debug(f"Last content retrieved: {last_content}")
            return last_content
        else:
            logger.debug("No content found.")
            return None
    
    def add_content(self, chat_content: ChatContent) -> None:
        """
        Add a ChatContent to the content list.
        
        Args:
            chat_content (ChatContent): The chat content to add.
        """
        self.content.append(chat_content)
        logger.debug(f"Content added: {chat_content}")

    def update_last_content(self, chat_content: ChatContent) -> None:
        """
        Update the last ChatContent in the content list.
        
        Args:
            chat_content (ChatContent): The new chat content to replace the last one.
        """
        if self.content:
            self.content[-1] = chat_content
            logger.debug(f"Last content updated to: {chat_content}")
        else:
            logger.debug("No content to update.")


class ChatHistory(BaseModel):

    messages: list[Any] = Field(default=[], description="List of chat messages in the conversation.")
    model: str = Field(default="gpt-4o", description="The model used for the chat conversation.")
    
    # option fields
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature for the model.")
    response_format: Optional[dict] = Field(default=None, description="Format of the response from the model.")

    def add_message(self, message: "ChatMessage") -> None:
        """
        Add a ChatMessage to the messages list.
        
        Args:
            message (ChatMessage): The chat message to add.
        """
        self.messages.append(message)
        logger.debug(f"Message added: {message.role}: {message.content}")

    def get_last_message(self) -> Optional["ChatMessage"]:
        """
        Get the last ChatMessage in the messages list.
        
        Returns:
            Optional[ChatMessage]: The last chat message or None if no messages exist.
        """
        if not self.messages:
            logger.debug("No messages found.")
            return None

        last_message = self.messages[-1]
        logger.debug(f"Last message retrieved: {last_message}")

        return last_message

    def update_last_message(self, message: "ChatMessage") -> None:
        """
        Update the last ChatMessage in the messages list.
        
        Args:
            message (ChatMessage): The new chat message to replace the last one.
        """
        if self.messages:
            self.messages[-1] = message
            logger.debug(f"Last message updated to: {message.role}: {message.content}")
        else:
            logger.debug("No messages to update.")

    def get_last_role_messages(self, role: str) -> list["ChatMessage"]:
        """
        Get all ChatMessages with role 'user' in the messages after the last assistant message.
        
        Returns:
            list[ChatMessage]: List of chat messages with role 'user'.
        """
        last_role_index = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role != role:
                last_role_index = i
                break
        role_messages = [msg for msg in self.messages[last_role_index + 1:] if msg.role == role]
        logger.debug(f"{role.capitalize()} messages retrieved: {len(role_messages)} messages found.")
        return role_messages

class ChatRequestContext(BaseModel):

    # split_mode
    split_mode_name_none: ClassVar[Literal["none"]] = "none"
    split_mode_name_normal: ClassVar[Literal["normal_split"]] = "normal_split"
    split_mode_name_split_and_summarize: ClassVar[Literal["split_and_summarize"]] = "split_and_summarize"

    # メッセージを分割するモード。分割しない場合は"none"、通常分割は"normal_split"、分割して要約は"split_and_summarize"
    split_mode: Literal["none", "normal_split", "split_and_summarize"] = Field(default="none", description="Mode to split messages. 'none' for no split, 'normal_split' for normal split, 'split_and_summarize' for split and summarize.")

    # メッセージ分割する文字数
    split_message_length: int = Field(default=2000, description="Maximum character count for message splitting.")
    # 複数画像URLがある場合に、1つのリクエストに含める最大画像数。分割しない場合は0を設定
    max_images_per_request: int = Field(default=0, description="Maximum number of images to include in a single request. Set to 0 if not splitting.")
    # SplitAndSummarizeモード時の要約用プロンプトテキスト
    summarize_prompt_text: str = Field(default="Summarize the content concisely.", description="Prompt text for summarization in SplitAndSummarize mode.")
    # プロンプトテンプレートテキスト. 分割モードがNone以外の場合に使用. 分割した各メッセージの前に付与する。
    # 分割モードがNone以外の場合は、各パートはこのプロンプトの指示に従うため、必ず設定すること。
    prompt_template_text: str = Field(default="", description="Prompt template text. Used when split mode is not 'None'. This text is prepended to each split message. When split mode is not 'None', this must be set to guide each part according to the prompt's instructions.")

class ChatRequest(BaseModel):
    chat_history: ChatHistory = Field(default=ChatHistory(), description="The chat history for the request.")
    chat_request_context: Optional[ChatRequestContext] = Field(default=ChatRequestContext(), description="The context for the chat request.")

class ChatResponse(BaseModel):
    output: str = Field(default="", description="The output text from the chat model.")
    input_tokens: int = Field(default=0, description="The number of tokens in the input to the model.")
    output_tokens: int = Field(default=0, description="The number of tokens in the model's output.")

    documents: Optional[list[dict]] = Field(default=None, description="List of documents retrieved during the chat interaction.")
