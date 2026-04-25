from typing import Annotated, Literal

from pydantic import Field

from ai_chat_util.ai_chat_util_base.chat import create_llm_client
from ai_chat_util.common.config.runtime import get_runtime_config, get_runtime_config_info
from ai_chat_util.ai_chat_util_base.ai_chatl_util_models import ChatContent, ChatMessage
from ai_chat_util.ai_chat_util_base.file_util.model import FileUtilDocument


def use_custom_pdf_analyzer() -> Annotated[bool, Field(description="Whether to use the custom PDF analyzer or not")]:
    """
    This function checks whether to use the custom PDF analyzer based on ai-chat-util-config.yml.
    """
    cfg = get_runtime_config()
    return cfg.features.use_custom_pdf_analyzer


def get_completion_model() -> Annotated[str, Field(description="The completion model used for LLM")]:
    """
    This function creates a ChatHistory object from a list of chat messages.
    """
    cfg = get_runtime_config()
    return cfg.llm.completion_model


def get_loaded_config_info() -> Annotated[dict, Field(description="Resolved config file path and raw config content")]:
    """
    Return the actual config file path and the raw YAML content loaded from it.
    """
    return get_runtime_config_info()


def create_user_message(
        chat_content_list: Annotated[list[ChatContent], Field(description="List of chat contents from the user messages")]
) -> Annotated[ChatMessage, Field(description="Chat history created from user messages")]:
    """
    This function creates a ChatHistory object from a list of user messages.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_user_message(chat_content_list)


def create_assistant_message(
        chat_content_list: Annotated[list[ChatContent], Field(description="List of chat contents from the assistant messages")]
) -> Annotated[ChatMessage, Field(description="Chat history created from assistant messages")]:
    """
    This function creates a ChatHistory object from a list of assistant messages.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_assistant_message(chat_content_list)


def create_system_message(
        chat_content_list: Annotated[list[ChatContent], Field(description="List of chat contents from the system messages")]
) -> Annotated[ChatMessage, Field(description="Chat history created from system messages")]:
    """
    This function creates a ChatHistory object from a list of system messages.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_system_message(chat_content_list)


def create_text_content(
        text: Annotated[str, Field(description="Text content for the chat message")]
) -> Annotated[ChatContent, Field(description="Chat content created from text")]:
    """
    This function creates a ChatContent object from text.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_text_content(text)


def create_image_content(
        image_bytes: Annotated[bytes, Field(description="Image bytes for the chat message content")],
        detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from image bytes")]:
    """
    This function creates a ChatContent object from image bytes.
    """
    llm_client = create_llm_client()
    identifier = "画像データのコンテンツ"
    document_type = FileUtilDocument(data=image_bytes, identifier=identifier)
    return llm_client.get_message_factory().create_image_content(document_type, detail)


def create_image_content_from_file(
        file_path: Annotated[str, Field(description="File path for the chat message content")],
        detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from image file")]:
    """
    This function creates a ChatContent object from an image file.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_image_content_from_file(file_path, detail)


def create_pdf_content(
        document_type: Annotated[FileUtilDocument, Field(description="PDF file data for the chat message content")],
        detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for PDF analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[list[ChatContent], Field(description="Chat content created from PDF file data")]:
    """
    This function creates a ChatContent object from PDF file data.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_pdf_content(document_type, detail)


def create_pdf_content_from_file(
    file_path: Annotated[str, Field(description="File path for the chat message content")],
    detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for PDF analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from file")]:
    """
    This function creates a ChatContent object from a file.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_pdf_content_from_file(file_path, detail)


def create_office_content(
        document_type: Annotated[FileUtilDocument, Field(description="Office document file data for the chat message content")],
        detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for Office document analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from Office document file data")]:
    """
    This function creates a ChatContent object from Office document file data.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_office_content(document_type, detail)


def create_office_content_from_file(
        file_path: Annotated[str, Field(description="File path for the chat message content")],
        detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for Office document analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from Office document file")]:
    """
    This function creates a ChatContent object from an Office document file.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_office_content_from_file(file_path, detail)


def create_multi_format_contents_from_file(
        file_path: Annotated[str, Field(description="File path for the chat message content")],
        detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for file analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from multi-format file")]:
    """
    This function creates a ChatContent object from a multi-format file.
    """
    llm_client = create_llm_client()
    return llm_client.get_message_factory().create_multi_format_contents_from_file(file_path, detail)