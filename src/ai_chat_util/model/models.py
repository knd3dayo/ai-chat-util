# 抽象クラス
import re
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field, model_validator
import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)
_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-(?P<span_id>[0-9a-f]{16})-(?P<trace_flags>[0-9a-f]{2})$",
    re.IGNORECASE,
)


def _normalize_trace_id(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return v

    if _TRACE_ID_RE.fullmatch(v):
        trace_id = v.lower()
    else:
        m = _TRACEPARENT_RE.fullmatch(v)
        if not m:
            raise ValueError(
                "trace_id は W3C traceparent の trace_id 部分（32桁の16進数）を指定してください。"
                "例: '4bf92f3577b34da6a3ce929d0e0e4736'"
            )
        trace_id = m.group("trace_id").lower()

    # W3C trace-id must not be all zeros.
    if trace_id == "0" * 32:
        raise ValueError("trace_id が不正です（全て0は許可されません）。")
    return trace_id

class WebRequestModel(BaseModel):
    url: str
    headers: dict[str, Any] = {}

class ChatContent(BaseModel):
    params: dict[str, Any] = Field(..., description="Parameters of the chat content.")
    def model_dump(self, *args, **kwargs):
            base = super().model_dump(*args, **kwargs)
            # paramsを展開
            return {**{k: v for k, v in base.items() if k != "params"}, **self.params}

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'assistant').")
    content: list[ChatContent] = Field(..., description="The content of the message, which can be text or other types.")

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

    messages: list[ChatMessage] = Field(..., description="List of chat messages in the conversation.")
    
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


class HitlRequest(BaseModel):
    """Human-in-the-loop request.

    - kind="input": 人間が質問に回答する必要がある
    - kind="approval": 人間が承認/却下する必要がある（将来拡張）
    """

    kind: Literal["input", "approval"] = Field(default="input")
    prompt: str = Field(..., description="Human-facing prompt/question/approval summary.")
    action_id: str = Field(..., description="Identifier for this HITL action.")
    source: Optional[str] = Field(default=None, description="Which component requested HITL (e.g., supervisor/tool/mcp).")

class ChatRequest(BaseModel):
    thread_id: Optional[str] = Field(
        default=None,
        description="LangGraph thread_id for checkpointing/resume. If omitted, a new thread is started.",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description=(
            "BFF等が発行する相関ID（W3C traceparentの trace_id 部分: 32桁hex）。"
            "thread_id が未指定の場合、この値を thread_id として使用できます。"
        ),
    )
    auto_approve: bool = Field(
        default=False,
        description=(
            "内部MCPクライアント実行時に、なるべくHITL pause（question）を発生させず自己完結するモード。"
            "true の場合、Supervisor/配下エージェントは追加確認を求めず、合理的な仮定のもとで完了回答を返すよう試みます。"
        ),
    )
    auto_approve_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description=(
            "auto_approve=true のとき、question が返った場合に Supervisor へ追加指示して再実行する最大回数。"
            "0 の場合はリトライせず、そのまま pause せず回答を返します。"
        ),
    )
    chat_history: ChatHistory = Field(..., description="The chat history for the request.")
    chat_request_context: Optional[ChatRequestContext] = Field(default=ChatRequestContext(), description="The context for the chat request.")

    @model_validator(mode="after")
    def _validate_trace_thread_consistency(self) -> "ChatRequest":
        # Normalize trace_id to the "trace-id part only" (32-hex). If traceparent is passed by mistake,
        # extract trace_id to keep compatibility.
        if self.trace_id is not None:
            stripped = self.trace_id.strip()
            if not stripped:
                self.trace_id = None
            else:
                self.trace_id = _normalize_trace_id(stripped)

        # If thread_id looks like a traceparent, normalize it too so `thread_id == trace_id` can hold.
        if self.thread_id and _TRACEPARENT_RE.fullmatch(self.thread_id.strip()):
            self.thread_id = _normalize_trace_id(self.thread_id)

        if self.thread_id and self.trace_id and self.thread_id != self.trace_id:
            raise ValueError(
                "ChatRequest.thread_id と ChatRequest.trace_id の両方が設定されていますが一致しません。"
                "trace_id を thread_id として流用する場合は同一値にするか、どちらか一方だけを指定してください。"
            )
        return self

class ChatResponse(BaseModel):
    status: Literal["completed", "paused"] = Field(default="completed", description="Execution status.")
    thread_id: Optional[str] = Field(default=None, description="LangGraph thread_id associated with this response.")
    trace_id: Optional[str] = Field(
        default=None,
        description=(
            "外部相関ID（trace_id）。thread_id を trace_id で代替する運用の場合、通常は thread_id と同一になります。"
        ),
    )
    hitl: Optional[HitlRequest] = Field(default=None, description="Present when status='paused'.")
    messages: list[ChatMessage] = Field(default_factory=list, description="The output messages from the chat model.")
    input_tokens: int = Field(default=0, description="The number of tokens in the input to the model.")
    output_tokens: int = Field(default=0, description="The number of tokens in the model's output.")

    documents: Optional[list[dict]] = Field(default=None, description="List of documents retrieved during the chat interaction.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_output_payload(cls, data: Any) -> Any:
        """Accept legacy payloads like {"output": "..."}.

        This keeps backward compatibility for callers that still send `output`.
        The model output (model_dump) remains messages-based.
        """
        if not isinstance(data, dict):
            return data

        messages = data.get("messages")
        if messages:
            return data

        legacy_output = data.get("output")
        if isinstance(legacy_output, str) and legacy_output.strip():
            copied = dict(data)
            copied["messages"] = [
                {
                    "role": "assistant",
                    "content": [
                        {"params": {"type": "text", "text": legacy_output}},
                    ],
                }
            ]
            return copied

        return data

    # messagesを文字列として結合して返すユーティリティプロパティ
    @property
    def output(self) -> str:
        return "\n".join(
            [
                "".join(
                    content.params.get("text", "") for content in message.content if content.params.get("type") == "text"
                )
                for message in self.messages
            ]
        )