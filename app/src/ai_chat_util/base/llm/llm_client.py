from __future__ import annotations

from typing import Optional, Any, cast
import asyncio
from abc import ABC, abstractmethod
from .abstract_llm_client import AbstractLLMClient

from ai_chat_util_base.config.runtime import get_runtime_config, AiChatUtilConfig
from ai_chat_util_base.model.ai_chatl_util_models import (
    ChatHistory, ChatResponse, ChatRequestContext, ChatMessage, 
    ChatContent, WebRequestModel, ChatRequest
)
from .llm_message_content_factory import LLMMessageContentFactoryBase, LLMMessageContentFactory

from .llm_mcp_client import MCPClient

import litellm

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LLMClientBase(AbstractLLMClient, ABC):

    @abstractmethod
    def get_config(self) -> AiChatUtilConfig | None:
        pass

    @abstractmethod
    def get_message_factory(self) -> LLMMessageContentFactoryBase:
        pass

    @abstractmethod
    def create(
        self, llm_config: AiChatUtilConfig | None = None
        ) -> "LLMClientBase":
        pass


    async def simple_chat(self, prompt: str,) -> str:
        '''
        簡易的なChatCompletionを実行する.
        引数として渡されたプロンプトをChatMessageに変換し、LLMに対してChatCompletionを実行する.
        その後、文字列を返す.
        Args:
            prompt (str): プロンプト文字列
        Returns:
            CompletionResponse: LLMからの応答
        '''
        chat_message = ChatMessage(
            role=self.get_message_factory().get_user_role_name(),
            content=[self.get_message_factory().create_text_content(prompt)]
        )
        chat_history = ChatHistory(messages=[chat_message])
        
        response = await self.chat(ChatRequest(chat_history=chat_history, chat_request_context=None))
        return response.output

    async def chat(
            self, chat_request: ChatRequest, **kwargs
            ) -> ChatResponse:
        '''
        LLMに対してChatCompletionを実行する.
        引数として渡されたChatMessageの前処理を実施した上で、LLMに対してChatCompletionを実行する.
        その後、後処理を実施し、CompletionResponseを返す.
        chat_messageがNoneの場合は、chat_historyから最後のユーザーメッセージを取得して処理を実施する.

        Args:
            chat_request (ChatRequest): チャットリクエスト
        Returns:
            ChatResponse: LLMからの応答
        '''
        if chat_request is None:
            raise ValueError("chat_request must be provided")

        # NOTE:
        # ChatRequest.chat_request_context はモデル定義上デフォルト値が入るため、常に truthy になり得る。
        # 分割/テンプレ/画像分割などの前処理が必要なときだけ request_context 経路を使い、
        # それ以外は通常経路へフォールバックする。
        ctx = chat_request.chat_request_context
        needs_context = False
        if ctx is not None:
            try:
                split_mode = getattr(ctx, "split_mode", ChatRequestContext.split_mode_name_none)
                max_images = int(getattr(ctx, "max_images_per_request", 0) or 0)
                prompt_template = str(getattr(ctx, "prompt_template_text", "") or "")

                needs_context = (
                    split_mode != ChatRequestContext.split_mode_name_none
                    or max_images > 0
                    or bool(prompt_template.strip())
                )
            except Exception:
                # If ctx is malformed, be conservative and use the context path.
                needs_context = True

        if needs_context and ctx is not None:
            return await self.__chat_with_request_context__(
                chat_request, ctx, **kwargs
            )
        return await self.__normal_chat__(
            chat_request, **kwargs
        )   

    @abstractmethod
    async def __chat_with_request_context__(
        self, chat_request: ChatRequest, chat_request_context: ChatRequestContext, **kwargs
    ) -> ChatResponse:
        pass

    @abstractmethod
    async def __normal_chat__(
        self, chat_request: ChatRequest, **kwargs
    ) -> ChatResponse:
        pass


    async def __postprocess_messages__(
        self,
        chat_responses: list[ChatResponse],
        request_context: ChatRequestContext
    ) -> ChatResponse:
        '''
        request_contextの内容に従い、メッセージの後処理を実施する
        * split_modeがsplit_and_summarizeの場合、
            ChatMessageのcontentのうち、typeがtextの要素を抽出し、
            summarize_prompt_textを用いて要約を実施する
            summarize_prompt_textが空文字列の場合は例外をスローする
        Args:
            chat_responses (list[CompletionResponse]): 後処理対象のCompletionResponseリスト
            request_context (ChatRequestContext): 後処理の設定情報
        Returns:
            ChatMessage: 後処理後のChatMessage
        '''
        if request_context.split_mode != ChatRequestContext.split_mode_name_split_and_summarize:
            # chat_responsesのサイズが1の場合はそのまま返す
            if len(chat_responses) == 1:
                return chat_responses[0]
            
            # split_modeがsplit_and_summarize以外の場合は、各テキストの冒頭に[answer_part_i]を付与して結合する
            result_text = ""
            for i, chat_response in enumerate(chat_responses):
                result_text += f"[answer_part_{i+1}]\n" + chat_response.output + "\n"
            return ChatResponse(
                messages=[
                    ChatMessage(role="assistant", content=[
                        ChatContent(params={"type": "text", "text": result_text.strip()})
                        ]
                        )]
                )
        
        if not request_context.summarize_prompt_text:
            raise ValueError("summarize_prompt_text must be set when split_mode is 'split_and_summarize'")
        # split_modeがsplit_and_summarizeの場合は要約を実施する
        summmarize_request_text = request_context.summarize_prompt_text + "\n"
        for chat_response in chat_responses:
            summmarize_request_text += chat_response.output + "\n"

        # request_contextはsplit_modeをnoneに設定して要約を実施する
        request_context = ChatRequestContext(
            split_mode=ChatRequestContext.split_mode_name_none,
            summarize_prompt_text=request_context.summarize_prompt_text
        )
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(
                messages=[]
                ), chat_request_context=request_context)
        client = self.create(self.get_config())
        text_content = client.get_message_factory().create_text_content(summmarize_request_text)
        message = ChatMessage(
            role=client.get_message_factory().get_user_role_name(),
            content=[text_content]
        )
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[message], ), chat_request_context=None)
        summarize_response = await client.chat(chat_request)
        return summarize_response


class LLMClient(LLMClientBase):

    llm_config: AiChatUtilConfig
    concurrency_limit: int = 16

    default_timeout_seconds: float = 300.0
    
    def __init__(self, llm_config: AiChatUtilConfig, use_mcp: bool = False):

        self.llm_config = llm_config
        self.use_mcp = use_mcp
        self.message_factory = LLMMessageContentFactory()

        # ai-chat-util-config.yml の non-secret 設定を既定値として採用
        try:
            self.default_timeout_seconds = float(getattr(self.llm_config.llm, "timeout_seconds", 60.0) or 60.0)
        except (TypeError, ValueError):
            raise ValueError(
                f"llm.timeout_seconds は数値である必要があります: {getattr(self.llm_config.llm, 'timeout_seconds', None)!r}"
            )

    def get_config(self) -> AiChatUtilConfig:
        return self.llm_config

    def get_message_factory(self) -> LLMMessageContentFactoryBase:
        return self.message_factory

    def create(
        self, llm_config: AiChatUtilConfig | None = None, use_mcp: bool = False
        ) -> "LLMClient":
        if llm_config is None:
            llm_config = get_runtime_config()
        return LLMClient(llm_config, use_mcp=use_mcp)

    async def _chat_completion_(self, chat_request: ChatRequest, **kwargs) -> ChatResponse:
        if self.use_mcp:
            raise NotImplementedError("MCP経由のChatCompletionはMCPClientクラスで直接呼び出してください。")
            return await self.run_mcp_chat_completion(
                self.llm_config, 
                chat_request, 
                self.default_timeout_seconds, 
                **kwargs
            )
        else:
            return await self.run_litellm_chat_completion(
                    self.llm_config, 
                    chat_request, 
                    self.default_timeout_seconds, 
                    **kwargs
            )


    async def run_mcp_chat_completion(
        self, llm_config: AiChatUtilConfig, chat_request: ChatRequest, default_timeout_seconds, **kwargs
    ) -> ChatResponse:
        messages = chat_request.chat_history.messages
        mcp_client = MCPClient(llm_config)
        # タイムアウトが未指定だと、ネットワーク待ちで無限に止まることがある
        kwargs.setdefault("timeout", default_timeout_seconds)

        # LiteLLM/OpenAI 側の timeout とは別に、アプリ側でも強制タイムアウトを掛けて
        # 予期せぬ接続待ち等で“体感ハング”しないようにする。
        hard_timeout: float = default_timeout_seconds
        timeout_kw = kwargs.get("timeout")
        if isinstance(timeout_kw, (int, float)) and float(timeout_kw) > 0:
            hard_timeout = float(timeout_kw)

        # async関数内で同期I/Oを呼ぶとイベントループがブロックされるため、acompletionを使う
        # NOTE: messages には画像base64等が入るため、ここでは内容をログ出力しない（巨大化防止）
        logger.debug(
            "LLM completion request: provider=%s model=%s messages=%d timeout=%s",
            llm_config.llm.provider,
            llm_config.llm.completion_model,
            len(messages),
            kwargs.get("timeout"),
        )
        try:
            response = await asyncio.wait_for(
                mcp_client.chat(chat_request),
                timeout=hard_timeout,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                "LLM呼び出しがタイムアウトしました。"
                f" timeout={hard_timeout}s provider={llm_config.llm.provider} model={llm_config.llm.completion_model}.\n"
                "対処: ai-chat-util-config.yml の llm.timeout_seconds を増やすか、CLIの --loglevel/--logfile でログを確認してください。"
            ) from e
        logger.debug("LLM completion response type: %s", type(response))

        if isinstance(response, ChatResponse):
            return response

        raise TypeError(f"Unexpected MCP response type: {type(response)!r}")

    async def run_litellm_chat_completion(
        self, llm_config: AiChatUtilConfig, chat_request: ChatRequest, default_timeout_seconds, **kwargs
    ) -> ChatResponse:
        messages = chat_request.chat_history.messages
        message_dict_list: list[dict[str, Any]] = [msg.model_dump() for msg in messages]
        params = {}
        # api_key の解決/未設定エラーは設定ロード時(runtime)に行う。
        provider = (llm_config.llm.provider or "").lower()
        api_key = llm_config.llm.api_key
        params["api_key"] = api_key
        params["model"] = f"{llm_config.llm.provider}/{llm_config.llm.completion_model}"
        params["messages"] = message_dict_list
        if llm_config.llm.base_url:
            params["base_url"] = llm_config.llm.base_url
        if llm_config.llm.api_version:
            params["api_version"] = llm_config.llm.api_version
        extra_headers = getattr(llm_config.llm, "extra_headers", None)
        if extra_headers:
            filtered = {
                k: v for k, v in extra_headers.items() if not (k or "").lower().startswith("x-mcp-")
            }
            if filtered:
                params["extra_headers"] = filtered

        # タイムアウトが未指定だと、ネットワーク待ちで無限に止まることがある
        kwargs.setdefault("timeout", default_timeout_seconds)

        # LiteLLM/OpenAI 側の timeout とは別に、アプリ側でも強制タイムアウトを掛けて
        # 予期せぬ接続待ち等で“体感ハング”しないようにする。
        hard_timeout: float = default_timeout_seconds
        timeout_kw = kwargs.get("timeout")
        if isinstance(timeout_kw, (int, float)) and float(timeout_kw) > 0:
            hard_timeout = float(timeout_kw)

        # async関数内で同期I/Oを呼ぶとイベントループがブロックされるため、acompletionを使う
        # NOTE: messages には画像base64等が入るため、ここでは内容をログ出力しない（巨大化防止）
        logger.debug(
            "LLM completion request: provider=%s model=%s messages=%d timeout=%s",
            provider,
            params.get("model"),
            len(message_dict_list),
            kwargs.get("timeout"),
        )
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    **params,
                    **kwargs
                ),
                timeout=hard_timeout,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                "LLM呼び出しがタイムアウトしました。"
                f" timeout={hard_timeout}s provider={provider} model={params.get('model')}.\n"
                "対処: ai-chat-util-config.yml の llm.timeout_seconds を増やすか、CLIの --loglevel/--logfile でログを確認してください。"
            ) from e
        logger.debug("LLM completion response type: %s", type(response))

        if isinstance(response, litellm.ModelResponse):
            # NOTE: litellm.ModelResponse は実行時に usage が載りますが、型定義上は
            # 属性として見えないことがあるため dict-style access を使う。
            usage = response.get("usage") or {}
            output_tokens = int(usage.get("completion_tokens", 0) or 0)
            input_tokens = int(usage.get("prompt_tokens", 0) or 0)

            choices = cast(list[Any], response.get("choices") or [])
            output = ""
            if choices:
                first_choice = cast(Any, choices[0])
                # OpenAI互換の {"message": {"content": "..."}} を優先して読む
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                else:
                    message = getattr(first_choice, "message", None)

                if isinstance(message, dict):
                    output = message.get("content") or ""
                else:
                    output = getattr(message, "content", "") or ""

            # choicesが空 or contentが空の場合は、明示的に失敗させて原因をユーザーに見せる。
            # （"何も出力されない" 体験を避ける）
            if not choices:
                err = response.get("error")
                if err:
                    raise RuntimeError(f"LLM応答にエラーが含まれています: {err}")
                response_dict = cast(dict[str, Any], response)
                raise RuntimeError(
                    "LLM応答の choices が空でした。"
                    f" response_keys={list(response_dict.keys())}"
                )
            if not str(output).strip():
                err = response.get("error")
                if err:
                    raise RuntimeError(f"LLM応答にエラーが含まれています: {err}")
                response_dict = cast(dict[str, Any], response)
                raise RuntimeError(
                    "LLM応答の content が空でした。"
                    f" response_keys={list(response_dict.keys())}"
                )

            return ChatResponse(
                messages=[ChatMessage(role="assistant", content=[ChatContent(params={"type": "text", "text": output})])],
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        raise TypeError(f"Unexpected response type: {type(response)!r}")


    async def __normal_chat__(self, chat_request: ChatRequest, **kwargs) -> ChatResponse:
        '''
        LLMに対してChatCompletionを実行する.
        引数として渡されたChatMessageをそのままLLMに対してChatCompletionを実行する.
        その後、CompletionResponseを返す.
        chat_messageがNoneの場合は、chat_historyから最後のユーザーメッセージを取得して処理を実施する.

        Args:
            chat_request (ChatRequest): チャットリクエスト
        Returns:
            ChatResponse: LLMからの応答
        '''
        # IMPORTANT:
        # chat() から渡される chat_message_list は self.chat_request.chat_history.messages と同一オブジェクトのことがある。
        # そのリストを反復しつつ add_message() で同じリストに追記すると、リストが伸び続けて無限ループする。
        # ここでは「既存履歴をそのまま使う」か「渡されたリストのコピーを履歴として採用」し、重複追加はしない。
        chat_response =  await self._chat_completion_(chat_request, **kwargs)
        text_content = self.get_message_factory().create_text_content(chat_response.output)
        chat_request.chat_history.add_message(ChatMessage(
            role=self.get_message_factory().get_assistant_role_name(),
            content=[text_content]
        ))
        return chat_response

    async def __chat_with_request_context__(self, chat_request: ChatRequest, request_context: ChatRequestContext, **kwargs) -> ChatResponse:
        '''
        LLMに対してChatCompletionを実行する.
        引数として渡されたChatRequestの前処理を実施した上で、LLMに対してChatCompletionを実行する.
        その後、後処理を実施し、ChatResponseを返す.
        chat_requestがNoneの場合は、chat_historyから最後のユーザーメッセージを取得して処理を実施する.

        Args:
            chat_request (ChatRequest): チャットリクエスト
        Returns:
            ChatResponse: LLMからの応答
        '''
        if not chat_request:
            raise ValueError("chat_request must be provided")

        # 前処理を実行
        last_user_messages, previous_messages = self.get_message_factory().__get_last_user_messages__(chat_request.chat_history)
        preprocessed_messages = self.get_message_factory().__preprocess_text_message__(last_user_messages, request_context)
        preprocessed_messages = self.get_message_factory().__preprocess_image_urls__(preprocessed_messages, request_context)

        # LLMに対してChatCompletionを実行. messageごとにasyncioのタスクを作成して実行する
        async def __process_message__(message_num: int, message: ChatMessage, previous_messages: list[ChatMessage]) -> tuple[int, ChatResponse]:
            # Preserve MCP mode when creating per-task clients.
            client = self.create(self.llm_config, use_mcp=self.use_mcp)
            chat_request: ChatRequest = ChatRequest(chat_history=ChatHistory(messages=previous_messages), chat_request_context=request_context)
            
            chat_request.chat_history.add_message(message)
            chat_response =  await client._chat_completion_(chat_request, **kwargs)
            return (message_num, chat_response)
            
        chat_response_tuples: list[tuple[int, ChatResponse]] = []

        sem = asyncio.Semaphore(self.concurrency_limit)

        async def __run_one__(i: int, message: ChatMessage) -> tuple[int, ChatResponse]:
            async with sem:
                return await __process_message__(i, message, previous_messages)

        tasks = [asyncio.create_task(__run_one__(i, message)) for i, message in enumerate(preprocessed_messages)]
        chat_response_tuples = await asyncio.gather(*tasks)

        # message_numでソートしてCompletionResponseのリストを作成
        chat_response_tuples.sort(key=lambda x: x[0])
        chat_responses = [t[1] for t in chat_response_tuples]

        # 後処理を実行
        postprocessed_response = await self.__postprocess_messages__(chat_responses, request_context)

        # chat_historyにpreprocessed_messageとpostprocessed_responseを追加する
        for preprocessed_message in preprocessed_messages:
            
            chat_request.chat_history.add_message(preprocessed_message)

        text_content = self.get_message_factory().create_text_content(postprocessed_response.output)
        response_message = ChatMessage(
            role=self.get_message_factory().get_assistant_role_name(),
            content=[text_content]
        )
        chat_request.chat_history.add_message(response_message)

        return postprocessed_response

