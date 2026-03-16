from __future__ import annotations

from typing import Callable
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.config.runtime import AiChatUtilConfig
from ai_chat_util.model.models import ChatRequest, ChatHistory, ChatMessage, ChatContent

class HITLClient:
    def __init__(self, llm_client: LLMClient, runtime_config: AiChatUtilConfig, trace_id: str | None = None):
        self.llm_client = llm_client
        self.trace_id = trace_id
        self.runtime_config = runtime_config

    def _mk_cli_user_request(self, text: str) -> ChatRequest:
        msg = ChatMessage(
            role="user",
            content=[ChatContent(params={"type": "text", "text": text})],
        )
        return ChatRequest(
            trace_id=self.trace_id,
            chat_history=ChatHistory(messages=[msg]),
            chat_request_context=None,
        )

    async def run_cli(self, initial_prompt: str) -> None:
        await self.run(initial_prompt, self._mk_cli_user_request)
    
    async def run(self, initial_prompt: str, mk_request_func: Callable[[str], ChatRequest]) -> None:
        chat_response = await self.llm_client.chat(mk_request_func(initial_prompt))
        print(chat_response.output)

        # Only MCP workflow returns paused today; still safe to handle generically.
        while getattr(chat_response, "status", "completed") == "paused":
            trace_id = getattr(chat_response, "trace_id", None) or self.trace_id
            prompt = None
            hitl = getattr(chat_response, "hitl", None)
            hitl_kind = getattr(hitl, "kind", None) if hitl is not None else None
            hitl_source = getattr(hitl, "source", None) if hitl is not None else None
            if hitl is not None:
                prompt = getattr(hitl, "prompt", None)
            if not prompt:
                prompt = chat_response.output

            if hitl_kind == "approval":
                tool_hint = ""
                if isinstance(hitl_source, str) and ":" in hitl_source:
                    tool_hint = f" ({hitl_source})"
                print(f"\n[HITL:APPROVAL]{tool_hint} 承認が必要です。")
                print("入力例: 'APPROVE TOOL_NAME' / 'REJECT TOOL_NAME'（TOOL_NAME は質問文中のもの）")
            else:
                print("\n[HITL] 次の質問に回答してください:")
            print(prompt)
            import sys

            sys.stdout.flush()
            try:
                answer = input("HITL> ").strip()
            except EOFError:
                print("\n標準入力が閉じているため、HITL の入力待ちができません。")
                raise SystemExit(1)
            if not answer:
                print("入力が空です。再度入力してください。")
                continue

            chat_response = await self.llm_client.chat(mk_request_func(answer))
            print(chat_response.output)

        return

