from __future__ import annotations
from abc import ABC, abstractmethod
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.config.runtime import AiChatUtilConfig
from ai_chat_util.model.models import ChatRequest, ChatHistory, ChatMessage, ChatContent
import sys

class IOManagerBase(ABC):
    @abstractmethod
    def read_input(self, prompt: str) -> str:
        pass

    @abstractmethod
    def write_output(self, output: str) -> None:
        pass

class StdIOManager(IOManagerBase):
    def read_input(self, prompt: str) -> str:
        print(prompt)
        sys.stdout.flush()
        try:
            return input("HITL> ").strip()
        except EOFError:
            print("\n標準入力が閉じているため、HITL の入力待ちができません。")
            raise SystemExit(1)

    def write_output(self, output: str) -> None:
        print(output)

class HITLClientBase(ABC):
    def __init__(self, llm_client: LLMClient, runtime_config: AiChatUtilConfig, trace_id: str | None = None):
        self.llm_client = llm_client
        self.trace_id = trace_id
        self.runtime_config = runtime_config

    @abstractmethod
    def get_io_manager(self) -> IOManagerBase:
        pass

    def _mk_user_request(self, text: str) -> ChatRequest:
        msg = ChatMessage(
            role="user",
            content=[ChatContent(params={"type": "text", "text": text})],
        )
        return ChatRequest(
            trace_id=self.trace_id,
            chat_history=ChatHistory(messages=[msg]),
            chat_request_context=None,
        )

    async def run(self, initial_prompt: str) -> None:
        io_manager = self.get_io_manager()
        chat_response = await self.llm_client.chat(self._mk_user_request(initial_prompt))
        io_manager.write_output(chat_response.output)

        # Only MCP workflow returns paused today; still safe to handle generically.
        while getattr(chat_response, "status", "completed") == "paused":
            self.trace_id = getattr(chat_response, "trace_id", None) 
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
                io_manager.write_output(f"\n[HITL:APPROVAL]{tool_hint} 承認が必要です。")
                io_manager.write_output("入力例: 'APPROVE TOOL_NAME' / 'REJECT TOOL_NAME'（TOOL_NAME は質問文中のもの）")
            else:
                io_manager.write_output("\n[HITL] 次の質問に回答してください:")
            io_manager.write_output(prompt)

            sys.stdout.flush()
            try:
                answer = io_manager.read_input("HITL> ").strip()
            except EOFError:
                io_manager.write_output("\n標準入力が閉じているため、HITL の入力待ちができません。")
                raise SystemExit(1)
            if not answer:
                io_manager.write_output("入力が空です。再度入力してください。")
                continue

            chat_response = await self.llm_client.chat(self._mk_user_request(answer))
            io_manager.write_output(chat_response.output)

        return

class StdIOHITLClient(HITLClientBase):
    def get_io_manager(self) -> IOManagerBase:
        return StdIOManager()