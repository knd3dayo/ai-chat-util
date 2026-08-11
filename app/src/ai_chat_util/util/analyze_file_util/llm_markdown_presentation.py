from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ai_chat_util.core.chat import create_llm_client
from ai_chat_util.core.chat.model import ChatHistory, ChatMessage, ChatRequest, ChatRequestContext, ChatContent


class LLMMarkdownPresentationUtil:
    """LLM を使って Markdown を PowerPoint 向けに前処理・整形するユーティリティ。"""

    @classmethod
    def prepare_markdown_for_presentation(
        cls,
        markdown_text: str,
        *,
        instruction: str | None = None,
        output_format: str | None = None,
        max_output_chars: int = 8000,
    ) -> str:
        prompt = cls._build_prompt(markdown_text, instruction=instruction, output_format=output_format)
        return cls._run_llm(prompt, max_output_chars=max_output_chars)

    @classmethod
    def _build_prompt(
        cls,
        markdown_text: str,
        *,
        instruction: str | None = None,
        output_format: str | None = None,
    ) -> str:
        base_instruction = (
            "You are converting a document into concise PowerPoint-friendly Markdown. "
            "Keep the content structured, readable, and suitable for slide generation."
        )
        instruction_text = instruction or "Summarize and reorganize the content for slide presentation."
        format_text = output_format or (
            "Return only Markdown. "
            "Use headings with #, ##, ###. "
            "Use bullet lists with - for key points. "
            "Keep paragraphs short. "
            "Do not include extra commentary."
        )
        return (
            f"{base_instruction}\n\n"
            f"Instruction: {instruction_text}\n\n"
            f"Output format: {format_text}\n\n"
            "Source markdown:\n\n"
            f"{markdown_text}"
        )

    @classmethod
    def _run_llm(cls, prompt: str, *, max_output_chars: int) -> str:
        try:
            llm_client = create_llm_client()
            message = ChatMessage(
                role=llm_client.get_message_factory().get_user_role_name(),
                content=[llm_client.get_message_factory().create_text_content(prompt)],
            )
            chat_request = ChatRequest(
                chat_history=ChatHistory(messages=[message]),
                chat_request_context=ChatRequestContext(),
            )
            response = asyncio.run(llm_client.chat(chat_request))
            output = response.output if hasattr(response, "output") else ""
            if isinstance(output, str):
                text = output.strip()
            else:
                text = ""
        except Exception:
            text = ""

        if not text:
            text = prompt
        if len(text) > max_output_chars:
            text = text[:max_output_chars]
        return text
