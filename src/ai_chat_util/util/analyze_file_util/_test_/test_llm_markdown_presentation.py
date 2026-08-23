from pathlib import Path

from ai_chat_util.util.analyze_file_util.llm_markdown_presentation import LLMMarkdownPresentationUtil


def test_prepare_markdown_for_presentation_uses_template_when_llm_unavailable(monkeypatch: object) -> None:
    def fake_run_llm(prompt: str, *, max_output_chars: int) -> str:
        return "# Prepared\n\n- point"

    monkeypatch.setattr(LLMMarkdownPresentationUtil, "_run_llm", staticmethod(fake_run_llm))
    result = LLMMarkdownPresentationUtil.prepare_markdown_for_presentation(
        "# Title\n\nBody",
        instruction="Make it slide-friendly",
        output_format="Use bullet points",
    )
    assert "# Prepared" in result
    assert "- point" in result
