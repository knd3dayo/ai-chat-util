from pathlib import Path

from browser_use.llm.openai.chat import ChatOpenAI

from ai_chat_util.core.common.config.runtime import get_runtime_config
import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

# Default Chromium binary installed by `playwright install chromium`
_PLAYWRIGHT_CHROMIUM_GLOB = "~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome"


def get_default_chromium_path() -> Path | None:
    """Return the path of the Playwright-managed Chromium binary, or None if not found."""
    import glob
    matches = sorted(glob.glob(str(Path(_PLAYWRIGHT_CHROMIUM_GLOB).expanduser())))
    if matches:
        return Path(matches[-1])
    return None


def create_browser_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance from the ai-chat-util runtime configuration.

    Reads the LLM settings (model, api_key, base_url) from ai-chat-util-config.yml
    and returns a ChatOpenAI suitable for use with browser-use's Agent.
    Since litellm exposes an OpenAI-compatible API, ChatOpenAI works with any
    provider by pointing base_url at the litellm proxy endpoint.
    """
    config = get_runtime_config()
    llm_config = config.llm

    model: str = llm_config.completion_model
    api_key: str | None = llm_config.api_key or None
    base_url: str | None = getattr(llm_config, "base_url", None) or None

    logger.debug(
        "create_browser_llm model=%s base_url=%s",
        model,
        base_url or "(default)",
    )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
