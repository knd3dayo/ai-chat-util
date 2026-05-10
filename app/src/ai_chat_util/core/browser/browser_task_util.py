from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm.openai.chat import ChatOpenAI

from ai_chat_util.core.browser.model import BrowserTaskResult
import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


class BrowserTaskUtil:
    """Utility class that wraps browser-use's Agent for AI-driven browser automation."""

    @classmethod
    async def run_task(
        cls,
        task: str,
        browser_llm: ChatOpenAI,
        *,
        allowed_domains: list[str] | None = None,
        headless: bool = True,
        max_steps: int = 50,
        use_vision: bool = True,
        save_gif_path: str | None = None,
        executable_path: str | Path | None = None,
    ) -> BrowserTaskResult:
        """Execute a browser automation task and return the result as plain text.

        Args:
            task: Natural language description of the task to perform.
            browser_llm: ChatLiteLLM instance created from runtime config.
            allowed_domains: Restrict navigation to these domains (e.g. ["google.com"]).
            headless: Run browser in headless mode. Default True.
            max_steps: Maximum number of agent steps before stopping. Default 50.
            use_vision: Whether to use vision (screenshot) for page understanding. Default True.
            save_gif_path: If set, save an animated GIF of the session to this path.

        Returns:
            BrowserTaskResult with the final output, done flag and step count.
        """
        started = time.perf_counter()
        logger.info(
            "BROWSER_TASK_START max_steps=%d headless=%s allowed_domains=%s",
            max_steps,
            headless,
            allowed_domains,
        )

        profile_kwargs: dict = {
            "headless": headless,
            "allowed_domains": allowed_domains or [],
        }
        if executable_path is not None:
            profile_kwargs["executable_path"] = Path(executable_path)
        browser_profile = BrowserProfile(**profile_kwargs)
        browser_session = BrowserSession(browser_profile=browser_profile)

        generate_gif: bool | str = save_gif_path if save_gif_path else False

        agent = Agent(
            task=task,
            llm=browser_llm,
            browser_session=browser_session,
            use_vision=use_vision,
            generate_gif=generate_gif,
        )

        try:
            history = await agent.run(max_steps=max_steps)
            output: str = history.final_result() or ""
            is_done: bool = history.is_done()
            n_steps: int = len(history)
            logger.info(
                "BROWSER_TASK_END is_done=%s n_steps=%d elapsed_ms=%d",
                is_done,
                n_steps,
                int((time.perf_counter() - started) * 1000),
            )
            return BrowserTaskResult(output=output, is_done=is_done, n_steps=n_steps)
        except Exception as exc:
            logger.exception("BROWSER_TASK_ERR")
            return BrowserTaskResult(
                output=_build_error_output("run_task", exc),
                is_done=False,
                n_steps=0,
            )
        finally:
            await _safe_kill(browser_session)

    @classmethod
    async def run_task_with_output(
        cls,
        task: str,
        browser_llm: ChatOpenAI,
        output_model_schema: type[Any],
        *,
        allowed_domains: list[str] | None = None,
        headless: bool = True,
        max_steps: int = 50,
        use_vision: bool = True,
        save_gif_path: str | None = None,
        executable_path: str | Path | None = None,
    ) -> BrowserTaskResult:
        """Execute a browser automation task with structured (Pydantic model) output.

        Args:
            task: Natural language description of the task to perform.
            browser_llm: ChatLiteLLM instance created from runtime config.
            output_model_schema: A Pydantic BaseModel class describing the expected output structure.
            allowed_domains: Restrict navigation to these domains.
            headless: Run browser in headless mode. Default True.
            max_steps: Maximum number of agent steps before stopping. Default 50.
            use_vision: Whether to use vision for page understanding. Default True.
            save_gif_path: If set, save an animated GIF of the session to this path.

        Returns:
            BrowserTaskResult where ``output`` is the JSON-serialised structured result.
        """
        started = time.perf_counter()
        logger.info(
            "BROWSER_TASK_STRUCTURED_START max_steps=%d headless=%s schema=%s allowed_domains=%s",
            max_steps,
            headless,
            output_model_schema.__name__,
            allowed_domains,
        )

        profile_kwargs2: dict = {
            "headless": headless,
            "allowed_domains": allowed_domains or [],
        }
        if executable_path is not None:
            profile_kwargs2["executable_path"] = Path(executable_path)
        browser_profile = BrowserProfile(**profile_kwargs2)
        browser_session = BrowserSession(browser_profile=browser_profile)

        generate_gif: bool | str = save_gif_path if save_gif_path else False

        agent = Agent(
            task=task,
            llm=browser_llm,
            browser_session=browser_session,
            use_vision=use_vision,
            generate_gif=generate_gif,
            output_model_schema=output_model_schema,
        )

        try:
            history = await agent.run(max_steps=max_steps)
            raw: str = history.final_result() or ""
            is_done: bool = history.is_done()
            n_steps: int = len(history)
            logger.info(
                "BROWSER_TASK_STRUCTURED_END is_done=%s n_steps=%d elapsed_ms=%d",
                is_done,
                n_steps,
                int((time.perf_counter() - started) * 1000),
            )
            return BrowserTaskResult(output=raw, is_done=is_done, n_steps=n_steps)
        except Exception as exc:
            logger.exception("BROWSER_TASK_STRUCTURED_ERR")
            return BrowserTaskResult(
                output=_build_error_output("run_task_with_output", exc),
                is_done=False,
                n_steps=0,
            )
        finally:
            await _safe_kill(browser_session)


def _build_error_output(operation: str, exc: Exception) -> str:
    """Build a JSON string that callers can parse when task execution fails."""
    return json.dumps(
        {
            "ok": False,
            "operation": operation,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    )


async def _safe_kill(browser_session: BrowserSession) -> None:
    """Close browser session without masking previous errors/returns."""
    try:
        await browser_session.kill()
    except Exception:
        logger.exception("BROWSER_SESSION_KILL_ERR")
