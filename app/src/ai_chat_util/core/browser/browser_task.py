from __future__ import annotations

import json
import time
from typing import Annotated

from pydantic import Field, create_model

from ai_chat_util.core.browser.base import create_browser_llm, get_default_chromium_path
from ai_chat_util.core.browser.browser_task_util import BrowserTaskUtil
import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


async def run_browser_task(
    task: Annotated[str, Field(description="Natural language description of the browser automation task to perform")],
    allowed_domains: Annotated[
        list[str] | None,
        Field(description="Restrict browser navigation to these domains (e.g. ['google.com', 'wikipedia.org']). None means no restriction."),
    ] = None,
    headless: Annotated[
        bool,
        Field(description="Run the browser in headless mode (no GUI). Default True."),
    ] = True,
    max_steps: Annotated[
        int,
        Field(description="Maximum number of agent steps before stopping. Default 50.", ge=1, le=500),
    ] = 50,
    use_vision: Annotated[
        bool,
        Field(description="Whether to use screenshots for visual page understanding. Default True."),
    ] = True,
    save_gif_path: Annotated[
        str | None,
        Field(description="Absolute path to save an animated GIF recording of the session. None to skip recording."),
    ] = None,
) -> Annotated[str, Field(description="Text output or extracted content from the completed task")]:
    """
    Execute a browser automation task driven by an AI agent and return the result.

    The agent navigates web pages, clicks elements, fills forms, and extracts information
    to accomplish the specified task using the configured LLM.
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=run_browser_task task_len=%d headless=%s max_steps=%d",
        len(task or ""),
        headless,
        max_steps,
    )
    browser_llm = create_browser_llm()
    executable_path = get_default_chromium_path()
    if executable_path:
        logger.debug("Using Chromium at %s", executable_path)
    try:
        result = await BrowserTaskUtil.run_task(
            task=task,
            browser_llm=browser_llm,
            allowed_domains=allowed_domains,
            headless=headless,
            max_steps=max_steps,
            use_vision=use_vision,
            save_gif_path=save_gif_path,
            executable_path=executable_path,
        )
        return result.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=run_browser_task")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=run_browser_task elapsed_ms=%s", elapsed_ms)


async def run_browser_task_with_output(
    task: Annotated[str, Field(description="Natural language description of the browser automation task to perform")],
    output_schema_json: Annotated[
        str,
        Field(
            description=(
                "JSON Schema string describing the expected structured output. "
                "Must be a JSON object schema with a 'properties' key. "
                'Example: {"properties": {"title": {"type": "string"}, "url": {"type": "string"}}}'
            )
        ),
    ],
    allowed_domains: Annotated[
        list[str] | None,
        Field(description="Restrict browser navigation to these domains. None means no restriction."),
    ] = None,
    headless: Annotated[
        bool,
        Field(description="Run the browser in headless mode (no GUI). Default True."),
    ] = True,
    max_steps: Annotated[
        int,
        Field(description="Maximum number of agent steps before stopping. Default 50.", ge=1, le=500),
    ] = 50,
    use_vision: Annotated[
        bool,
        Field(description="Whether to use screenshots for visual page understanding. Default True."),
    ] = True,
    save_gif_path: Annotated[
        str | None,
        Field(description="Absolute path to save an animated GIF recording of the session. None to skip recording."),
    ] = None,
) -> Annotated[str, Field(description="JSON string of the structured output matching the provided schema")]:
    """
    Execute a browser automation task and return structured JSON output.

    The output_schema_json defines the expected structure as a JSON Schema object.
    The agent extracts information matching this schema from the web pages it visits.
    The result is returned as a JSON string.
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=run_browser_task_with_output task_len=%d headless=%s max_steps=%d",
        len(task or ""),
        headless,
        max_steps,
    )
    browser_llm = create_browser_llm()
    executable_path = get_default_chromium_path()
    if executable_path:
        logger.debug("Using Chromium at %s", executable_path)
    try:
        schema_dict = json.loads(output_schema_json)
        properties: dict = schema_dict.get("properties", {})
        if not properties:
            raise ValueError(
                "output_schema_json must be a JSON object schema with a 'properties' key. "
                f"Got: {output_schema_json!r}"
            )

        # Build a dynamic Pydantic model from the JSON Schema properties
        field_definitions: dict = {}
        for field_name, field_info in properties.items():
            field_type_str: str = field_info.get("type", "string")
            python_type = _json_schema_type_to_python(field_type_str)
            description = field_info.get("description", "")
            required: list = schema_dict.get("required", [])
            if field_name in required:
                field_definitions[field_name] = (python_type, Field(description=description))
            else:
                field_definitions[field_name] = (python_type | None, Field(default=None, description=description))

        output_model = create_model("BrowserOutput", **field_definitions)

        result = await BrowserTaskUtil.run_task_with_output(
            task=task,
            browser_llm=browser_llm,
            output_model_schema=output_model,
            allowed_domains=allowed_domains,
            headless=headless,
            max_steps=max_steps,
            use_vision=use_vision,
            save_gif_path=save_gif_path,
            executable_path=executable_path,
        )
        return result.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=run_browser_task_with_output")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=run_browser_task_with_output elapsed_ms=%s", elapsed_ms)


def _json_schema_type_to_python(type_str: str) -> type:
    """Map a JSON Schema primitive type string to a Python type."""
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(type_str, str)
