from __future__ import annotations

import json
import time
import traceback
from typing import Annotated, Any

from pydantic import Field, create_model

from ai_chat_util.core.browser.base import create_browser_llm, get_default_chromium_path
from ai_chat_util.core.browser.browser_task_util import BrowserTaskUtil
import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


_JSON_SCHEMA_PRIMITIVE_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


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
    except Exception as exc:
        logger.exception("MCP_TOOL_ERR tool=run_browser_task")
        return _format_exception_response("run_browser_task", exc)
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
        output_model = _build_output_model_from_schema(schema_dict)
        model_schema = output_model.model_json_schema()
        logger.debug(
            "BROWSER_OUTPUT_SCHEMA_BUILT fields=%s array_item_types=%s",
            sorted((model_schema.get("properties") or {}).keys()),
            _extract_array_item_types(model_schema),
        )

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
    except Exception as exc:
        logger.exception("MCP_TOOL_ERR tool=run_browser_task_with_output")
        return _format_exception_response("run_browser_task_with_output", exc)
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=run_browser_task_with_output elapsed_ms=%s", elapsed_ms)


def _json_schema_type_to_python(type_str: str) -> type:
    """Map a JSON Schema primitive type string to a Python type."""
    return _JSON_SCHEMA_PRIMITIVE_TYPES[type_str]


def _build_output_model_from_schema(schema_dict: dict[str, Any]) -> type[Any]:
    """Build a dynamic Pydantic model from a constrained JSON Schema subset."""
    properties, required_fields = _validate_output_schema(schema_dict)

    field_definitions: dict[str, tuple[Any, Field]] = {}
    for field_name, field_schema in properties.items():
        python_type = _json_schema_field_to_python(field_name, field_schema)
        description = str(field_schema.get("description", ""))
        if field_name in required_fields:
            field_definitions[field_name] = (python_type, Field(description=description))
        else:
            field_definitions[field_name] = (python_type | None, Field(default=None, description=description))

    return create_model("BrowserOutput", **field_definitions)


def _validate_output_schema(schema_dict: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Validate supported output schema shape and return normalized properties/required set."""
    properties_obj = schema_dict.get("properties")
    if not isinstance(properties_obj, dict) or not properties_obj:
        raise ValueError("output_schema_json must be a JSON object schema with a non-empty 'properties' key.")

    required_obj = schema_dict.get("required", [])
    if not isinstance(required_obj, list) or not all(isinstance(x, str) for x in required_obj):
        raise ValueError("output_schema_json 'required' must be a list of property names when present.")
    required_fields = set(required_obj)

    normalized_properties: dict[str, dict[str, Any]] = {}
    for field_name, field_schema in properties_obj.items():
        if not isinstance(field_schema, dict):
            raise ValueError(f"Property '{field_name}' must be an object schema.")
        field_type = field_schema.get("type")
        if not isinstance(field_type, str):
            raise ValueError(f"Property '{field_name}' must define a string 'type'.")
        if field_type == "array":
            items_schema = field_schema.get("items")
            if not isinstance(items_schema, dict) or not isinstance(items_schema.get("type"), str):
                raise ValueError(f"Array property '{field_name}' must define 'items.type'.")
        normalized_properties[field_name] = field_schema

    return normalized_properties, required_fields


def _json_schema_field_to_python(field_name: str, field_schema: dict[str, Any]) -> Any:
    """Convert one JSON Schema field into a Python typing annotation."""
    field_type = field_schema["type"]

    if field_type in _JSON_SCHEMA_PRIMITIVE_TYPES:
        return _json_schema_type_to_python(field_type)
    if field_type == "object":
        return dict[str, Any]
    if field_type == "array":
        items_schema = field_schema["items"]
        return list[_json_schema_array_item_to_python(field_name, items_schema)]

    raise ValueError(f"Unsupported type for property '{field_name}': {field_type!r}")


def _json_schema_array_item_to_python(field_name: str, items_schema: dict[str, Any]) -> Any:
    """Convert JSON Schema array item schema into a Python typing annotation."""
    item_type = items_schema.get("type")
    if item_type in _JSON_SCHEMA_PRIMITIVE_TYPES:
        return _json_schema_type_to_python(item_type)
    if item_type == "object":
        return dict[str, Any]
    if item_type == "array":
        nested_items_schema = items_schema.get("items")
        if not isinstance(nested_items_schema, dict):
            raise ValueError(f"Nested array in property '{field_name}' must define object 'items'.")
        return list[_json_schema_array_item_to_python(field_name, nested_items_schema)]

    raise ValueError(f"Unsupported array items.type for property '{field_name}': {item_type!r}")


def _extract_array_item_types(model_schema: dict[str, Any]) -> dict[str, str]:
    """Extract top-level array item types from generated model schema for diagnostics."""
    extracted: dict[str, str] = {}
    properties = model_schema.get("properties")
    if not isinstance(properties, dict):
        return extracted

    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue

        raw_schema = field_schema
        candidates = field_schema.get("anyOf")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict) and candidate.get("type") == "array":
                    raw_schema = candidate
                    break

        if raw_schema.get("type") != "array":
            continue

        items = raw_schema.get("items")
        if isinstance(items, dict):
            extracted[field_name] = str(items.get("type", "(missing)"))
        else:
            extracted[field_name] = "(missing)"

    return extracted


def _format_exception_response(tool_name: str, exc: Exception) -> str:
    """Return a JSON string with error details and full traceback for callers."""
    return json.dumps(
        {
            "ok": False,
            "tool": tool_name,
            "exception_occurred": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    )
