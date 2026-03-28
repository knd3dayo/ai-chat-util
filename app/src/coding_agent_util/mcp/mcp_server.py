from __future__ import annotations

import argparse
import asyncio
import logging
import os
from functools import wraps
import inspect
from typing import Callable
import time
from pathlib import Path
import tempfile

from ai_chat_util_base.config.runtime import (
	init_coding_runtime,
	get_coding_runtime_config,
	apply_logging_overrides,
)
from fastmcp import FastMCP, Context

from ..core.abstract_endpoint import CodingEndPointBase
from ai_chat_util_base.model.request_headers import RequestHeaders, bind_current_request_headers
from ..core.endpoint import EndPoint

default_port = 7101


logger = logging.getLogger(__name__)


def _ensure_workspace_root_writable() -> None:
	"""Fail fast if configured workspace_root is not writable.

	This MCP server creates the workspace directory (mkdir) for each execute request.
	In rewrite-free deployments, the workspace_root must be consistently bind-mounted
	and writable in the MCP server's own runtime environment.
	"""
	try:
		cfg = get_coding_runtime_config()
	except Exception as e:
		raise RuntimeError(f"Failed to load coding runtime config: {e}") from e

	root_raw = (getattr(getattr(cfg, "paths", None), "workspace_root", None) or "").strip()
	if not root_raw:
		# Keep existing behavior if user did not configure it.
		return

	root = Path(os.path.expanduser(root_raw))
	if not root.is_absolute():
		raise RuntimeError(f"paths.workspace_root must be an absolute path: {root_raw!r}")

	try:
		root.mkdir(parents=True, exist_ok=True)
	except Exception as e:
		raise RuntimeError(f"paths.workspace_root is not creatable: {root.as_posix()} ({e})") from e

	# Writable check: create a temporary directory under the root.
	try:
		with tempfile.TemporaryDirectory(prefix="ai-chat-util-wscheck-", dir=root.as_posix()):
			pass
	except Exception as e:
		raise RuntimeError(f"paths.workspace_root is not writable: {root.as_posix()} ({e})") from e


class CodingMCPServer:
	def _clip_text(self, text: object, *, max_chars: int) -> str | None:
		if not isinstance(text, str):
			return None
		value = text.strip()
		if not value:
			return None
		if len(value) <= max_chars:
			return value
		head = max_chars // 2
		tail = max_chars - head
		return value[:head] + "\n...<truncated>...\n" + value[-tail:]

	def _compact_task_status_result(self, result):
		cfg = get_coding_runtime_config()
		max_chars = int(getattr(getattr(cfg, "endpoint", None), "max_tool_result_chars", 4000) or 4000)
		status_obj = result.model_copy(deep=True)
		status_obj.stdout = self._clip_text(status_obj.stdout, max_chars=max_chars // 2)
		status_obj.stderr = self._clip_text(status_obj.stderr, max_chars=max_chars // 3)
		if isinstance(status_obj.artifacts, list) and len(status_obj.artifacts) > 10:
			status_obj.artifacts = status_obj.artifacts[:10]
		if isinstance(status_obj.metadata, dict):
			status_obj.metadata = {
				"metadata_keys": sorted(str(k) for k in status_obj.metadata.keys())[:10],
				"artifact_count": len(status_obj.artifacts or []),
			}
		return status_obj

	def _compact_get_result_payload(self, result):
		cfg = get_coding_runtime_config()
		max_chars = int(getattr(getattr(cfg, "endpoint", None), "max_tool_result_chars", 4000) or 4000)
		if not isinstance(result, dict):
			return result
		stdout = self._clip_text(result.get("stdout"), max_chars=max_chars)
		stderr = self._clip_text(result.get("stderr"), max_chars=max_chars // 2)
		return {
			"stdout": stdout,
			"stderr": stderr,
			"stdout_truncated": isinstance(result.get("stdout"), str) and stdout is not None and stdout != result.get("stdout"),
			"stderr_truncated": isinstance(result.get("stderr"), str) and stderr is not None and stderr != result.get("stderr"),
		}

	def _compact_mcp_result(self, tool_name: str, result):
		if tool_name == "status" and hasattr(result, "model_copy"):
			return self._compact_task_status_result(result)
		if tool_name == "get_result":
			return self._compact_get_result_payload(result)
		return result

	"""
	Standalone MCP server exposing coding endpoint tools.

	Supports multiple transport modes (stdio, sse, streamable-http) and configurable tool selection.
	Designed for easy integration with various clients while providing robust logging and error handling.
	"""
	def _summarize_mcp_args(self,tool_name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> dict[str, object]:
		# Avoid logging secrets and large payloads. Only log argument keys and selected safe metadata.
		summary: dict[str, object] = {"tool": tool_name, "arg_count": len(args), "kw_keys": sorted(kwargs.keys())}

		# FastMCP passes tool args via kwargs; our tools typically use either:
		# - req={...}
		# - task_id=... tail=...
		req = kwargs.get("req")
		if isinstance(req, dict):
			keys = sorted(str(k) for k in req.keys())
			summary["req_keys"] = keys
			prompt = req.get("prompt")
			if isinstance(prompt, str):
				summary["prompt_chars"] = len(prompt)
				summary["prompt_lines"] = prompt.count("\n") + 1 if prompt else 0
			ws = req.get("workspace_path")
			if isinstance(ws, str):
				summary["workspace_path"] = ws
			for k in ("task_id", "trace_id"):
				v = req.get(k)
				if isinstance(v, str) and v:
					summary[k] = v
			timeout = req.get("timeout")
			if isinstance(timeout, int):
				summary["timeout"] = timeout

		for k in ("task_id", "tail"):
			v = kwargs.get(k)
			if isinstance(v, (str, int)):
				summary[k] = v
		wait_seconds = kwargs.get("wait_seconds")
		if isinstance(wait_seconds, (int, float)):
			summary["wait_seconds"] = wait_seconds

		return summary

	def parse_args(self) -> argparse.Namespace:
		parser = argparse.ArgumentParser(description="Run Coding Agent Executor MCP server")
		parser.add_argument(
			"--config",
			type=str,
			default="",
			help=(
				"Path to config YAML (ai-chat-util-config.yml). If omitted, resolved by env AI_CHAT_UTIL_CONFIG "
				"(with root-level coding_agent_util section), or searched from CWD/project root."
			),
		)
		parser.add_argument(
			"-m",
			"--mode",
			choices=["sse", "http", "stdio"],
			default="stdio",
			help=(
				"Transport mode: 'stdio' (default), 'sse', or 'http' (streamable-http)."
			),
		)
		parser.add_argument(
			"-t",
			"--tools",
			type=str,
			default="",
			help=(
				"Comma-separated list of tool names to expose. "
				"Supported: healthz, execute, status, cancel, workspace_path, get_result. "
				"If omitted, all tools are exposed."
			),
		)
		parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host for sse/http")
		parser.add_argument("-p", "--port", type=int, default=default_port, help="Bind port for sse/http")
		parser.add_argument(
			"-v",
			"--log_level",
			type=str,
			default="",
			help=(
				"Log level override (process-local). "
				"fastmcp itself may not use it, but downstream libs can."
			),
		)
		parser.add_argument(
			"--sync_mode",
			action="store_true",
			help="Run MCP server in synchronous mode.",
		)

		return parser.parse_args()


	def _normalize_tool_name(self, name: str) -> str:
		name = name.strip()
		if not name:
			return name
		# Allow endpoint.healthz / endpoint.healthz-like inputs.
		if "." in name:
			name = name.split(".")[-1]
		return name


	def prepare_mcp(self, endpoint: CodingEndPointBase, mcp: FastMCP, tools_option: str, sync_mode: bool) -> None:
		def header_aware_tool(mcp_instance: FastMCP, *, tool_name: str):
			def decorator(func):
				@wraps(func)
				async def wrapper(*args, **kwargs):
					start = time.perf_counter()
					context = kwargs.pop("context", None)
					headers_obj: RequestHeaders | None = None
					if isinstance(context, Context):
						request_context = getattr(context, "request_context", None)
						request = getattr(request_context, "request", None) if request_context else None
						if request is not None:
							headers = {str(k).lower(): str(v) for k, v in request.headers.items()}
							headers_obj = RequestHeaders.from_mapping(headers)

					# Log tool call (avoid secrets; rely on redacting formatter as backstop).
					try:
						trace_id = headers_obj.trace_id if headers_obj else None
						logger.info("mcp.request %s", {**self._summarize_mcp_args(tool_name, args, kwargs), "trace_id": trace_id})
					except Exception:
						# Never fail the request due to logging.
						pass

					with bind_current_request_headers(headers_obj):
						try:
							result = await func(*args, **kwargs)
						except Exception:
							dt_ms = int((time.perf_counter() - start) * 1000)
							try:
								logger.exception(
									"mcp.error tool=%s dt_ms=%s trace_id=%s",
									tool_name,
									dt_ms,
									headers_obj.trace_id if headers_obj else None,
								)
							except Exception:
								pass
							raise

						result = self._compact_mcp_result(tool_name, result)

					dt_ms = int((time.perf_counter() - start) * 1000)
					try:
						# Best-effort response summary.
						shape = type(result).__name__
						logger.info("mcp.response tool=%s dt_ms=%s trace_id=%s result_type=%s", tool_name, dt_ms, headers_obj.trace_id if headers_obj else None, shape)
					except Exception:
						pass
					return result

				# IMPORTANT: Expose stable tool names expected by clients
				# (e.g. client calls "execute" while underlying implementation may be "execute_async").
				wrapper.__name__ = tool_name

				sig = inspect.signature(func)
				params = list(sig.parameters.values())
				if "context" not in [p.name for p in params]:
					params.append(
						inspect.Parameter(
							"context",
							inspect.Parameter.KEYWORD_ONLY,
							annotation=Context,
							default=None,
							)
					)
				setattr(wrapper, "__signature__", sig.replace(parameters=params))

				# fastmcp uses Pydantic TypeAdapter/get_type_hints on the function.
				# When we add a param via __signature__, we must also provide its annotation.
				# Otherwise get_type_hints() won't include it and schema generation fails.
				ann = dict(getattr(wrapper, "__annotations__", {}) or {})
				ann.setdefault("context", Context)
				wrapper.__annotations__ = ann
				return mcp_instance.tool()(wrapper)

			return decorator

		tool_registry: dict[str, Callable[..., object]] = {
			"healthz": endpoint.healthz,
			"execute": endpoint.execute_async if not sync_mode else endpoint.execute_sync,
			"status": endpoint.status,
			"cancel": endpoint.cancel,
			"workspace_path": endpoint.workspace_path,
			"get_result": endpoint.get_result,
		}

		# By default expose all supported tools.
		default_tools = ("healthz", "execute", "status", "cancel", "workspace_path", "get_result")

		if tools_option:
			selected = [self._normalize_tool_name(t) for t in tools_option.split(",")]
			missing = [t for t in selected if t and t not in tool_registry]
			if missing:
				raise ValueError(
					f"Unknown tool(s): {missing}. Supported: {sorted(tool_registry.keys())}"
				)
			for t in selected:
				if not t:
					continue
				header_aware_tool(mcp, tool_name=t)(tool_registry[t])
			return

		for name in default_tools:
			header_aware_tool(mcp, tool_name=name)(tool_registry[name])


	async def main(self, endpoint: CodingEndPointBase) -> None:
		args = self.parse_args()
		init_coding_runtime(args.config or None)
		_ensure_workspace_root_writable()
		if args.log_level:
			apply_logging_overrides(level=args.log_level)

		mcp = FastMCP("coding_agent_executor")
		self.prepare_mcp(endpoint, mcp, args.tools, args.sync_mode)

		if args.mode == "stdio":
			await mcp.run_async()
			return

		if args.mode == "sse":
			await mcp.run_async(transport="sse", host=args.host, port=args.port)
			return

		# args.mode == "http"
		await mcp.run_async(transport="streamable-http", host=args.host, port=args.port)


def main() -> None:
	server = CodingMCPServer()
	try:
		asyncio.run(server.main(EndPoint()))
	except KeyboardInterrupt:
		pass


if __name__ == "__main__":
	main()


