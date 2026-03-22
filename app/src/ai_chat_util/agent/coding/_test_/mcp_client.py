"""Test FastMCP client for the coding executor MCP server.

This module is intended for manual / integration testing.

Server (streamable-http) example:

	uv run -m ai_chat_util.agent.coding.mcp.mcp_server --mode http --host 127.0.0.1 -p 7101

Client example:

	uv run -m ai_chat_util.agent.coding._test_.mcp_client \
	  --url http://127.0.0.1:7101/mcp \
	  --workspace-path /tmp/ai_platform_ws_mcp_client_1 \
	  --prompt "sleep 1; echo hello > hello.txt" \
	  --wait

Notes:
- The server exposes MCP tools: healthz, execute, status, cancel.
- `execute` takes a single argument named `req`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from mcp.types import TextContent


def _parse_headers(raw_headers: Iterable[str]) -> dict[str, str]:
	headers: dict[str, str] = {}
	for item in raw_headers:
		s = (item or "").strip()
		if not s:
			continue
		if ":" not in s:
			raise ValueError(f"Invalid --header value (expected key:value): {item!r}")
		k, v = s.split(":", 1)
		k = k.strip()
		v = v.strip()
		if not k:
			raise ValueError(f"Invalid --header value (empty key): {item!r}")
		headers[k] = v
	return headers


def _extract_json_from_call_tool_result(tool_result: Any) -> Dict[str, Any]:
	"""Best-effort JSON extraction from FastMCP Client.call_tool result."""

	if isinstance(tool_result, dict):
		return tool_result

	content = getattr(tool_result, "content", None)
	if not isinstance(content, list) or not content:
		return {"result": tool_result}

	for item in content:
		if isinstance(item, TextContent):
			text = item.text
			if not isinstance(text, str):
				continue
			text = text.strip()
			if not text:
				continue
			try:
				parsed = json.loads(text)
				if isinstance(parsed, dict):
					return parsed
				return {"result": parsed}
			except Exception:
				return {"result": text}

	return {"result": repr(content)}


@dataclass(frozen=True)
class McpClientConfig:
	url: str
	headers: dict[str, str]


class CodingExecutorTestMcpClient:
	def __init__(self, *, url: str, headers: Optional[dict[str, str]] = None) -> None:
		url = (url or "").strip()
		if not url:
			raise ValueError("MCP url is required")
		self._cfg = McpClientConfig(url=url, headers=headers or {})

	async def list_tools(self) -> Any:
		transport = StreamableHttpTransport(url=self._cfg.url, headers=self._cfg.headers)
		async with Client(transport) as client:
			return await client.list_tools()

	async def call_tool(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> Dict[str, Any]:
		transport = StreamableHttpTransport(url=self._cfg.url, headers=self._cfg.headers)
		async with Client(transport) as client:
			res = await client.call_tool(tool_name, arguments=arguments or {})
		return _extract_json_from_call_tool_result(res)

	async def healthz(self) -> Dict[str, Any]:
		return await self.call_tool("healthz")

	async def execute(
		self,
		*,
		prompt: str,
		workspace_path: str,
		timeout: int,
		trace_id: Optional[str] = None,
		task_id: Optional[str] = None,
	) -> str:
		req: dict[str, Any] = {
			"prompt": prompt,
			"workspace_path": workspace_path,
			"timeout": int(timeout),
		}
		if trace_id:
			req["trace_id"] = trace_id
		if task_id:
			req["task_id"] = task_id

		# Server-side implementation may expose either:
		# - "execute" (if explicitly named), or
		# - "execute_async" / "execute_sync" (default FastMCP naming from function name)
		last_error: Optional[BaseException] = None
		data: Dict[str, Any] = {}
		for tool_name in ("execute", "execute_async", "execute_sync"):
			try:
				data = await self.call_tool(tool_name, {"req": req})
				last_error = None
				break
			except BaseException as e:
				last_error = e
				continue

		if last_error is not None:
			raise last_error
		tid = data.get("task_id")
		if not isinstance(tid, str) or not tid:
			raise RuntimeError(f"Invalid execute response: {data}")
		return tid

	async def status(self, task_id: str, *, tail: Optional[int] = 200) -> Dict[str, Any]:
		args: dict[str, Any] = {"task_id": task_id}
		if tail is not None:
			args["tail"] = tail
		return await self.call_tool("status", args)

	async def cancel(self, task_id: str) -> Dict[str, Any]:
		return await self.call_tool("cancel", {"task_id": task_id})


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Test FastMCP client for coding executor")
	p.add_argument(
		"--url",
		type=str,
		default="http://127.0.0.1:7101/mcp",
		help="MCP endpoint URL (FastMCP streamable-http). Example: http://127.0.0.1:7101/mcp",
	)
	p.add_argument(
		"--header",
		action="append",
		default=[],
		help="HTTP header to send (repeatable). Format: 'Key: Value'",
	)
	p.add_argument("--list-tools", action="store_true", help="List tools and exit")
	p.add_argument("--healthz", action="store_true", help="Call healthz and exit")

	p.add_argument("--prompt", type=str, default="echo hello > hello.txt", help="Prompt to execute")
	p.add_argument(
		"--workspace-path",
		type=str,
		default="/tmp/ai_platform_ws_mcp_client",
		help="Absolute workspace path on the executor host",
	)
	p.add_argument("--timeout", type=int, default=60, help="Task timeout seconds")
	p.add_argument("--trace-id", type=str, default="", help="Optional trace id")
	p.add_argument("--task-id", type=str, default="", help="Optional task id")

	p.add_argument(
		"--wait",
		action="store_true",
		help="After execute, poll status until the task exits",
	)
	p.add_argument(
		"--poll-interval",
		type=float,
		default=0.25,
		help="Polling interval seconds when --wait is set",
	)
	p.add_argument(
		"--max-polls",
		type=int,
		default=240,
		help="Max polling iterations when --wait is set",
	)
	p.add_argument(
		"--tail",
		type=int,
		default=50,
		help="Tail lines when polling status",
	)
	p.add_argument(
		"--cancel-after",
		type=float,
		default=0.0,
		help="If >0, cancel the task after this many seconds (requires --wait)",
	)

	return p.parse_args(argv)


def _is_running(status: Any) -> bool:
	return status in ("pending", "running")


async def _run() -> int:
	args = _parse_args()
	headers = _parse_headers(args.header)
	client = CodingExecutorTestMcpClient(url=args.url, headers=headers)

	if args.list_tools:
		tools = await client.list_tools()
		for t in tools:
			name = getattr(t, "name", "")
			desc = getattr(t, "description", "")
			print(f"- {name}: {desc}")
		return 0

	if args.healthz:
		res = await client.healthz()
		print(json.dumps(res, ensure_ascii=False, indent=2))
		return 0

	print(f"url={args.url}")
	if headers:
		print(f"headers={list(headers.keys())}")

	# Pre-flight: healthz
	hz = await client.healthz()
	print("healthz=", json.dumps(hz, ensure_ascii=False))

	t0 = time.time()
	task_id = await client.execute(
		prompt=args.prompt,
		workspace_path=args.workspace_path,
		timeout=args.timeout,
		trace_id=(args.trace_id or None),
		task_id=(args.task_id or None),
	)
	dt = time.time() - t0
	print(f"execute.task_id={task_id} (took {dt:.3f}s)")

	st = await client.status(task_id, tail=args.tail)
	print("status=", json.dumps(st, ensure_ascii=False))

	if not args.wait:
		return 0

	if args.cancel_after and args.cancel_after > 0:
		cancel_at = time.time() + float(args.cancel_after)
	else:
		cancel_at = None

	final: Optional[dict[str, Any]] = None
	for i in range(int(args.max_polls)):
		st = await client.status(task_id, tail=args.tail)
		if i % 10 == 0:
			print(
				f"poll[{i}] status={st.get('status')} sub_status={st.get('sub_status')}"
			)

		if cancel_at is not None and time.time() >= cancel_at:
			print("cancelling...")
			res = await client.cancel(task_id)
			print("cancel=", json.dumps(res, ensure_ascii=False))
			cancel_at = None

		if not _is_running(st.get("status")):
			final = st
			break
		await asyncio.sleep(float(args.poll_interval))

	if final is None:
		print("final_status=timeout(wait loop)")
		return 2

	print("final_status=", json.dumps(final, ensure_ascii=False, indent=2))
	return 0


def main() -> None:
	raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
	main()

