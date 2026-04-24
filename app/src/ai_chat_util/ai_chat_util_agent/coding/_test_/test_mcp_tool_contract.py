from __future__ import annotations

import asyncio

from fastmcp import FastMCP

from ai_chat_util.ai_chat_util_agent.coding.core.endpoint import EndPoint
from ai_chat_util.ai_chat_util_agent.coding.mcp.mcp_server import CodingMCPServer

an_server = CodingMCPServer()
endpoint = EndPoint()


def test_mcp_tools_are_stable_and_execute_requires_req() -> None:
    async def _run() -> None:
        mcp = FastMCP("test")
        an_server.prepare_mcp(endpoint, mcp, tools_option="", sync_mode=False)

        tools = await mcp._list_tools()
        names = {t.name for t in tools}
        assert names == {"healthz", "execute", "status", "cancel", "workspace_path", "get_result"}

        execute = next(t for t in tools if t.name == "execute")
        status = next(t for t in tools if t.name == "status")
        get_result = next(t for t in tools if t.name == "get_result")
        dumped = execute.model_dump()
        status_dumped = status.model_dump()
        get_result_dumped = get_result.model_dump()

        params = dumped["parameters"]
        assert params.get("type") == "object"
        assert "req" in params.get("required", [])

        req_schema = params["properties"]["req"]
        if "$ref" in req_schema:
            ref = req_schema["$ref"]
            assert isinstance(ref, str)
            name = ref.split("/")[-1]
            defs = params.get("$defs") or {}
            assert name in defs
            req_schema = defs[name]

        assert req_schema.get("type") == "object"
        assert set(req_schema.get("required", [])) >= {"prompt", "workspace_path"}

        status_params = status_dumped["parameters"]
        assert "wait_seconds" in status_params.get("properties", {})

        get_result_params = get_result_dumped["parameters"]
        assert "wait_seconds" in get_result_params.get("properties", {})

    asyncio.run(_run())