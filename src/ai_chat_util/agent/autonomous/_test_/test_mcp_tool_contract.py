from __future__ import annotations

import asyncio

import pytest
from fastmcp import FastMCP

from ai_chat_util.agent.autonomous.mcp.mcp_server import AutonomousMCPServer
from ai_chat_util.agent.autonomous.core.endpoint import EndPoint
an_server = AutonomousMCPServer()
endpoint = EndPoint()
def test_mcp_tools_are_stable_and_execute_requires_req() -> None:
    async def _run() -> None:
        mcp = FastMCP("test")
        an_server.prepare_mcp(endpoint,mcp, tools_option="", sync_mode=False)

        tools = await mcp._list_tools()
        names = {t.name for t in tools}
        assert names == {"healthz", "execute", "status", "cancel", "workspace_path", "get_result"}

        execute = next(t for t in tools if t.name == "execute")
        dumped = execute.model_dump()

        params = dumped["parameters"]
        assert params.get("type") == "object"
        assert "req" in params.get("required", [])

        req_schema = params["properties"]["req"]
        # JSON Schema may either inline the object schema or use a $ref into $defs.
        if "$ref" in req_schema:
            ref = req_schema["$ref"]
            assert isinstance(ref, str)
            # e.g. "#/$defs/ExecuteRequest"
            name = ref.split("/")[-1]
            defs = params.get("$defs") or {}
            assert name in defs
            req_schema = defs[name]

        assert req_schema.get("type") == "object"
        # At minimum, prompt/workspace_path are required
        assert set(req_schema.get("required", [])) >= {"prompt", "workspace_path"}

    asyncio.run(_run())
