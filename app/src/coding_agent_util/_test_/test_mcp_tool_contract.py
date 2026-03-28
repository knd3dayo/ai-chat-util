from __future__ import annotations

import asyncio

from fastmcp import FastMCP

from coding_agent_util.core.endpoint import EndPoint
from coding_agent_util.mcp.mcp_server import CodingMCPServer

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
        dumped = execute.model_dump()

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

    asyncio.run(_run())