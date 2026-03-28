from __future__ import annotations

import asyncio

from coding_agent_util.core.endpoint import EndPoint
from coding_agent_util.mcp.mcp_server import CodingMCPServer

__all__ = ["CodingMCPServer"]


if __name__ == "__main__":
    server = CodingMCPServer()
    try:
        asyncio.run(server.main(EndPoint()))
    except KeyboardInterrupt:
        pass


