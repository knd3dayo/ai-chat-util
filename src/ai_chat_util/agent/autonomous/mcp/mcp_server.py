import asyncio
from ..core.endpoint import EndPoint
from ai_chat_util_base.mcp.autonomous_mcp_server import AutonomousMCPServer

if __name__ == "__main__":

	server = AutonomousMCPServer()
	asyncio.run(server.main(EndPoint()))

