import asyncio
from ..core.endpoint import EndPoint
from ai_chat_util.agent.autonomous.mcp.autonomous_mcp_server import AutonomousMCPServer

if __name__ == "__main__":

	server = AutonomousMCPServer()
	try:
		asyncio.run(server.main(EndPoint()))
	except KeyboardInterrupt:
		# Allow clean shutdown without traceback on Ctrl+C.
		pass

