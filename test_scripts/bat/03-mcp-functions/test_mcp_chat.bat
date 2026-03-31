setlocal enabledelayedexpansion
chcp 65001

rem Force UTF-8 for Python I/O on Windows
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd "%~dp0"
set "UV_PROJECT_DIR=%~dp0..\..\..\app"
set "AI_CHAT_UTIL_CONFIG=%~dp0ai-chat-util-config.yml"

rem start with a fresh log file each run
del /q chat_timeout_5s.log 2>nul
del /q mcp_server.log 2>nul

uv --directory "%UV_PROJECT_DIR%" run -m ai_chat_util.cli --config "%AI_CHAT_UTIL_CONFIG%" --loglevel DEBUG --logfile chat_timeout_5s.log agent_chat -p "C:\Users\user\source\repos\util\ai-chat-util\work\test\data\test.pngを分析してください。"
exit /b %errorlevel%
