setlocal enabledelayedexpansion
chcp 65001

rem Force UTF-8 for Python I/O on Windows
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd "%~dp0"
set AI_CHAT_UTIL_CONFIG=config.yml

rem start with a fresh log file each run
del /q chat_timeout_5s.log 2>nul
del /q mcp_server.log 2>nul

uv run -m ai_chat_util.cli --loglevel DEBUG --logfile chat_timeout_5s.log chat --use_mcp -p "C:\Users\user\source\repos\util\ai-chat-util\work\test\data\test.pngを分析してください。"
exit /b %errorlevel%
