@echo off
setlocal

chcp 65001

cd %~dp0

rem Force UTF-8 for Python I/O on Windows
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set AI_CHAT_UTIL_CONFIG=%~dp0ai-chat-util-config.yml

rem start with a fresh log file each run
del /q %~dp0chat_timeout_5s.log 2>nul

uv run -m ai_chat_util.cli --config "%AI_CHAT_UTIL_CONFIG%" --loglevel DEBUG --logfile "%~dp0chat_timeout_5s.log" chat -p "こんにちは、これはテストです。"

set RET=%ERRORLEVEL%
exit /b %RET%
