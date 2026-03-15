@echo off
setlocal

chcp 65001 >nul

rem Force UTF-8 for Python I/O on Windows
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

pushd "%~dp0" || exit /b 1
set "AI_CHAT_UTIL_CONFIG=%~dp0config.yml"

rem start with a fresh log file each run
del /q "%~dp0chat_timeout_5s.log" 2>nul

uv run -m ai_chat_util.cli --config "%AI_CHAT_UTIL_CONFIG%" --loglevel DEBUG --logfile "%~dp0chat_timeout_5s.log" chat -p "こんにちは、これはテストです。"

set "RET=%ERRORLEVEL%"
popd
exit /b %RET%
