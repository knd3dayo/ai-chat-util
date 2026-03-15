@echo off
setlocal

chcp 65001 >nul
pushd "%~dp0" || exit /b 1

set "AI_CHAT_UTIL_CONFIG=%~dp0config.yml"

uv run ..\..\..\src\ai_chat_util\test\simple_chat.py

set "RET=%ERRORLEVEL%"
popd
exit /b %RET%
