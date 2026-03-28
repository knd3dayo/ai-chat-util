@echo off
setlocal
pushd "%~dp0" || exit /b 1

set "UV_PROJECT_DIR=%~dp0..\..\..\app"
set "LLM_API_KEY=sk-poc-master-key-12345"
set "AI_CHAT_UTIL_CONFIG=%~dp0ai-chat-util-config.yml"

uv --directory "%UV_PROJECT_DIR%" run -m coding_agent_util.mcp.mcp_server -m http -p 7102 --config "%AI_CHAT_UTIL_CONFIG%" -v DEBUG

set "RET=%ERRORLEVEL%"
popd
exit /b %RET%

