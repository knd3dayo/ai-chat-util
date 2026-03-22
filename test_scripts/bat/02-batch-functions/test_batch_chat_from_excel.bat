setlocal enabledelayedexpansion
chcp 65001
cd "%~dp0"
set "UV_PROJECT_DIR=%~dp0..\..\..\app"
set "AI_CHAT_UTIL_CONFIG=%~dp0ai-chat-util-config.yml"
uv --directory "%UV_PROJECT_DIR%" run -m ai_chat_util.cli --config "%AI_CHAT_UTIL_CONFIG%" batch_chat -i ..\data\input.xlsx -p "要約してください"
exit /b %errorlevel%
    