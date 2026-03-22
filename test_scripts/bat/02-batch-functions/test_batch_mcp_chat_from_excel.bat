setlocal enabledelayedexpansion
chcp 65001
cd "%~dp0"
set "UV_PROJECT_DIR=%~dp0..\..\..\app"
set "AI_CHAT_UTIL_CONFIG=%~dp0ai-chat-util-config.yml"
uv --directory "%UV_PROJECT_DIR%" run -m ai_chat_util.cli --config "%AI_CHAT_UTIL_CONFIG%" batch_chat --use_mcp -i ..\data\input2.xlsx -p "指定したファイルをMCPツールを使って分析してください"
exit /b %errorlevel%
