setlocal enabledelayedexpansion
chcp 65001
cd "%~dp0"
set AI_CHAT_UTIL_CONFIG=ai-chat-util-config.yml
uv run -m ai_chat_util.cli  batch_chat -i ..\data\input.xlsx -p "要約してください"
exit /b %errorlevel%
    