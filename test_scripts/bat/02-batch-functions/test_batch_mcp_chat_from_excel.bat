setlocal enabledelayedexpansion
chcp 65001
cd "%~dp0"
set AI_CHAT_UTIL_CONFIG=config.yml
uv run -m ai_chat_util.cli  batch_chat --use_mcp -i ..\data\input2.xlsx -p "指定したファイルをMCPツールを使って分析してください"
exit /b %errorlevel%
