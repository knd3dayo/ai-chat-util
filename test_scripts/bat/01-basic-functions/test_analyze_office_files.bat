@echo off
setlocal

chcp 65001 >nul
pushd "%~dp0" || exit /b 1

set "AI_CHAT_UTIL_CONFIG=%~dp0ai-chat-util-config.yml"

set "input_file=%~1"
if "%input_file%"=="" set "input_file=..\..\..\work\test\data\test.xlsx"

uv run -m ai_chat_util.cli --config "%AI_CHAT_UTIL_CONFIG%" analyze_office_files -i "%input_file%" -p "分析して"

set "RET=%ERRORLEVEL%"
popd
exit /b %RET%
