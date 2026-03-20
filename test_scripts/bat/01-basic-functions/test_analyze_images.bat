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

set "input_file=%~1"
if "%input_file%"=="" set "input_file=..\..\..\work\test\data\test.png"

uv run -m ai_chat_util.cli --config "%AI_CHAT_UTIL_CONFIG%" analyze_image_files -i "%input_file%" -p "分析して"

set "RET=%ERRORLEVEL%"
exit /b %RET%
