@echo off
chcp 65001

rem Ensure relative paths resolve from this script directory.
pushd "%~dp0"

set "UV_PROJECT_DIR=%~dp0..\..\..\app"
set LLM_API_KEY=sk-poc-master-key-12345
set AI_CHAT_UTIL_AUTO_APPROVE=1

rem Pin config to this folder's test config.
set AI_CHAT_UTIL_CONFIG=%~dp0ai-chat-util-config.yml

rem NOTE: `SET ERRORLEVEL=0` does not reset ERRORLEVEL.
rem Run a succeeding command right before uv to ensure ERRORLEVEL is 0.
ver >nul
set workspace_dir=%cd%\..\..\..\work\e2e_sv_ws_1
if not exist "%workspace_dir%" mkdir "%workspace_dir%"
uv --directory "%UV_PROJECT_DIR%" run -m ai_chat_util.cli --config "%AI_CHAT_UTIL_CONFIG%" --loglevel INFO --logfile chat_timeout_5s.log chat --use_mcp -p "WORKSPACE_DIR=%workspace_dir% ; MCP execute の workspace_path は WORKSPACE_DIR を指定。WORKSPACE_DIR\done.txt が既に存在する場合は、上書き前に必ず内容を読み取る。内容が既に『完了2』なら書き換えずそのまま。内容が違う場合は read の後に『完了2』で上書き。その後ワークスペース内のファイル一覧を表示。execute の戻り task_id を使って get_result を呼び、get_result の stdout は（可能なら全文）そのまま出力。stderr は長い可能性があるため、tail=200 を指定して末尾のみ出力し、tail=200 と明記。"


ver >nul

rem uv run -m ai_chat_util.cli --loglevel INFO --logfile chat_timeout_5s.log chat --use_mcp -p "WORKSPACE_DIR=%workspace_dir% ; MCP execute の workspace_path は WORKSPACE_DIR を指定。ワークスペース内のファイル一覧を表示。execute の戻り task_id を使って get_result を呼び、ログを出力。"

popd

