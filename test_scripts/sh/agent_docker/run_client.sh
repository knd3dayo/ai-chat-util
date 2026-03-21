#!/bin/sh
export LLM_API_KEY=sk-poc-master-key-12345

uv run -m ai_chat_util.cli --loglevel INFO --logfile chat_timeout_5s.log chat --use_mcp -p "自律側エージェントMCPツールで、/home/user/data/workspace/e2e_sv_ws_1/done.txtに「完了2」と書き込んで.最後にget_resultでログを出力して"

# uv run -m ai_chat_util.cli --loglevel INFO --logfile chat_timeout_5s.log chat --use_mcp -p "/srv/ai_platform/workspaces/e2e_sv_ws_1/をワークスペースとしてください。 自律側エージェントMCPツールで、ワークスペース内のファイル一覧を表示して。最後にget_resultでログを出力して"

cat /home/user/data/workspace/e2e_sv_ws_1/done.txt