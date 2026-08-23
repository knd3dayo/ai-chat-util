# ai_chat_util Technical Reference

このドキュメントは、ai_chat_util の実装/運用を担当する開発者向けの技術リファレンスです。
クイックスタートは [README.md](README.md) を参照してください。

## Scope

現行バージョンは次の機能に限定されています。

- テキストチャット
- Excel ベースのバッチチャット
- 画像/PDF/Office/複合ファイル解析
- FastAPI 経由の API 提供
- FastMCP 経由の MCP ツール提供
- browser-use を使ったブラウザタスク

次の機能は削除済みです。

- Docker 実行/生成機能
- Fileサーバー機能
- agent/workflow ルーティング
- DeepAgent 系エントリポイント
- coding-agent 専用 API/CLI/runtime

## Architecture

主要レイヤは次の通りです。

- Core chat: [app/src/ai_chat_util/core/chat](app/src/ai_chat_util/core/chat)
- File analysis: [app/src/ai_chat_util/core/analysis](app/src/ai_chat_util/core/analysis)
- API interface: [app/src/ai_chat_util/interfaces/api/api_server.py](app/src/ai_chat_util/interfaces/api/api_server.py)
- CLI interface: [app/src/ai_chat_util/interfaces/cli/__main__.py](app/src/ai_chat_util/interfaces/cli/__main__.py)
- MCP interface: [app/src/ai_chat_util/interfaces/mcp/mcp_server.py](app/src/ai_chat_util/interfaces/mcp/mcp_server.py)

データフローは概ね以下です。

1. Interface 層（CLI/API/MCP）が入力を受け取る
2. Core chat/analysis 層がモデル呼び出しや前処理を実行
3. 出力を interface 層で整形して返す

## Runtime Configuration

標準設定ファイルは [app/ai-chat-util-config.yml](app/ai-chat-util-config.yml) です。

- 読み込み順序
1. CLI/API 起動時の --config
2. 環境変数 AI_CHAT_UTIL_CONFIG
3. カレントディレクトリの config.yml
4. プロジェクトルートの config.yml

- 秘密情報
1. API キー等は YAML に直書きしない
2. os.environ/VAR_NAME 形式で環境変数参照する

主に使う設定セクション。

- llm
- mcp
- features（現行は analyzer 系や監査ログ中心）
- logging
- network
- office2pdf

削除済み（設定不可）セクション。

- ai_chat_util_config.file_server
- coding_agent_util.compose
- coding_agent_util.backend.task_backend=docker|compose

上記キーは ai-chat-util では受け付けません。misc-util 側の設定へ移行してください。

## CLI Reference

エントリポイントは [app/pyproject.toml](app/pyproject.toml) の ai-chat-util です。

現行サブコマンド。

- chat
- batch_chat
- analyze_image_files
- analyze_pdf_files
- analyze_office_files
- analyze_files
- show_config

補足。

- batch_chat は Excel 入出力を前提
- analyze_* は detail パラメータに low/high/auto を利用可能

## API Reference

実装は [app/src/ai_chat_util/interfaces/api/api_server.py](app/src/ai_chat_util/interfaces/api/api_server.py) です。
プレフィックスは /api/ai_chat_util。

代表エンドポイント。

- /chat
- /simple_chat
- /batch_chat
- /batch_chat_from_excel
- /analyze_image_files
- /analyze_pdf_files
- /analyze_office_files
- /analyze_files
- /analyze_image_urls
- /analyze_pdf_urls
- /analyze_office_urls
- /analyze_file_urls
- /convert_office_files_to_pdf
- /convert_pdf_files_to_images

## MCP Reference

実装は [app/src/ai_chat_util/interfaces/mcp/mcp_server.py](app/src/ai_chat_util/interfaces/mcp/mcp_server.py) です。
エントリポイントは [app/pyproject.toml](app/pyproject.toml) の ai-chat-util-mcp。

現行ツール群（抜粋）。

- run_chat
- run_simple_chat
- run_batch_chat
- run_batch_chat_from_excel
- analyze_image_files / analyze_image_urls
- analyze_pdf_files / analyze_pdf_urls
- analyze_office_files / analyze_office_urls
- analyze_files
- convert_office_files_to_pdf
- convert_pdf_files_to_images
- run_browser_task
- run_browser_task_with_output

## Testing Notes

今回の整理後に影響確認したコマンド。

- import スモーク
1. CLI/API/MCP モジュールを venv 上で import

- pytest
1. config/browser/analyze_file_util の影響範囲テストを実施
2. 33 passed
3. test_excel_util は work/detect_tables_test/answer.md 前提のため、環境依存で収集時エラー

## Migration Notes

Docker/Fileサーバー/agent/workflow から移行する場合の注意。

1. 旧コマンド（agent_chat, run_workflow, docker_compose_* 等）は利用不可
2. file_server 関連 endpoint/tool は利用不可
3. 旧 endpoint（/agent_chat, /run_deepagent_* など）は提供しない
4. MCP の script 名は ai-chat-util-mcp
5. coding-agent 系 script は廃止

必要なら、旧運用スクリプトは chat / batch_chat / analyze_* ベースに置き換えてください。
