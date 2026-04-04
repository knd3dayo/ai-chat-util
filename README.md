# ai_chat_util

ai_chat_util は、生成AIを使ったチャット、文書解析、バッチ処理、MCP 連携をまとめて扱うためのユーティリティです。初見の人が最短で試せるように、この README はクイックスタート中心に整理しています。

高度な技術者または AI エージェント向けの詳細仕様、設定項目、運用メモ、アーキテクチャ、HITL、structured routing、DeepAgents、file_server、coding-agent の詳細は [README_FOR_EXPERTS.md](README_FOR_EXPERTS.md) を参照してください。

## できること

- テキストチャットを LLM に送る
- Excel 入力でバッチ処理を回す
- 画像、PDF、Office 文書を解析する
- MCP サーバーとして外部エージェントにツール提供する
- API サーバーや coding-agent 実行基盤として組み込む

## クイックスタート

### 1. 依存を入れる

```bash
uv sync
```

DeepAgents を使う場合だけ追加依存を入れます。

```bash
cd app
uv sync --extra deepagents
```

### 2. 設定ファイルを用意する

非秘密設定は YAML、秘密情報は環境変数または .env で管理します。

```bash
cp app/ai-chat-util-config.yml ./ai-chat-util-config.yml
```

設定ファイルのベースは [app/ai-chat-util-config.yml](app/ai-chat-util-config.yml) です。API キーなどの秘密情報は YAML に直書きせず、環境変数参照を使ってください。

例:

```yml
ai_chat_util_config:
  llm:
    provider: openai
    completion_model: gpt-5
    api_key: os.environ/LLM_API_KEY
```

### 3. まずは CLI で試す

リポジトリルートから実行する場合の最短例です。

通常チャット:

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml chat -p "こんにちは"
```

MCP 連携付きチャット:

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml agent_chat -p "work ディレクトリを確認して要約してください"
```

複数ファイル解析:

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml analyze_files \
  -i note.txt document.pdf image.png \
  -p "内容を要約してください"
```

Excel バッチ:

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml batch_chat \
  -i data/input.xlsx \
  -p "要約してください" \
  -o output.xlsx
```

## よく使う入口

### CLI

- `chat`: LLM へ直接チャット
- `agent_chat`: MCP ツール込みチャット
- `batch_chat`: Excel ベースの一括処理
- `analyze_image_files` / `analyze_pdf_files` / `analyze_office_files` / `analyze_files`: ファイル解析
- `run_workflow`: Markdown + mermaid ベースのワークフロー実行

### MCP サーバー

stdio で起動する最小例です。

```bash
uv --directory ./app run -m ai_chat_util.mcp.mcp_server
```

クライアント設定例は [app/sample_cline_mcp_settings.json](app/sample_cline_mcp_settings.json) を参照してください。

### API サーバー

FastAPI サーバーを使う場合は、設定ファイルへのパスを環境変数で渡します。

```bash
export AI_CHAT_UTIL_CONFIG=$PWD/ai-chat-util-config.yml
uv --directory ./app run uvicorn ai_chat_util.api.api_server:app
```

## 補足

- coding-agent の Docker 実行例は [docker/coding-agent/images/all-in-one-image/README.md](docker/coding-agent/images/all-in-one-image/README.md) と [docker/coding-agent/images/dood/README.md](docker/coding-agent/images/dood/README.md) を参照してください。
- ワークスペース構成、依存ルール、HITL、監査ログ、MCP 詳細設定、file_server、structured routing などは [README_FOR_EXPERTS.md](README_FOR_EXPERTS.md) に分離しています。

## 詳細ドキュメント

- 人向けの短い導線としてはこの README を参照してください。
- 高度な技術者または AI エージェントは [README_FOR_EXPERTS.md](README_FOR_EXPERTS.md) を参照してください。