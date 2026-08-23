# ai_chat_util

ai_chat_util は、生成AIを使ったチャット、文書解析、バッチ処理、MCP 連携をまとめて扱うためのユーティリティです。初見の人が最短で試せるように、この README はクイックスタート中心に整理しています。

高度な技術者または AI エージェント向けの詳細仕様、設定項目、運用メモ、アーキテクチャは [README_FOR_EXPERTS.md](README_FOR_EXPERTS.md) を参照してください。

## できること

- テキストチャットを LLM に送る
- Excel 入力でバッチ処理を回す
- 画像、PDF、Office 文書を解析する
- MCP サーバーとして外部エージェントにツール提供する
- API サーバーとして組み込む

## クイックスタート

### 1. 依存を入れる

```bash
uv sync
```

### 2. 設定ファイルを用意する

非秘密設定は YAML、秘密情報は環境変数または .env で管理します。

```bash
cp ./ai-chat-util-config.yml ./config.yml
```

設定ファイルのベースは [ai-chat-util-config.yml](ai-chat-util-config.yml) です。API キーなどの秘密情報は YAML に直書きせず、環境変数参照を使ってください。

例:

```yml
ai_chat_util_config:
  llm:
    provider: openai
    completion_model: poc-chat-model
    api_key: os.environ/LLM_API_KEY
```

補足:

- 現在の self-host LiteLLM 検証環境では、既定の chat / embedding モデル名は `poc-chat-model` / `poc-embedding-model` です。
- Office 文書解析は、LibreOffice がある場合は PDF 変換経由、ない場合は Word / Excel / PowerPoint の本文テキスト抽出フォールバックで動作します。

### 3. まずは CLI で試す

リポジトリルートから実行する場合の最短例です。

通常チャット:

```bash
uv run -m ai_chat_util.cli --config ./ai-chat-util-config.yml chat -p "こんにちは"
```

複数ファイル解析:

```bash
uv run -m ai_chat_util.cli --config ./ai-chat-util-config.yml analyze_files \
  -i note.txt document.pdf image.png \
  -p "内容を要約してください"
```

Office 文書解析の例:

```bash
uv run -m ai_chat_util.cli --config ./ai-chat-util-config.yml analyze_files \
  -i data/sample.docx data/sample.xlsx data/sample.pptx \
  -p "各ファイルの概要を要約してください"
```

Excel バッチ:

```bash
uv run -m ai_chat_util.cli --config ./ai-chat-util-config.yml batch_chat \
  -i data/input.xlsx \
  -p "要約してください" \
  -o output.xlsx
```

## よく使う入口

### CLI

- `chat`: LLM へ直接チャット
- `batch_chat`: Excel ベースの一括処理
- `analyze_image_files` / `analyze_pdf_files` / `analyze_office_files` / `analyze_files`: ファイル解析

### MCP サーバー

stdio で起動する最小例です。

```bash
uv run -m ai_chat_util.mcp.mcp_server
```

クライアント設定例は [sample_cline_mcp_settings.json](sample_cline_mcp_settings.json) を参照してください。

### API サーバー

FastAPI サーバーを使う場合は、設定ファイルへのパスを環境変数で渡します。

```bash
export AI_CHAT_UTIL_CONFIG=$PWD/ai-chat-util-config.yml
uv run uvicorn ai_chat_util.interfaces.api.api_server:app
```

chat リクエスト例:

```bash
curl -X POST http://127.0.0.1:8000/api/ai_chat_util/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "chat_history": {
      "messages": [
        {
          "role": "user",
          "content": [
            {"params": {"type": "text", "text": "work ディレクトリを確認して要約してください"}}
          ]
        }
      ]
    }
  }'
```

## 補足

- ワークスペース構成、依存ルール、監査ログ、MCP 詳細設定などは [README_FOR_EXPERTS.md](README_FOR_EXPERTS.md) に分離しています。

## 詳細ドキュメント

- 人向けの短い導線としてはこの README を参照してください。
- 高度な技術者または AI エージェントは [README_FOR_EXPERTS.md](README_FOR_EXPERTS.md) を参照してください。