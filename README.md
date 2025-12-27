# ai_chat_util

## 概要

**ai_chat_util** は、生成AI（大規模言語モデル）を活用するためのクライアントライブラリです。  
チャット形式での対話、バッチ処理による一括実行、画像やPDFファイルをAIに渡して解析・応答を得るなど、柔軟な利用が可能です。

このライブラリは、MCP（Model Context Protocol）サーバーを通じてAIモデルと通信し、  
開発者が簡単に生成AI機能を自分のアプリケーションに統合できるよう設計されています。

---

## 主な機能

### 💬 チャットクライアント
- 対話型のAIチャットを実現。
- LLM（大規模言語モデル）との自然な会話をサポート。
- コンテキストを保持した継続的な会話が可能。
- OpenAI、Azure OpenAIのみ対応

### ⚙️ バッチクライアント
- 複数の入力をまとめてAIに処理させるバッチ実行機能。
- 自動化スクリプトやデータ処理パイプラインに組み込みやすい設計。

### 🖼️ 画像・PDF・Office解析
- 画像ファイル、PDFファイル、Officeドキュメント（Word, Excel, PowerPointなど）をAIに渡して内容を解析。
- 画像認識、文書要約、表データ抽出などの高度な処理をサポート。

### 🧩 MCPサーバー連携
- `mcp_server.py` により、MCPプロトコルを介して外部ツールや他のAIサービスと連携可能。
- Chat、PDF解析、画像解析などのMCPツールを提供。

---

## ディレクトリ構成

```
src/ai_chat_util/
├── agent/          # エージェント関連ユーティリティ
├── batch/          # バッチクライアント
├── llm/            # LLMクライアント・モデル設定
├── log/            # ログ設定
├── mcp/            # MCPサーバー実装
└── util/           # PDFなどのユーティリティ
```

---

## インストール

```bash
uv sync
```
## 環境変数設定

このプロジェクトでは、`.env` ファイルを使用して環境変数を管理します。  
`.env_template` を参考に `.env` ファイルを作成してください。

例：

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_COMPLETION_MODEL=gpt-5
LIBREOFFICE_PATH=C:\\Program Files\\LibreOffice\\program\\soffice.exe
```

### 主な環境変数の説明

| 変数名 | 説明 |
|--------|------|
| `LLM_PROVIDER` | 使用するLLMプロバイダ（例: openai） |
| `OPENAI_API_KEY` | OpenAI APIキー |
| `OPENAI_EMBEDDING_MODEL` | 埋め込みモデル名 |
| `OPENAI_COMPLETION_MODEL` | テキスト生成モデル名 |
| `LIBREOFFICE_PATH` | LibreOffice実行ファイルのパス（例: C:\\Program Files\\LibreOffice\\program\\soffice.exe） |

---

## コマンドラインクライアント

`ai_chat_util` には、`argparse + subcommand` で実装されたCLIが含まれます。

### 起動方法（uv）

```bash
uv run -m ai_chat_util.cli --help
```

> 補足: CLI起動時に `.env` を読み込みます（`python-dotenv`）。

### 共通オプション

```text
--loglevel  LOGLEVEL 環境変数を設定します（例: DEBUG, INFO）
--logfile   LOGFILE 環境変数を設定します（ログをファイル出力）
```

### サブコマンド

#### chat（テキストチャット）

```bash
uv run -m ai_chat_util.cli chat -p "こんにちは"
```

#### batch_chat（Excel入力のバッチチャット）

Excel の各行（`content` / `file_path`）を読み込み、指定した `prompt` を前置して LLM に送信し、
応答を `output` 列（既定）に書き込んだ Excel を出力します。

```bash
uv run -m ai_chat_util.cli batch_chat \
  -i data/input.xlsx \
  -p "要約してください" \
  -o output.xlsx
```

入力Excelの列（既定）:

- `content`: 行ごとのテキスト（空でも可）
- `file_path`: 解析対象ファイルのパス（空でも可。存在しない場合は無視）

> 注意: 入力Excelは `content` / `file_path` の **どちらか少なくとも1列** を含む必要があります。

主要オプション:

- `-i/--input_excel_path` : 入力Excelファイルパス（必須）
- `-o/--output_excel_path` : 出力Excelファイルパス（既定: `output.xlsx`）
- `--concurrency` : 同時実行数（既定: 16）
- `--content_column` : メッセージ列名（既定: `content`）
- `--file_path_column` : ファイルパス列名（既定: `file_path`）
- `--output_column` : LLM応答の出力列名（既定: `output`）
- `--image_detail` : 画像解析の detail（low/high/auto、既定: auto）

#### analyze_image_files（画像解析）

```bash
uv run -m ai_chat_util.cli analyze_image_files \
  -i a.png b.jpg \
  -p "内容を説明して" \
  --detail auto
```

#### analyze_pdf_files（PDF解析）

```bash
uv run -m ai_chat_util.cli analyze_pdf_files \
  -i document.pdf \
  -p "このPDFの要約を作成して" \
  --detail auto
```

#### analyze_office_files（Office解析：PDF化→解析）

```bash
uv run -m ai_chat_util.cli analyze_office_files \
  -i data.xlsx slide.pptx \
  -p "内容を要約して" \
  --detail auto
```

#### analyze_files（複数形式まとめて解析）

```bash
uv run -m ai_chat_util.cli analyze_files \
  -i note.txt a.png document.pdf data.xlsx \
  -p "これらをまとめて要約して" \
  --detail auto
```

---

## MCPサーバー

`ai_chat_util` は MCP（Model Context Protocol）サーバーを提供します。
MCPクライアント（例: Cline / 独自エージェント）から接続することで、チャット・画像解析・PDF解析・Office解析などのツールを利用できます。

> 補足: MCPサーバー起動時に `.env` を読み込みます（`python-dotenv` / `load_dotenv()`）。
> そのため、事前に `.env` に `OPENAI_API_KEY` 等を設定してください。

### 起動方法

#### stdio（デフォルト）

標準入出力（stdio）で起動します。MCPクライアントがサブプロセスとして起動して接続する用途を想定しています。

```bash
uv run -m ai_chat_util.mcp.mcp_server
# または明示
uv run -m ai_chat_util.mcp.mcp_server -m stdio
```

#### SSE

SSE（Server-Sent Events）で起動します。

```bash
uv run -m ai_chat_util.mcp.mcp_server -m sse -p 5001
```

#### Streamable HTTP

```bash
uv run -m ai_chat_util.mcp.mcp_server -m http -p 5001
```

### 提供ツールの指定（任意）

`-t/--tools` で、登録するツールをカンマ区切りで指定できます。
未指定の場合は、チャット/画像/PDF/Office/複数形式（files/urls）解析系がデフォルトで登録されます。

```bash
uv run -m ai_chat_util.mcp.mcp_server -m stdio -t "run_chat,analyze_pdf_files"
```

> 注意: 指定できる名前は `ai_chat_util.core.app` から import されている関数名です。

### MCPクライアント（例: Cline）向け設定例

同梱の `sample_cline_mcp_settings.json` は Cline 等のMCPクライアント設定例です。
`<REPO_PATH>` をこのリポジトリのパスに置き換えてください（例: `c:\\Users\\user\\source\\repos\\util\\ai-chat-util`）。

```json
{
  "mcpServers": {
    "AIChatUtil": {
      "timeout": 60,
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory",
        "<REPO_PATH>",
        "run",
        "-m",
        "ai_chat_util.mcp.mcp_server"
      ],
      "env": {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-****",
        "OPENAI_COMPLETION_MODEL": "gpt-4.1",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "USE_CUSTOM_PDF_ANALYZER": "true"
      }
    }
  }
}
```

> `.env` を使う場合は、上記 `env` は不要（または最小限）です。
