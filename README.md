# ai_chat_util

## 概要

**ai_chat_util** は、生成AI（大規模言語モデル）を活用するためのクライアントライブラリです。  
チャット形式での対話、バッチ処理による一括実行、画像やPDFファイルをAIに渡して解析・応答を得るなど、柔軟な利用が可能です。

このライブラリは、LLM への直接呼び出し（LiteLLM 経由）に加えて、MCP（Model Context Protocol）連携（内部クライアント / MCPサーバー提供）もサポートし、
開発者が生成AI機能を自分のアプリケーションに統合しやすいよう設計されています。

---

## 主な機能

### 💬 チャットクライアント
- 対話型のAIチャットを実現。
- LLM（大規模言語モデル）との自然な会話をサポート。
- コンテキストを保持した継続的な会話が可能。
- OpenAI / Azure OpenAI / Anthropic をサポート（`config.yml` の `llm.provider` で切り替え）

### ⚙️ バッチクライアント
- 複数の入力をまとめてAIに処理させるバッチ実行機能。

### 🖼️ 画像・PDF・Office解析
- 画像ファイル、PDFファイル、Officeドキュメント（Word, Excel, PowerPointなど）をAIに渡して内容を解析。
- 画像認識、文書要約、表データ抽出などの処理をサポート。

### 🧩 MCPサーバー連携
- `mcp_server.py` により、MCPプロトコルを介して外部ツールや他のAIサービスと連携可能。
- Chat、PDF解析、画像解析などのMCPツールを提供。

---

## ディレクトリ構成

```
src/ai_chat_util/
├── api/            # FastAPI APIサーバー
├── cli/            # CLI（argparse + subcommand）
├── config/         # 設定（config.yml / MCP設定JSON）
├── core/           # コア処理（チャット/バッチ等）
├── llm/            # LLMクライアント実装（LiteLLM / MCP内部クライアント）
├── log/            # ログ設定
├── mcp/            # MCPサーバー実装
├── model/          # Pydanticモデル
├── test/           # サンプル/簡易スクリプト
└── util/           # PDF/Office等のユーティリティ
```

---

## インストール

```bash
uv sync
```
## 設定（config.yml + .env）

本プロジェクトは、

- **非秘密設定** → `config.yml`
- **秘密情報（APIキー/トークン等）** → 環境変数 / `.env`

に分離しています。

### 1) config.yml（必須）

`config.example.yml` をコピーして `config.yml` を作成してください。

```bash
copy config.example.yml config.yml
```

設定ファイルの探索順は以下です。

1. `--config`
2. 環境変数 `AI_CHAT_UTIL_CONFIG`
3. `./config.yml`（実行時カレント）
4. `<project-root>/config.yml`

> `config.yml` が見つからない場合はエラーになります。

#### 秘密情報（APIキー等）の扱い

秘密情報は `config.yml` に直書きできません。

- `llm.api_key` は **環境変数参照**（`os.environ/ENV_VAR_NAME`）の形式でのみ指定できます。
- `llm.extra_headers` も秘密情報を含み得るため、**各ヘッダー値は環境変数参照**（`os.environ/ENV_VAR_NAME`）でのみ指定できます。
- OpenAI / Azure OpenAI / Anthropic のプロバイダ（`llm.provider: openai | azure | anthropic`）を使う場合、`llm.api_key` は必須です（設定ロード時に検証されます）。

例（OpenAI）:

```yml
llm:
  provider: openai
  completion_model: gpt-5
  timeout_seconds: 60
  api_key: os.environ/OPENAI_API_KEY
```

例（Azure OpenAI）:

```yml
llm:
  provider: azure
  completion_model: <AZURE_DEPLOYMENT_NAME>
  base_url: https://<resource-name>.openai.azure.com/
  api_version: 2024-xx-xx
  api_key: os.environ/AZURE_API_KEY
```

例（追加ヘッダーを付けたい場合）:

```yml
llm:
  provider: openai
  completion_model: gpt-5
  api_key: os.environ/OPENAI_API_KEY
  extra_headers:
    Authorization: os.environ/OPENAI_AUTH_HEADER
    X-My-Org: os.environ/MY_ORG_ID
```

#### `--config` を渡せない起動（例: `uvicorn ...:app`）

環境変数 `AI_CHAT_UTIL_CONFIG` で `config.yml` の場所を指定してください。

```powershell
$env:AI_CHAT_UTIL_CONFIG = "C:\\path\\to\\config.yml"
uvicorn ai_chat_util.api.api_server:app
```

### 2) .env（任意・秘密のみ）

`.env.example` をコピーして `.env` を作成し、利用するプロバイダの API キー等を設定してください。

```bash
copy .env.example .env
```

### Proxy環境で `certificate verify failed` が出る場合

`analyze_*_urls` / `download_files` は URL からファイルを取得します。
社内Proxyが SSL インスペクション（MITM）を行う環境では、CA を信頼できずエラーになることがあります。

推奨は **社内CAをPEMにして `config.yml` の `network.ca_bundle` で指定**することです。

```yml
network:
  ca_bundle: "C:\\path\\to\\corp-ca.pem"
  requests_verify: true
```

切り分け用途（非推奨）で SSL 検証を無効化する場合は `network.requests_verify: false` を設定してください。

### LibreOffice（Office→PDF 変換）

Office 解析で LibreOffice を使う場合は、`config.yml` の `office2pdf.libreoffice_path` を設定します。

```yml
office2pdf:
  libreoffice_path: "C:\\Program Files\\LibreOffice\\program\\soffice.exe"
```

### 環境変数（一覧）

| 変数名 | 種別 | 説明 |
|---|---|---|
| `OPENAI_API_KEY` | 秘密 | OpenAI APIキー |
| `AZURE_API_KEY` | 秘密 | Azure OpenAI APIキー |
| `ANTHROPIC_API_KEY` | 秘密 | Anthropic APIキー |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | 秘密 | AWS Bedrock を使う場合の認証情報 |
| `AI_CHAT_UTIL_CONFIG` | 非秘密 | `config.yml` のパス（`--config` を渡せない起動で使用） |

### MCP設定（内部MCPクライアント用：任意）

CLIの `chat` / `batch_chat` は `--use_mcp` を指定すると、内部の MCP クライアントを使って「MCPツール込みのワークフロー」で実行できます。
このとき、`config.yml` の `paths.mcp_config_path`（互換キー: `paths.mcp_server_config_file_path`）に、MCPサーバー設定JSONのパスを指定してください。

```yml
paths:
  mcp_config_path: mcp_settings.json
```

> 補足: 同梱の `sample_cline_mcp_settings.json` と同じ形式です。

#### MCP設定JSONで使えるキー（補足）

- `mcpServers.<name>.type` と `mcpServers.<name>.transport` はどちらでも指定できます（内部で `transport` に正規化します）。
- `allowedTools`（任意）を指定すると、内部MCPクライアント側で「使ってよいツール名」をホワイトリストできます。

例:

```json
{
  "mcpServers": {
    "AIChatUtil": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "<REPO_PATH>", "run", "-m", "ai_chat_util.mcp.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "sk-****",
        "AI_CHAT_UTIL_CONFIG": "<REPO_PATH>\\config.yml"
      },
      "allowedTools": ["analyze_pdf_files", "analyze_files"]
    }
  }
}
```

> 注意: `--use_mcp` を付けても `paths.mcp_config_path` が未設定の場合、MCPツールはロードされません（警告ログが出ます）。

### 内部MCPクライアントの安全弁（無限ループ/タイムアウト）

内部MCPクライアント（LangGraph Supervisor + MCPツール）は、暴走/無限ループ/長時間停止を避けるために `config.yml` の `features` で上限を設定できます。

```yml
features:
  # LangGraph の再帰(ステップ)上限。大きすぎるとループ時に時間がかかります。
  mcp_recursion_limit: 15

  # 1ユーザー入力(=1 trace_id)あたりのツール呼び出し回数上限。
  mcp_tool_call_limit: 2

  # MCPツール呼び出しのハードタイムアウト(秒)。未設定(null)なら llm.timeout_seconds を使用。
  mcp_tool_timeout_seconds: null

  # タイムアウト時の再試行回数（0なら再試行しない）
  mcp_tool_timeout_retries: 1
```

---

## HITL（Human-in-the-Loop：一時停止/再開）

`ai_chat_util` は、内部MCPクライアント（LangGraph Supervisor + MCPツール）経由で実行する場合に限り、
**人間への確認が必要になったタイミングで処理を一時停止（pause）し、回答/承認後に同じスレッドで再開（resume）**できます。

### 前提（重要）

- HITL（pause/resume）が発生するのは **`--use_mcp`（内部MCPクライアント）を使う場合のみ**です。
  - `--use_mcp` を付けない場合（LiteLLM経由の直接呼び出し）は、HITLは発生しません。
- `run_simple_chat` / `run_simple_batch_chat` は常に LiteLLM 経由の直接呼び出し（MCP非対応）で実行されます（`use_mcp` 引数はありません）。
- `--use_mcp` を使う場合は、`config.yml` の `paths.mcp_config_path`（互換: `paths.mcp_server_config_file_path`）が必要です。
- CLI の `chat` は、同一プロセス内での対話として pause/resume を処理します（プロセスを跨いだ再開のための `trace_id` 指定オプションは現状ありません）。
  - プロセスを跨いで再開したい場合は、API/ライブラリ利用で `ChatRequest.trace_id` を指定してください。

### レスポンス形（API/ライブラリ利用時）

内部MCPクライアントが「人間の判断が必要」と判断した場合、`ChatResponse` が以下の形で返ります。

- `status: "paused"`
- `trace_id`: 再開に必要（LangGraph checkpointのキーとしても使用）
- `hitl`: 人間向けのプロンプト
  - `kind: "input"`（質問への回答が必要）または `"approval"`（承認/却下が必要）
  - `prompt`: 表示すべき質問/承認内容

再開は、**同じ `trace_id` を付けて次の `ChatRequest` を送る**だけです。

### SQLiteチェックポイント（状態保存）

内部MCPクライアントは SQLite にチェックポイントを保存します。

- 既定パス: `(<working_directory または config.yml のあるディレクトリ>)/.ai_chat_util/langgraph_checkpoints.sqlite`
- `trace_id` が同一であれば、プロセスが変わっても（同じDBを参照できる限り）再開できます。

### trace_id（BFF相関ID）運用

`ChatRequest.trace_id` は **W3C trace-id 部分（32桁hex）**を想定しています。

- 例: `4bf92f3577b34da6a3ce929d0e0e4736`
- 誤って `traceparent` 全文（`00-<trace_id>-<span_id>-<flags>`）を渡しても、trace-id部分へ正規化します。
- `trace_id == 000...0`（全ゼロ）は不正として拒否します。
運用としては、BFFが発行した `trace_id` をそのまま会話キーとして渡す（pause/resume でも同じ値を使う）と扱いやすいです。

### 承認（approval）対象ツールの設定

`config.yml` の `features.hitl_approval_tools` に、**実行前に人間の承認を求めたいツール名**を列挙できます。

```yml
features:
  # HITL（Human in the loop）: 承認が必要なツール名リスト
  # 例: ["analyze_files", "analyze_pdf_files"]
  hitl_approval_tools: []
```

注意: 現状の承認ゲートは「プロンプト規約による停止（ベストエフォート）」です。
将来、ツール実行直前にコードで強制ブロックする形へ強化する余地があります。

### auto_approve（pause抑制）

`ChatRequest.auto_approve=true` を指定すると、内部MCPクライアントは **なるべく `question`（pause）を出さず自己完結**するように促します。

- `question` が返ってきた場合でも、`auto_approve_max_retries` 回まで「完了（complete）で返す」よう追加指示してリトライします。
- それでも `question` の場合は **`status="paused"` にせず、`status="completed"` でベストエフォートの回答**を返します。

注意: `auto_approve=true` の場合、`features.hitl_approval_tools` に列挙したツールも「自動承認されたもの」として扱うため、
本番利用では適用範囲や権限を限定することを推奨します。

---

## コマンドラインクライアント

`ai_chat_util` には、`argparse + subcommand` で実装されたCLIが含まれます。

### 起動方法（uv）

```bash
uv run -m ai_chat_util.cli --help
```

また、`console_scripts` のエントリポイントとして `ai-chat-util` も提供しています。

```bash
uv run ai-chat-util --help
```

> 補足: 起動時に `.env` を読み込みます（秘密情報のみ）。`config.yml` は必須です。

### 共通オプション

```text
--loglevel  ログレベルを上書き（例: DEBUG, INFO）
--logfile   ログのファイル出力先を上書き
--config    設定ファイル(config.yml)のパス
```

### サブコマンド

#### chat（テキストチャット）

```bash
uv run -m ai_chat_util.cli chat -p "こんにちは"
```

内部MCPクライアントを使う場合:

```bash
uv run -m ai_chat_util.cli chat -p "こんにちは" --use_mcp
```

HITL（pause/resume）が発生した場合:

- CLIは自動で `status="paused"` を検知し、質問/承認内容を表示して入力待ちに入ります。
- 入力後、同じ `trace_id` で自動的に再開します。

#### batch_chat（Excel入力のバッチチャット）

Excel の各行（`content` / `file_path`）を読み込み、指定した `prompt` を前置して LLM に送信し、
応答を `output` 列（既定）に書き込んだ Excel を出力します。

```bash
uv run -m ai_chat_util.cli batch_chat \
  -i data/input.xlsx \
  -p "要約してください" \
  -o output.xlsx
```

内部MCPクライアントを使う場合:

```bash
uv run -m ai_chat_util.cli batch_chat \
  -i data/input.xlsx \
  -p "要約してください" \
  -o output.xlsx \
  --use_mcp
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
> そのため、事前に `.env` に `OPENAI_API_KEY` 等（秘密情報）を設定し、`config.yml` の `llm.api_key` で参照してください。`config.yml` は必須です。

### 起動方法

#### stdio（デフォルト）

標準入出力（stdio）で起動します。MCPクライアントがサブプロセスとして起動して接続する用途を想定しています。

```bash
uv run -m ai_chat_util.mcp.mcp_server
# または明示
uv run -m ai_chat_util.mcp.mcp_server -m stdio
```

stdio モードでは、ログが stdout に混ざるとクライアント側のパースを壊すことがあるため、必要に応じて `--log_file` を指定してください。

```bash
uv run -m ai_chat_util.mcp.mcp_server -m stdio --log_file mcp_server.log -v INFO
```

#### SSE

SSE（Server-Sent Events）で起動します。

```bash
uv run -m ai_chat_util.mcp.mcp_server -m sse --host 0.0.0.0 -p 5001
```

#### Streamable HTTP

```bash
uv run -m ai_chat_util.mcp.mcp_server -m http --host 0.0.0.0 -p 5001
```

### 提供ツールの指定（任意）

`-t/--tools` で、登録するツールをカンマ区切りで指定できます。
未指定の場合は、以下の解析系ツール（files/urls）がデフォルトで登録されます。

- `analyze_image_files` / `analyze_pdf_files` / `analyze_office_files` / `analyze_files` / `analyze_documents_data`
- `analyze_image_urls` / `analyze_pdf_urls` / `analyze_office_urls` / `analyze_urls`

```bash
uv run -m ai_chat_util.mcp.mcp_server -m stdio -t "run_chat,analyze_pdf_files"
```

> 注意: `--tools` で指定したツール名が未対応の場合、起動時にエラーになります（黙って無視しません）。

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
        "OPENAI_API_KEY": "sk-****",
        "AI_CHAT_UTIL_CONFIG": "<REPO_PATH>\\config.yml"
      }
    }
  }
}
```

---

## APIサーバー（FastAPI）

FastAPI のAPIサーバーを提供します。起動時に `config.yml` が必須です。

### 起動方法

`uvicorn ...:app` のように `--config` を渡せない起動では、環境変数 `AI_CHAT_UTIL_CONFIG` を使って `config.yml` の場所を指定してください。

```powershell
$env:AI_CHAT_UTIL_CONFIG = "C:\\path\\to\\config.yml"
uvicorn ai_chat_util.api.api_server:app
```

### エンドポイント（prefix）

すべて `/api/ai_chat_util` 配下にルーティングされます。

- `POST /api/ai_chat_util/analyze_image_files`
- `POST /api/ai_chat_util/analyze_pdf_files`
- `POST /api/ai_chat_util/analyze_office_files`
- `POST /api/ai_chat_util/analyze_files`
- `POST /api/ai_chat_util/analyze_documents_data`
- `POST /api/ai_chat_util/analyze_image_urls`
- `POST /api/ai_chat_util/analyze_pdf_urls`
- `POST /api/ai_chat_util/analyze_office_urls`
- `POST /api/ai_chat_util/analyze_urls`
