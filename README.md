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
- OpenAI / Azure OpenAI / Anthropic をサポート（`ai-chat-util-config.yml` の `llm.provider` で切り替え）
- 実験的に、Supervisor ではなく DeepAgents を明示的に起動する `run_deepagent_chat` 入口を CLI / API / FastMCP から利用可能。

### ⚙️ バッチクライアント
- 複数の入力をまとめてAIに処理させるバッチ実行機能。

### 🖼️ 画像・PDF・Office解析
- 画像ファイル、PDFファイル、Officeドキュメント（Word, Excel, PowerPointなど）をAIに渡して内容を解析。
- 画像認識、文書要約、表データ抽出などの処理をサポート。
- 解析前処理として、Office→PDF 変換と PDF→ページ画像変換を API / MCP ツールとして利用可能。

### 🧩 MCPサーバー連携
- `mcp_server.py` により、MCPプロトコルを介して外部ツールや他のAIサービスと連携可能。
- Chat、PDF解析、画像解析などのMCPツールを提供。

### 🗺️ Markdown 駆動の WF 型ワークフロー
- Markdown 内の mermaid 図をもとに WF 型ワークフローを構築し、LangGraph で実行可能。
- 実行前に Markdown 本文と利用可能 MCP ツール一覧から mermaid 図を補正できる。
- plan モードでは更新済み Markdown を返して承認待ちにできる。

### 🤖 コーディングエージェント実行（coding-agent-util）
- コーディングエージェント実行タスクの起動・進捗確認・キャンセルを提供（HTTP API / MCP サーバ / CLI）。
- 設定は `ai-chat-util-config.yml`（`coding_agent_util:` セクションで統合）、秘密情報は `.env` / 環境変数で管理。
- 実装本体は `ai_chat_util.agent.coding` に統合しており、正規の起動点は `coding-agent-api` / `coding-agent-mcp` / `coding-agent-util` です。
- Docker 実行例は [docker/coding-agent/images/all-in-one-image/README.md](docker/coding-agent/images/all-in-one-image/README.md) と [docker/coding-agent/images/dood/README.md](docker/coding-agent/images/dood/README.md) を参照。

---

## マルチエージェント構成

本アプリの MCP 連携付きチャット実行は、LangChain の Router パターンと Subagents / Supervisor パターンを組み合わせた構成です。

- Router の考え方:
  ユーザー要求を最初に分類し、どの経路へ進むべきかを決めます。LangChain の Router パターンでいう「前段の routing step」に相当します。
- Supervisor の考え方:
  route 決定後は中央の supervisor が会話コンテキストを持ち、配下のツール実行エージェントへ委譲しながら結果をまとめます。LangChain の Subagents / Supervisor パターンに相当します。
- Tool agents の考え方:
  実際のツール呼び出しは、役割ごとに分かれた tool agent が担当します。現在は主に coding 系と general tool 系の 2 系統です。
  また、オプションで DeepAgents ベースの `deep_agent` route を追加できます。

参考 URL:

- [LangChain Router](https://docs.langchain.com/oss/python/langchain/multi-agent/router)
- [LangChain Subagents / Supervisor](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)

### このアプリでの対応関係

#### 1. Router

- 役割: ユーザー入力、明示ファイルパス、利用可能ツール群をもとに、最初に進むべき経路を決定します。
- 主な遷移先: `coding_agent` / `general_tool_agent` / `direct_answer` / `reject`
- オプション遷移先: `deep_agent`
- 性質: 会話全体を統括する main agent ではなく、最初の経路選択に特化した軽量な routing step です。
- 実装の中心: `MCPClient.chat()` と `MCPClientUtil.decide_route()`

#### 2. Supervisor

- 役割: route 決定後に中央の supervisor がどの tool agent を使うかを判断し、結果を最終応答へ収束させます。
- 性質: LangChain の supervisor パターンと同様に、中央の supervisor が配下エージェントの出力を受け取り、最終応答を組み立てます。
- 実装の中心: `MCPClientUtil.create_workflow()`

#### 3. Tool agents

- 役割: supervisor 配下で実際の MCP ツールを実行します。
- 現在の主な構成:
  - `tool_agent_coding`: `execute` / `status` / `get_result` など、coding-agent 系の複数ステップ実行を担当
  - `tool_agent_general`: `get_loaded_config_info` や各種 analyze 系など、一般 MCP ツールの単発実行を担当
  - `deep_agent`: DeepAgents ベースの複数ステップ調査を担当。初期実装では `execute` / `status` / `get_result` を使う非同期ジョブ系は含めません
- 実装の中心: `AgentBuilder.create_sub_agents()`

`deep_agent` を選んだ場合も、外側の実行契約は `MCPClient.chat()` 側で維持します。つまり `route_decided`、`tool_catalog_resolved`、`final_answer_validated` などの audit event は従来どおり発火し、`tool_catalog_resolved` には deep-agent 実行時に実際に公開したツール名が記録されます。

### 実行イメージ

1. ユーザー入力を受け取る
2. Router が最初の経路を選ぶ
3. Supervisor が必要な tool agent へ委譲する
4. Tool agent が MCP ツールを実行する
5. Supervisor が結果を統合し、最終応答を返す

このため、本アプリは「単一の supervisor のみ」でも「単純な router のみ」でもなく、router + supervisor + tool agents のハイブリッド構成として整理できます。

---

## Workflow 機能

WF 型 workflow は Mermaid 単体ではなく Markdown 全体を入力にします。

- Markdown には mermaid block をちょうど 1 つだけ含めてください。
- Markdown 本文と利用可能な MCP ツール説明をもとに、実行前に Mermaid 図を補正します。
- LangGraph のノードプロンプトは、補正後 Markdown とノードごとの関連ツール候補を使って生成されます。
- plan モードでは実行せず、更新済み Markdown を approval HITL として返します。

### 制約

- 1 Markdown に複数の mermaid block がある場合はエラーになります。
- 現在のノード別 MCP 実行は、要承認ツールを除外した候補ツールに限定しています。
- 将来的な複数 Mermaid 対応は未実装です。

### CLI 実行例

通常実行:

```bash
cd app
uv run ai-chat-util run_workflow -f src/ai_chat_util/workflow/samples/data/sample2.md -m "work ディレクトリを確認してください"
```

plan モード:

```bash
cd app
uv run ai-chat-util run_workflow -f src/ai_chat_util/workflow/samples/data/sample2.md -m "work ディレクトリを確認してください" --plan-mode
```

plan モードでは、CLI は既存の HITL ループを使って更新済み Markdown を表示し、`APPROVE` を受けると同じ trace_id で実行を再開します。

---

## ディレクトリ構成

実際のソースコードは `app/src` 配下で、責務ごとに複数 package に分かれています。

```
app/src/
├── ai_chat_util/        # LLMアプリ本体 + 共有層 + coding-agent runtime
│   ├── analysis/        # ファイル/文書解析の共通サービス層
│   ├── agent/           # coding-agent runtime
│   │   └── coding/      # coding-agent の API / CLI / MCP / core / test
│   ├── api/             # FastAPI APIサーバー
│   ├── cli/             # CLI
│   ├── common/          # 共通設定・共通モデル・契約
│   ├── core/            # chat / batch / tool / resource の公開 facade
│   ├── log/             # ログ設定
│   ├── mcp/             # MCPサーバー
│   ├── base/            # 内部実装層（llm / agent）
│   └── test/            # サンプル/簡易スクリプト
└── file_util/           # ファイル処理基盤
    ├── api/             # ファイル処理API
    ├── core/            # ファイル処理ツール公開関数
    ├── model/           # FileUtilDocument などのファイルモデル
    └── util/            # MIME判定、Excel、ZIP、PDFなどのユーティリティ
```

### 責務と依存ルール

依存方向は次のルールを前提に保守します。

- `file_util` は最下層のファイル処理基盤です。`ai_chat_util` の app 層へ依存しません。
- `ai_chat_util.common` は共通設定・共通モデル・契約を持つ共有層です。
- `ai_chat_util.analysis` はファイル/文書解析の共通サービス層です。CLI と API/MCP 公開層の双方から利用します。
- `ai_chat_util.core` は chat / batch / tool / resource の公開 facade です。
- `ai_chat_util.base` は内部実装層です。現在は主に `llm` と `agent` を保持します。
- `ai_chat_util.agent.coding` は coding-agent の runtime / entrypoint / test を持ちます。
- `ai_chat_util` 配下の app 層は `ai_chat_util.common` と `file_util` に依存して構いません。
- LLM 非依存のファイル処理は `file_util` に置きます。
- LLM 固有のメッセージ組み立てや chat workflow は `ai_chat_util` に置きます。

### import 方針

- package 配下に `__init__.py` の公開 export がある場合は、利用側は原則として package import を使います。
  - 例: `from ai_chat_util.base.chat import AbstractChatClient`
  - 例: `from ai_chat_util.base.batch import BatchClient`
  - 例: `from ai_chat_util.base.llm import LLMFactory`
  - 例: `from ai_chat_util.base.hitl import HITLClientBase`
- `abstract_*.py` は純粋抽象、`*_base.py` は部分実装、具体実装は機能名の module に置く方針です。
- 個別 module 直 import は、その module 自体を編集している最中か、まだ package export を用意していない場合に限ります。
- 再編中に import 経路を変える場合でも、呼び出し側はできるだけ package export 側へ寄せ、内部ファイル名変更の影響を局所化します。

依存関係のイメージ:

```
file_util
    ^
    |
ai_chat_util.common
  ^
  |
ai_chat_util.analysis / ai_chat_util.core / ai_chat_util.base / ai_chat_util.agent.coding / ai_chat_util.api / ai_chat_util.mcp / ai_chat_util.cli
```

`ai_chat_util.common -> ai_chat_util.analysis/core/base` の逆向き依存は作らないでください。

### 再編の優先順

責務整理は一度に大きく動かさず、次の順番で進めます。

1. 共通で使うが LLM 非依存な処理を `file_util` または `ai_chat_util.common` に移す
2. `ai_chat_util/base` に残るものを「公開 facade か内部実装か」で分類する
3. 公開する責務は `ai_chat_util.analysis` / `ai_chat_util.core` に寄せ、内部実装は `ai_chat_util.base` に残す

### 現在の再編状況

移動済みの共通ファイル処理:

- `file_path_resolver` は `file_util/util/file_path_resolver.py` に配置
- `downloader` は `file_util/util/downloader.py` に配置
- `office2pdf` は `file_util/util/office2pdf.py` に配置
- PDF のテキスト/画像抽出は `file_util/util/pdf_util.py` に配置
- ファイル/文書解析の共通サービスは `ai_chat_util/analysis` に配置
- chat / batch / tool / resource の公開 facade は `ai_chat_util/core` に配置
- `ai_chat_util/base` は `llm` / `agent` の内部実装層として維持
- 共有設定・共有モデルは `ai_chat_util.common` に集約
- coding-agent の API / CLI / MCP / subprocess entrypoint は `ai_chat_util.agent.coding` に集約
- console script は `coding-agent-api` / `coding-agent-mcp` / `coding-agent-util` を提供
- coding-agent の自動テストは `ai_chat_util/agent/coding/_test_` に配置

`ai_chat_util/base/util` の file util shim は削除済みです。`ai_chat_util/base/core` と `ai_chat_util/base/analysis` もトップレベルへ移動済みです。
chat / batch は `abstract_*` / `*_base` / concrete module の 3 層構成へ整理済みです。
`ai_chat_util.base.hitl` を新設し、CLI 向けの HITL 対話ループは `llm` から分離済みです。
`chat` / `batch` / `llm` / `hitl` は package `__init__.py` からの公開 import を持ち、利用側は package import を優先します。
`LLMFactory` は現状 `ai_chat_util.base.llm` に維持します。理由は、LiteLLM ベースの `LLMClient` 生成が主責務であり、`core` / `cli` / `test` から使われる内部実装ファクトリとしての性格が強いためです。
ただし `create_stdio_hitl_client()` は HITL 側責務も含むため、将来 Web/UI 向けの別 HITL 実装が増える場合は factory 分割を再検討します。

### 次の移動候補

次に見直す候補は以下です。

- `ai_chat_util/base/util` 全体
  - 最終的には「LLM 文脈を知らなくても成立する処理」を残さない方針です。
- `ai_chat_util/base/llm` と `ai_chat_util/base/agent`
  - 現在は内部実装層として維持します。公開 API として使う前提ではなく、外部からは `ai_chat_util.core` / `ai_chat_util.analysis` の利用を推奨します。

---

## インストール

```bash
uv sync
```

DeepAgents を使う場合は、追加依存を含めてインストールしてください。

```bash
cd app
uv sync --extra deepagents
```

`pip` を使う場合は、`app` ディレクトリで extras を指定します。

```bash
cd app
pip install -e ".[deepagents]"
```

`features.enable_deep_agent: true` を設定したのに `deepagents` が未導入の場合、`deep_agent route requires the deepagents package` という明示エラーになります。

明示的に DeepAgents を使う検証入口:

- CLI: `ai-chat-util run_deepagent_chat -p "..."`
- CLI(batch): `ai-chat-util run_deepagent_batch_chat -p "..." -i input.xlsx -o output.xlsx`
- CLI(batch alias): `ai-chat-util deepagent_batch_chat -p "..." -i input.xlsx -o output.xlsx`
- API: `POST /api/ai_chat_util/run_deepagent_chat`
- API(batch): `POST /api/ai_chat_util/run_deepagent_batch_chat` / `POST /api/ai_chat_util/run_deepagent_batch_chat_from_excel`
- API(batch alias): `POST /api/ai_chat_util/deepagent_batch_chat` / `POST /api/ai_chat_util/deepagent_batch_chat_from_excel`
- FastMCP tool: `run_deepagent_chat`
- FastMCP tool(batch): `run_deepagent_batch_chat` / `run_deepagent_batch_chat_from_excel`
- FastMCP tool(batch alias): `deepagent_batch_chat` / `deepagent_batch_chat_from_excel`

この入口は既存の `agent_chat` とは別で、Supervisor を経由せず DeepAgent route を強制するための実験用 API です。`trace_id`、pause/resume、audit event の外部契約は既存経路をできる限り維持します。
## 設定（ai-chat-util-config.yml + .env）

本プロジェクトは、

- **非秘密設定** → `ai-chat-util-config.yml`
- **秘密情報（APIキー/トークン等）** → 環境変数 / `.env`

に分離しています。

### 1) ai-chat-util-config.yml（必須）

同梱の `app/ai-chat-util-config.yml` をベースに、必要に応じて `ai-chat-util-config.yml` を用意してください。

```bash
cp app/ai-chat-util-config.yml ./ai-chat-util-config.yml
```

設定ファイルの探索順は以下です。

1. `--config`
2. 環境変数 `AI_CHAT_UTIL_CONFIG`
3. `./ai-chat-util-config.yml`（実行時カレント）
4. `<project-root>/ai-chat-util-config.yml`

> `ai-chat-util-config.yml` が見つからない場合はエラーになります。

実際にどの設定ファイルが読まれたか確認したい場合は、CLI の `show_config`、API エンドポイント `/api/ai_chat_util/get_loaded_config_info`、または MCP ツール `get_loaded_config_info` を使ってください。

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml show_config
```

この出力には、実際に読まれた設定ファイルのパスと、そのファイルの生の設定内容が含まれます。`os.environ/VAR_NAME` の参照は解決されないため、環境変数の実値は表示されません。

#### 秘密情報（APIキー等）の扱い

秘密情報は `ai-chat-util-config.yml` に直書きできません。

- `llm.api_key` は **環境変数参照**（`os.environ/ENV_VAR_NAME`）の形式でのみ指定できます。
- `llm.extra_headers` も秘密情報を含み得るため、**各ヘッダー値は環境変数参照**（`os.environ/ENV_VAR_NAME`）でのみ指定できます。
- `mcp.extra_headers` に `x-mcp-*` / `x-mcp-env-*` を指定すると MCP 接続に転送されます（LiteLLM には送信されません）。
- OpenAI / Azure OpenAI / Anthropic のプロバイダ（`llm.provider: openai | azure | anthropic`）を使う場合、`llm.api_key` は必須です（設定ロード時に検証されます）。

例（OpenAI）:

```yml
ai_chat_util_config:
  llm:
    provider: openai
    completion_model: gpt-5
    timeout_seconds: 60
    api_key: os.environ/LLM_API_KEY
```

例（Azure OpenAI）:

```yml
ai_chat_util_config:
  llm:
    provider: azure
    completion_model: <AZURE_DEPLOYMENT_NAME>
    base_url: https://<resource-name>.openai.azure.com/
    api_version: 2024-xx-xx
    api_key: os.environ/LLM_API_KEY
```

例（追加ヘッダーを付けたい場合）:

```yml
ai_chat_util_config:
  llm:
    provider: openai
    completion_model: gpt-5
    api_key: os.environ/LLM_API_KEY
    extra_headers:
      Authorization: os.environ/OPENAI_AUTH_HEADER
      X-My-Org: os.environ/MY_ORG_ID

    mcp:
      extra_headers:
        # MCP HTTP/SSE/WebSocket transport の headers に転送（キーの "x-mcp-" は除去されます）
        x-mcp-Authorization: os.environ/MCP_AUTH_HEADER
        # MCP stdio transport の env に転送（ENV_NAME は [A-Za-z_][A-Za-z0-9_]*）
        x-mcp-env-HTTP_PROXY: os.environ/HTTP_PROXY
```

#### DeepAgents route の有効化

`deep_agent` route は既定では無効です。使う場合は `ai-chat-util-config.yml` に非秘密設定として追加してください。

```yml
ai_chat_util_config:
  llm:
    provider: openai
    completion_model: gpt-5
    api_key: os.environ/LLM_API_KEY

  features:
    enable_deep_agent: true
    preferred_coding_route: deep_agent
```

- `enable_deep_agent: true`
  `deep_agent` route を有効化します。
- `preferred_coding_route: deep_agent`
  複雑な調査系要求で `coding_agent` より `deep_agent` を優先します。

初回実装では、`deep_agent` route は `execute` / `status` / `get_result` / `workspace_path` / `cancel` を使いません。これらが必要な非同期ジョブ系要求は `coding_agent` 側に残します。

#### file_server 設定

ファイルサーバー一覧 API / MCP ツールは `ai_chat_util_config.file_server` で制御します。

- `enabled`: 機能全体の有効化
- `allowed_roots`: API から参照可能なルート一覧
- `default_root`: `root_name` 未指定時に使うルート名
- `max_depth`, `max_entries`: 再帰列挙の上限
- `include_hidden_default`, `follow_symlinks`, `include_mime_default`: 既定の列挙挙動
- `smb`: SMB/CIFS 利用時の接続設定。秘密情報は `.env` / 環境変数参照で管理

例:

```yml
ai_chat_util_config:
  llm:
    provider: openai
    completion_model: gpt-5
    api_key: os.environ/LLM_API_KEY

  file_server:
    enabled: true
    default_root: workspace
    max_depth: 3
    max_entries: 1000
    allowed_roots:
      - name: workspace
        provider: local
        path: ./work
        description: local working directory
      - name: smb-workspace
        provider: smb
        path: ""
        description: SMB share root directory
    smb:
      enabled: true
      server: 192.168.35.89
      share: workspace
      port: 445
      domain: null
      username: os.environ/FILE_SERVER_SMB_USERNAME
      password: os.environ/FILE_SERVER_SMB_PASSWORD
```

SMB 共有直下全体を公開する場合は、`allowed_roots[].path` を空文字にします。`default_root` は既存のローカル用のままにして、SMB は `root_name=smb-workspace` を明示して利用する構成を推奨します。

注意:
- `follow_symlinks` はローカル provider 向けの設定です。SMB には適用されません。
- `include_mime_default` は SMB では実質無効です。
- `max_entries` を超えるとページングではなくエラーで打ち切ります。最初は `recursive=false` で確認してください。

HTTP API:

- `GET /api/file_util/list_file_server_roots`: 設定済みルート一覧を返す
- `GET /api/file_util/list_file_server_entries`: 指定ルート配下のディレクトリ一覧を返す

例:

```bash
curl "http://localhost:8000/api/file_util/list_file_server_roots"
curl "http://localhost:8000/api/file_util/list_file_server_entries?root_name=workspace&path=.&recursive=true&max_depth=2"
curl "http://localhost:8000/api/file_util/list_file_server_entries?root_name=smb-workspace&path=.&recursive=false"
```

MCP ツール:

- `list_file_server_roots`
- `list_file_server_entries`

#### `--config` を渡せない起動（例: `uvicorn ...:app`）

環境変数 `AI_CHAT_UTIL_CONFIG` で `ai-chat-util-config.yml` の場所を指定してください。

```powershell
$env:AI_CHAT_UTIL_CONFIG = "C:\\path\\to\\ai-chat-util-config.yml"
uvicorn ai_chat_util.api.api_server:app
```

### 2) .env（任意・秘密のみ）

`.env.example` をコピーして `.env` を作成し、利用するプロバイダの API キー等を設定してください。

SMB パスワードも `.env` で管理できます。

```dotenv
FILE_SERVER_SMB_USERNAME=user1
FILE_SERVER_SMB_PASSWORD=user1
```

```bash
copy .env.example .env
```

### Proxy環境で `certificate verify failed` が出る場合

`analyze_*_urls` / `download_files` は URL からファイルを取得します。
社内Proxyが SSL インスペクション（MITM）を行う環境では、CA を信頼できずエラーになることがあります。

推奨は **社内CAをPEMにして `ai-chat-util-config.yml` の `network.ca_bundle` で指定**することです。

```yml
ai_chat_util_config:
  network:
    ca_bundle: "C:\\path\\to\\corp-ca.pem"
    requests_verify: true
```

切り分け用途（非推奨）で SSL 検証を無効化する場合は `network.requests_verify: false` を設定してください。

### LibreOffice（Office→PDF 変換）

Office 解析で LibreOffice を使う場合は、`ai-chat-util-config.yml` の `office2pdf.libreoffice_path` を設定します。

```yml
ai_chat_util_config:
  office2pdf:
    libreoffice_path: "C:\\Program Files\\LibreOffice\\program\\soffice.exe"
```

### 環境変数（一覧）

| 変数名 | 種別 | 説明 |
|---|---|---|
| `LLM_API_KEY` | 秘密 | LLM の APIキー（OpenAI / Azure OpenAI / Anthropic など） |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | 秘密 | AWS Bedrock を使う場合の認証情報 |
| `AI_CHAT_UTIL_CONFIG` | 非秘密 | `ai-chat-util-config.yml` のパス（`--config` を渡せない起動で使用） |

### MCP設定（agent_chat / agent_batch_chat 用：任意）

CLIの `agent_chat` / `agent_batch_chat` は、内部の MCP クライアントを使って「MCPツール込みのワークフロー」で実行します。
このとき、`ai-chat-util-config.yml` の `mcp.mcp_config_path`に、MCPサーバー設定JSONのパスを指定してください。

```yml
ai_chat_util_config:
  mcp:
    mcp_config_path: mcp_settings.json
```

> 補足: 同梱の `sample_cline_mcp_settings.json` と同じ形式です。

#### MCP設定JSONで使えるキー（補足）

- `mcpServers.<name>.type` と `mcpServers.<name>.transport` はどちらでも指定できます（内部で `transport` に正規化します）。
- `allowedTools`（任意）を指定すると、内部MCPクライアント側で「使ってよいツール名」をホワイトリストできます。
- `mcpServers.<name>.env` は stdio サーバー起動時の環境変数です。秘密情報は直書きせず、`os.environ/ENV_VAR` 参照を推奨します。

例:

```json
{
  "mcpServers": {
    "AIChatUtil": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "<REPO_PATH>", "run", "-m", "ai_chat_util.mcp.mcp_server"],
      "env": {
        "LLM_API_KEY": "os.environ/LLM_API_KEY",
        "AI_CHAT_UTIL_CONFIG": "<REPO_PATH>\\ai-chat-util-config.yml"
      },
      "allowedTools": ["analyze_pdf_files", "analyze_files"]
    }
  }
}
```

> 注意: `agent_chat` / `agent_batch_chat` を使っても `mcp.mcp_config_path` が未設定の場合、MCPツールはロードされません（警告ログが出ます）。

### 内部MCPクライアントの安全弁（無限ループ/タイムアウト）

内部MCPクライアント（LangGraph Supervisor + MCPツール）は、暴走/無限ループ/長時間停止を避けるために `ai-chat-util-config.yml` の `features` で上限を設定できます。

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

### supervisor routing / audit 設定

内部MCPクライアントでは、tool 選択と結果評価を観測しやすくするために、structured routing と audit log を有効化できます。

```yml
ai_chat_util_config:
  features:
    # legacy: 従来どおり deterministic routing のみ
    # structured: routing 専用 LLM 判定を使う
    # hybrid: deterministic routing と structured routing を併用するための拡張用
    routing_mode: structured

    # structured routing の confidence がこの値未満なら clarification/HITL に倒すための閾値
    routing_confidence_threshold: 0.6

    # MCP 実行後に SufficiencyDecision を評価し、追加照会か HITL かを分類する
    sufficiency_check_enabled: true

    # trace_id 単位の構造化監査ログを JSONL で残す
    audit_log_enabled: true
    audit_log_path: ./work/structured-routing-audit.jsonl
```

補足:

- `routing_mode: legacy` の場合は従来どおり prompt ベースの supervisor routing を使います。
- `routing_mode: structured` の場合は routing 専用 LLM 呼び出しを 1 回行い、`route_decision_model_output` と `route_decided` を audit log に残します。
- `audit_log_enabled: true` にしても既存の自然文ログはそのまま残ります。比較検証には JSONL を優先してください。
- explicit coding-agent 要求で、かつプロンプトが「まず get_loaded_config_info」と順序を指定している場合は、supervisor 実行前に `get_loaded_config_info` の preflight を 1 回実行し、`preflight_applied` を audit log に残します。

### structured routing 検証手順

supervisor が「どの route を選び、どのツールを呼び、最終的に回答十分と判断したか」を確認したい場合は、structured routing 用の設定を使って `agent_chat` を実行します。

設定例:

```yml
ai_chat_util_config:
  mcp:
    mcp_config_path: /path/to/mcp_servers.local.json
    working_directory: /path/to/workspace
    coding_agent_endpoint:
      mcp_server_name: coding-agent

  features:
    routing_mode: structured
    routing_confidence_threshold: 0.6
    sufficiency_check_enabled: true
    audit_log_enabled: true
    audit_log_path: /path/to/work/structured-routing-audit.jsonl
    hitl_approval_tools: []
```

explicit coding-agent 検証例:

```bash
PROMPT='まず get_loaded_config_info を使って現在の設定ファイルの場所を確認してください。その後 coding agent を使って /path/to/doc.md を調査し、文書内で重要な見出しを 3 点挙げてください。最後に、設定ファイルの場所と見出し 3 点をまとめて回答してください。'

uv --directory ./app run -m ai_chat_util.cli \
  --config ./work/ai-chat-util-config.structured-routing-test.yml \
  --loglevel INFO \
  --logfile ./work/structured-route-explicit.log \
  agent_chat -p "$PROMPT" > ./work/structured-route-explicit.out
```

通常ツール寄りの検証例:

```bash
PROMPT='現在読み込まれている設定ファイルの場所を確認し、主要なLLM設定を簡潔に要約してください。'

uv --directory ./app run -m ai_chat_util.cli \
  --config ./work/ai-chat-util-config.structured-routing-test.yml \
  --loglevel INFO \
  --logfile ./work/structured-route-general.log \
  agent_chat -p "$PROMPT" > ./work/structured-route-general.out
```

判断系で「見出し抽出は不要」を明示する検証例:

```bash
PROMPT='現在の設定ファイルと /path/to/doc.md を踏まえて、この情報だけで本番投入判断に足りるか評価してください。足りない場合は、不足情報と追加確認事項を列挙してください。見出し抽出は不要です。'

uv --directory ./app run -m ai_chat_util.cli \
  --config ./work/ai-chat-util-config.structured-routing-test.yml \
  --loglevel INFO \
  --logfile ./work/structured-route-judgment.log \
  agent_chat -p "$PROMPT" > ./work/structured-route-judgment.out
```

期待される観測ポイント:

- `structured-route-explicit.out` には設定ファイルの場所と見出し 3 点が含まれる
- `structured-route-general.out` には設定ファイルの場所と LLM 設定要約が含まれる
- `structured-route-judgment.out` には見出し列挙ではなく、「足りている情報」「不足している情報」「追加確認項目」のような判断結果が含まれる
- `structured-routing-audit.jsonl` には少なくとも次の event が trace_id 単位で並ぶ

  - `request_received`
  - `route_decided`
  - `tool_catalog_resolved`
  - `preflight_applied`（順序指定つき explicit coding-agent ケースのみ）
  - `tool_selected`
  - `tool_result_received`
  - `sufficiency_judged`
  - `final_answer_validated`

- `routing_mode: structured` かつ deterministic rule で確定しないケースでは、追加で `route_decision_model_output` が出る
- `tool_catalog_resolved` には、その run で supervisor が前提にした tool agent 名と tool 名一覧が安定して記録される
- 通常ツールで十分な問い合わせを `general_tool_agent` に routing した場合、supervisor から見える tool agent は `tool_agent_general` のみに絞られる。`tool_catalog_resolved.route_name=general_tool_agent` と payload の `tool_agent_names` で確認できる
- 監査 JSONL を開かなくても、通常ログに `Resolved tool catalog: route=... catalog=...` が出るため、supervisor がどのツール集合を見ていたかを追跡できる
- ツール一覧確認専用のプロンプトでは workflow 実行前に `resolve_route_tool_catalog()` の結果をそのまま最終回答へ整形するため、回答本文・`tool_catalog_resolved`・通常ログの catalog が同じ集合を指すことが期待動作です
- ツール一覧確認が「名称、説明、主要な引数を一覧で示してください」のような広めの表現でも、tool catalog intent として扱い、`tool_agent_general` の `analyze_files` / `analyze_pdf_files` / `analyze_image_files` を含む route inventory を本文へ直接反映するのが期待動作です
- 判断系プロンプトで「見出し抽出は不要」を明示した場合は、`route_decided=general_tool_agent` を維持したまま最終回答が評価文になり、`final_answer_validated.payload.evidence_summary.successful_tools` に `heading_extraction` が入らないのが期待動作です

explicit coding-agent のケースでは、`route_decided.reason_code=route.explicit_coding_agent_request` が残りつつ、順序指定がある場合は `tool_agent_general` の `get_loaded_config_info` が先に 1 回だけ実行され、`preflight_applied` に確定した config path が記録されます。その後に `tool_agent_coding` の `execute/status/get_result` が続くのが期待動作です。

判断系のケースでは、`tool_catalog_resolved` で supervisor が見ていた利用可能ツール一覧を確認したうえで、`tool_selected(get_loaded_config_info)` と `tool_selected(analyze_files)` が 1 回ずつ成功し、最終回答がその証拠を踏まえた評価文に収束するのが基準です。`見出し抽出は不要` を含むにもかかわらず見出し列挙へフォールバックした場合は、intent 判定または evidence fallback の回帰を疑ってください。

通常ツールのケースでは、`tool_catalog_resolved` や `Resolved tool catalog` の内容に `tool_agent_coding` や `execute/status/get_result` が含まれていないことも確認してください。ここに coding-agent 側が露出している場合は、通常問い合わせが `execute` に流れる退行の兆候です。

監査ログ上の stable error code:

- `error=invalid_followup_task_id`: 既に無効と判定された task_id への followup。再追跡せず、取得済み結果で収束するべきケース。
- `error=stale_followup_task_id`: 最新ではない execute task_id への followup。古い task_id を追わず、最新 task_id または取得済み結果で収束するべきケース。
- `error=execute_request_invalid`: execute の入力不備。引数修正は 1 回までに留め、同じ意図の execute を繰り返さない。
- `error=tool_timeout`: 一時的なタイムアウト。再試行は 1 回までを目安にし、それ以上は収束する。
- `error=execute_backend_error`: execute バックエンド側の一時失敗。再試行や planner 補助は 1 回まで。
- `error=execute_invocation_failed`: execute の恒常的失敗。無限再試行せず、既存証拠で回答する。
- `error=tool_request_invalid` / `error=tool_backend_error` / `error=tool_invocation_failed`: 一般ツール側の入力不備・一時失敗・恒常失敗を区別するためのコード。

実運用メモ:

- OpenAI / LiteLLM 側の quota 超過時は supervisor 本体が途中停止しても、preflight までの audit event は残ります。
- quota 超過時は `structured-route-explicit.out` がエラー終了でも、`structured-routing-audit.jsonl` の先頭で `route_decided` → `tool_selected(get_loaded_config_info)` → `tool_result_received` → `preflight_applied` の順序を確認できます。

共有すると比較しやすい情報:

- 入力プロンプト
- 利用可能ツール一覧
- `route_decided` と `route_decision_model_output` の payload
- `tool_catalog_resolved` の payload
- 実際に実行された `tool_selected` / `tool_result_received`
- `sufficiency_judged` と `final_answer_validated`
- 最終回答 (`*.out`)

---

## HITL（Human-in-the-Loop：一時停止/再開）

`ai_chat_util` は、`agent_chat` または `agent_batch_chat` のように内部MCPクライアント（LangGraph Supervisor + MCPツール）経由で実行する場合に限り、
**人間への確認が必要になったタイミングで処理を一時停止（pause）し、回答/承認後に同じスレッドで再開（resume）**できます。

### 前提（重要）

- HITL（pause/resume）が発生するのは **`agent_chat` / `agent_batch_chat`（内部MCPクライアント）を使う場合のみ**です。
  - `chat` / `batch_chat`（LiteLLM経由の直接呼び出し）では、HITLは発生しません。
- `run_simple_chat` / `run_simple_batch_chat` は常に LiteLLM 経由の直接呼び出し（MCP非対応）で実行されます。
- `agent_chat` / `agent_batch_chat` を使う場合は、`ai-chat-util-config.yml` の `mcp.mcp_config_path`が必要です。
- CLI の `agent_chat` は、同一プロセス内での対話として pause/resume を処理します（プロセスを跨いだ再開のための `trace_id` 指定オプションは現状ありません）。
  - プロセスを跨いで再開したい場合は、API/ライブラリ利用で `ChatRequest.trace_id` を指定してください。

### レスポンス形（API/ライブラリ利用時）

内部MCPクライアントが「人間の判断が必要」と判断した場合、`ChatResponse` が以下の形で返ります。

- `status: "paused"`
- `trace_id`: 再開に必要（LangGraph checkpointのキーとしても使用）
- `hitl`: 人間向けのプロンプト
  - `kind: "input"`（質問への回答が必要）または `"approval"`（承認/却下が必要）
  - `prompt`: 表示すべき質問/承認内容

再開は、**同じ `trace_id` を付けて次の `ChatRequest` を送る**だけです。

### API エンドポイント

FastAPI サーバーでは、CLI の `chat` / `agent_chat` 相当機能を次のエンドポイントで公開しています。

- `POST /api/ai_chat_util/chat`
  - Body: `ChatRequest`
- `POST /api/ai_chat_util/agent_chat`
  - Body: `ChatRequest`

#### 初回リクエスト例

```json
{
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "auto_approve": false,
  "chat_history": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "params": {
              "type": "text",
              "text": "このリポジトリの API サーバー構成を説明してください。"
            }
          }
        ]
      }
    ]
  }
}
```

#### paused 後の再開例

`status="paused"` で `hitl.prompt` が返った場合は、同じ `trace_id` を使い、人間の回答を `chat_history.messages` に追加して再度 `POST /api/ai_chat_util/agent_chat` を呼び出します。

```json
{
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "auto_approve": false,
  "chat_history": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "params": {
              "type": "text",
              "text": "このリポジトリの API サーバー構成を説明してください。"
            }
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "params": {
              "type": "text",
              "text": "はい、その前提で進めてください。"
            }
          }
        ]
      }
    ]
  }
}
```

`POST /api/ai_chat_util/agent_chat` の場合にのみ内部MCPクライアント経由の pause/resume が有効になります。`POST /api/ai_chat_util/chat` では通常の completed 応答のみを返します。

#### 承認 pause/resume のテスト手順

`approval` による pause を再現するには、少なくとも次の前提が必要です。

- `ai_chat_util_config.mcp.mcp_config_path` に有効な `mcp.json` を設定している
- `ai_chat_util_config.features.hitl_approval_tools` に対象ツール名を設定している
- `POST /api/ai_chat_util/agent_chat` を呼び出す

たとえば `hitl_approval_tools: ["analyze_files"]` を設定し、`analyze_files` を使わせる入力を送ると、ツール実行前に `status="paused"` が返ります。

初回リクエスト例:

```json
{
  "auto_approve": false,
  "chat_history": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "params": {
              "type": "text",
              "text": "Use the analyze_files tool to inspect /path/to/file.txt and summarize the file in Japanese."
            }
          }
        ]
      }
    ]
  }
}
```

レスポンス例:

```json
{
  "status": "paused",
  "trace_id": "9e6ca688a7174864a410fff76aa3de51",
  "hitl": {
    "kind": "approval",
    "prompt": "analyze_files ツールの実行承認をお願いします。",
    "action_id": "...",
    "source": "supervisor:analyze_files"
  },
  "messages": [
    {
      "role": "assistant",
      "content": [
        {
          "params": {
            "type": "text",
            "text": "analyze_files ツールの実行承認をお願いします。"
          }
        }
      ]
    }
  ]
}
```

再開時は、返ってきた `trace_id` をそのまま使い、承認メッセージを追加して再送します。

```json
{
  "trace_id": "9e6ca688a7174864a410fff76aa3de51",
  "auto_approve": false,
  "chat_history": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "params": {
              "type": "text",
              "text": "Use the analyze_files tool to inspect /path/to/file.txt and summarize the file in Japanese."
            }
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "params": {
              "type": "text",
              "text": "承認します。続けてください。"
            }
          }
        ]
      }
    ]
  }
}
```

この再開リクエストが成功すると、同じ `trace_id` のまま `status="completed"` が返り、承認待ちだったツール実行後の最終回答が含まれます。

### SQLiteチェックポイント（状態保存）

内部MCPクライアントは SQLite にチェックポイントを保存します。

- 既定パス: `(<working_directory または ai-chat-util-config.yml のあるディレクトリ>)/.ai_chat_util/langgraph_checkpoints.sqlite`
- `trace_id` が同一であれば、プロセスが変わっても（同じDBを参照できる限り）再開できます。

### trace_id（BFF相関ID）運用

`ChatRequest.trace_id` は **W3C trace-id 部分（32桁hex）**を想定しています。

- 例: `4bf92f3577b34da6a3ce929d0e0e4736`
- 誤って `traceparent` 全文（`00-<trace_id>-<span_id>-<flags>`）を渡しても、trace-id部分へ正規化します。
- `trace_id == 000...0`（全ゼロ）は不正として拒否します。
運用としては、BFFが発行した `trace_id` をそのまま会話キーとして渡す（pause/resume でも同じ値を使う）と扱いやすいです。

### 承認（approval）対象ツールの設定

`ai-chat-util-config.yml` の `features.hitl_approval_tools` に、**実行前に人間の承認を求めたいツール名**を列挙できます。

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
uv --directory ./app run -m ai_chat_util.cli --help
```

また、`console_scripts` のエントリポイントとして `ai-chat-util` も提供しています。

```bash
uv run ai-chat-util --help
```

> 補足: 起動時に `.env` を読み込みます（秘密情報のみ）。`ai-chat-util-config.yml` は必須です。

### 共通オプション

```text
--loglevel  ログレベルを上書き（例: DEBUG, INFO）
--logfile   ログのファイル出力先を上書き
--config    設定ファイル(ai-chat-util-config.yml)のパス
```

### サブコマンド

#### chat（テキストチャット）

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml chat -p "こんにちは"
```

#### agent_chat（MCP テキストチャット）

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml agent_chat -p "こんにちは"
```

HITL（pause/resume）が発生した場合:

- CLIは自動で `status="paused"` を検知し、質問/承認内容を表示して入力待ちに入ります。
- 入力後、同じ `trace_id` で自動的に再開します。

#### batch_chat（Excel入力のバッチチャット）

Excel の各行（`content` / `file_path`）を読み込み、指定した `prompt` を前置して LLM に送信し、
応答を `output` 列（既定）に書き込んだ Excel を出力します。

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml batch_chat \
  -i data/input.xlsx \
  -p "要約してください" \
  -o output.xlsx
```

#### agent_batch_chat（MCP の Excel 入力バッチチャット）

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml agent_batch_chat \
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
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml analyze_image_files \
  -i a.png b.jpg \
  -p "内容を説明して" \
  --detail auto
```

#### analyze_pdf_files（PDF解析）

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml analyze_pdf_files \
  -i document.pdf \
  -p "このPDFの要約を作成して" \
  --detail auto
```

#### analyze_office_files（Office解析：PDF化→解析）

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml analyze_office_files \
  -i data.xlsx slide.pptx \
  -p "内容を要約して" \
  --detail auto
```

#### analyze_files（複数形式まとめて解析）

```bash
uv --directory ./app run -m ai_chat_util.cli --config ./ai-chat-util-config.yml analyze_files \
  -i note.txt a.png document.pdf data.xlsx \
  -p "これらをまとめて要約して" \
  --detail auto
```

---

## MCPサーバー

`ai_chat_util` は MCP（Model Context Protocol）サーバーを提供します。
MCPクライアント（例: Cline / 独自エージェント）から接続することで、チャット・画像解析・PDF解析・Office解析などのツールを利用できます。

> 補足: MCPサーバー起動時に `.env` を読み込みます（`python-dotenv` / `load_dotenv()`）。
> そのため、事前に `.env` に `LLM_API_KEY`（秘密情報）を設定し、`ai-chat-util-config.yml` の `llm.api_key` で参照してください。`ai-chat-util-config.yml` は必須です。

### 起動方法

#### stdio（デフォルト）

標準入出力（stdio）で起動します。MCPクライアントがサブプロセスとして起動して接続する用途を想定しています。

```bash
uv --directory ./app run -m ai_chat_util.mcp.mcp_server
# または明示
uv --directory ./app run -m ai_chat_util.mcp.mcp_server -m stdio
```

stdio モードでは、ログが stdout に混ざるとクライアント側のパースを壊すことがあるため、必要に応じて `--log_file` を指定してください。

```bash
uv --directory ./app run -m ai_chat_util.mcp.mcp_server -m stdio --log_file mcp_server.log -v INFO
```

#### SSE

SSE（Server-Sent Events）で起動します。

```bash
uv --directory ./app run -m ai_chat_util.mcp.mcp_server -m sse --host 0.0.0.0 -p 5001
```

#### Streamable HTTP

```bash
uv --directory ./app run -m ai_chat_util.mcp.mcp_server -m http --host 0.0.0.0 -p 5001
```

### 提供ツールの指定（任意）

`-t/--tools` で、登録するツールをカンマ区切りで指定できます。
未指定の場合は、以下の解析系ツール（files/urls）がデフォルトで登録されます。

- `analyze_image_files` / `analyze_pdf_files` / `analyze_office_files` / `analyze_files` / `analyze_documents_data`
- `analyze_image_urls` / `analyze_pdf_urls` / `analyze_office_urls` / `analyze_urls`

```bash
uv --directory ./app run -m ai_chat_util.mcp.mcp_server -m stdio -t "run_chat,analyze_pdf_files"
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
        "LLM_API_KEY": "sk-****",
        "AI_CHAT_UTIL_CONFIG": "<REPO_PATH>\\ai-chat-util-config.yml"
      }
    }
  }
}
```

---

## APIサーバー（FastAPI）

FastAPI のAPIサーバーを提供します。起動時に `ai-chat-util-config.yml` が必須です。

### 起動方法

`uvicorn ...:app` のように `--config` を渡せない起動では、環境変数 `AI_CHAT_UTIL_CONFIG` を使って `ai-chat-util-config.yml` の場所を指定してください。

```powershell
$env:AI_CHAT_UTIL_CONFIG = "C:\\path\\to\\ai-chat-util-config.yml"
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

---

## coding-agent-util（コーディングエージェント実行ユーティリティ）

本プロジェクト（ai-chat-util）には、`src/ai_chat_util/agent/coding/` 配下に **coding-agent-util（コーディングエージェント実行ユーティリティ）** が統合されています。

ローカルまたは Docker 上で「コーディングエージェント実行タスク」を起動し、進捗確認（status）やキャンセル（cancel）を行うためのユーティリティです。

- **HTTP API**（FastAPI）: `/execute` でタスク起動、`/status/{task_id}` でログ/結果取得
- **MCP サーバ**（fastmcp）: `execute/status/cancel/healthz` 等のツールを提供
- **CLI**（Typer）: タスクの起動・一覧・状態確認・キャンセル

正規の起動コマンドは次の 3 つです。

- `coding-agent-api`
- `coding-agent-mcp`
- `coding-agent-util`

本機能では **非秘匿の設定を `ai-chat-util-config.yml` に統合（ルート直下の `coding_agent_util:`）**し、**秘匿情報（API key 等）は `.env` / 環境変数**で供給する方針です。

Docker 系の実行パターンは 2 系統あります。

- [docker/coding-agent/images/all-in-one-image/README.md](docker/coding-agent/images/all-in-one-image/README.md): MCP と backend を同じ all-in-one コンテナ内で扱う構成
- [docker/coding-agent/images/dood/README.md](docker/coding-agent/images/dood/README.md): MCP は bundle コンテナ、backend は docker.sock 経由で host daemon に起動する DooD 構成

---

## 前提

- Python: `>=3.11, <=3.13`
- 依存管理: `uv` 推奨（`pyproject.toml`/`uv.lock`）
- エージェント実行コマンド: `process.command` / `compose.command` で任意指定（例: opencode / codex / claude など）

---

## セットアップ

```bash
uv sync
```

（例として）`opencode` を使う場合は、コマンドが実行できることを確認してください。

```bash
opencode --version
```

---

## 設定（coding-agent-util: を ai-chat-util-config.yml に統合）

### 設定ファイルの探索・指定方法

設定ファイルの解決順は以下です（推奨: `ai-chat-util-config.yml` に統合）。

1. CLI/サーバ起動引数 `--config`
2. 環境変数 `AI_CHAT_UTIL_CONFIG`（ai-chat-util-config.yml。ルート直下の `coding_agent_util:` セクションを読む）
3. カレントディレクトリの `./ai-chat-util-config.yml`
4. プロジェクトルート（`pyproject.toml` があるディレクトリ）の `./ai-chat-util-config.yml`

### 秘密情報（Secrets）の方針

- **`ai-chat-util-config.yml` に API key などの秘密情報を直書きしないでください**
- `.env` または環境変数で供給してください
- どうしても YAML から参照したい場合は、次の形式のみ許可します

```yml
ai_chat_util_config:
  llm:
    api_key: os.environ/LLM_API_KEY
```

実際の値はプロセス起動時に `.env`/環境変数から解決されます。

### よく使う設定項目

- `ai_chat_util_config.llm.provider`, `ai_chat_util_config.llm.completion_model`, `ai_chat_util_config.llm.base_url`: LLM の非秘匿設定
- `paths.workspace_root`: ワークスペースのデフォルト作成先
- `paths.host_projects_root`: タスクDB（`tasks_db.json`）の保存先ルート
- `ai_chat_util_config.mcp.coding_agent_endpoint.mcp_server_name`: `mcp.json` の `mcpServers.<name>` の `<name>`（coding-agent を識別するためのサーバー名）
- `backend.task_backend`: API/MCP が使う実行バックエンド（`process/windows_process/linux_process/docker/compose`）
- `compose.*`: Docker/Compose 実行時の設定
- `process.command`: process 実行時のコマンド（必須）

---

## Secrets（.env / 環境変数）

最低限、`openai` 等の provider を使う場合は API キー（例: `LLM_API_KEY`）が必要です（実行開始時にチェックされます）。

`.env`（例）:

```dotenv
LLM_API_KEY=xxxxxxxx
```

`.env.example` も `LLM_API_KEY` を基準にしています。OpenAI / Azure OpenAI / Anthropic のいずれでも、まず `LLM_API_KEY` を設定し、プロバイダ固有の model / base_url / api_version は `ai-chat-util-config.yml` 側で設定してください。

---

## 使い方

### 1) API サーバ（HTTP）

起動:

```bash
uv --directory ./app run coding-agent-api --config ./ai-chat-util-config.yml --host 127.0.0.1 -p 7101
```

`--config` で明示する場合:

```bash
uv --directory ./app run coding-agent-api --config ./ai-chat-util-config.yml --host 127.0.0.1 -p 7101
```

実行（非同期）:

```bash
curl -sS -X POST http://127.0.0.1:7101/execute \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Hello. Please respond with a single word and exit.",
    "workspace_path": "/tmp/coding_agent_tasks/demo",
    "timeout": 300
  }'
```

状態確認:

```bash
curl -sS http://127.0.0.1:7101/status/<task_id>?tail=200
```

キャンセル:

```bash
curl -sS -X DELETE http://127.0.0.1:7101/cancel/<task_id>
```

注意:

- `workspace_path` は **絶対パス必須**です（存在しない場合は作成されます）
- `paths.executor_allowed_workspace_root` を設定すると、受け付ける workspace のルートを制限できます

補足（重要）:

- `workspace_path` は **Executor（この API/MCP サーバと同じ実行環境）から見える絶対パス**である必要があります。
  Supervisor と Executor が別ホスト/別コンテナの場合、Supervisor 側で「存在する」パスでも Executor からは見えないことがあります。
- その場合の対処は次のいずれかです。
  - Supervisor が渡す `workspace_path` を Executor 視点のパスにする（推奨）
  - Executor 側に同じパスで workspace ルートを bind mount する
  - `paths.workspace_path_rewrites` でパス変換を設定する（例: `/srv/ai_platform/workspaces` → `/workspaces`）

#### rewrite を使わない（推奨運用）

`paths.workspace_path_rewrites` を使わずに運用したい場合は、次の条件を満たすように構成してください。

- `coding_agent_util.paths.workspace_root` を 1つの絶対パス（workspace のルート）として固定する
- `coding_agent_util.paths.executor_allowed_workspace_root` を同じ値に設定し、`workspace_path` をその配下に限定する
- MCPサーバを **コンテナで動かす** 場合でも、ホストの workspace_root を **同じ絶対パス** で bind mount する
  - 例: ホスト `/srv/ai_platform/workspaces` を、MCPサーバコンテナ内にも `/srv/ai_platform/workspaces` としてマウント

この構成にすると、クライアントは常に `workspace_path=<workspace_root>/<id>` を送ればよくなります。
（逆に、MCPサーバがホストで動いているのに `workspace_path=/workspace/...` のような“コンテナ内パス”を送ると、ホスト上の `/workspace` へ mkdir しようとして失敗します。）

### 2) MCP サーバ

起動（stdio）:

```bash
uv --directory ./app run coding-agent-mcp --config ./ai-chat-util-config.yml --mode stdio
```

起動（HTTP）:

```bash
uv --directory ./app run coding-agent-mcp --config ./ai-chat-util-config.yml --mode http --host 127.0.0.1 -p 7102
```

> 注意: MCPサーバ（http/sse）の既定ポートは 7101 です。APIサーバ（既定 7101）と同時に動かす場合は、上記例のように `-p 7102` 等を指定してください。

`--config` を使う場合:

```bash
uv --directory ./app run coding-agent-mcp --config ./ai-chat-util-config.yml --mode http --host 127.0.0.1 -p 7102
```

公開されるツール（代表）:

- `healthz`
- `execute`
- `status`
- `cancel`
- `workspace_path`
- `get_result`

#### MCP/HTTP 契約（互換性のための固定仕様）

Supervisor 等のクライアントから呼び出す際は、以下を **契約** として扱う想定です。

**MCP (fastmcp / streamable-http)**

- エンドポイント: `http(s)://<host>:<port>/mcp`
- ツール名（安定）: `healthz`, `execute`, `status`, `cancel`, `workspace_path`, `get_result`
- 引数形:
  - `healthz`: 引数なし
  - `execute`: 引数は **1つ** で、名前は `req`
    - `req.prompt: str`
    - `req.workspace_path: str`（絶対パス必須）
    - `req.timeout: int`
    - `req.task_id?: str`
    - `req.trace_id?: str`
  - `status`: `{ "task_id": "...", "tail"?: 200 | null }`
  - `workspace_path`: `{ "task_id": "..." }`
  - `get_result`: `{ "task_id": "...", "tail"?: 200 | null }` → `{ "stdout": "...", "stderr": "..." }`
  - `cancel`: `{ "task_id": "..." }`

**HTTP API (FastAPI)**

- `GET /healthz` → `{ "status": "ok" }`
- `POST /execute` → `{ "task_id": "..." }`（既定は非同期）
- `GET /status/{task_id}?tail=200` → `TaskStatus`
- `DELETE /cancel/{task_id}` → `CancelResponse`

**TaskStatus（代表フィールド）**

- `task_id: str`
- `workspace_path: str`
- `trace_id?: str`（相関ID。ヘッダ/req から伝播されます）
- `status?: "pending" | "running" | "exited"`
- `sub_status?: "not-started" | "starting" | "running-foreground" | "running-background" | "failed" | "timeout" | "cancelled" | "completed"`
- `stdout?: str`, `stderr?: str`（`tail` パラメータで末尾のみ返る場合があります）
- `artifacts?: list[str]`（バックエンドにより埋まる場合があります）
- `metadata: object`（拡張メタ情報。互換性のため dict を維持）

**ヘッダ（任意）**

- `Authorization`: 下流の実行バックエンドへ環境変数として伝播されます
- `X-Trace-Id`（互換: `trace-id`, `trace_id`）: `ExecuteRequest.trace_id` が未指定の場合に補完されます

> 伝播される環境変数名（代表）: `AI_PLATFORM_AUTHORIZATION`, `AUTHORIZATION`, `AI_PLATFORM_TRACE_ID`, `TRACE_ID`

### 3) CLI

現状の CLI は `DockerTaskService` を利用しており、**Docker/Compose 実行を前提**にしています。
（ローカル process で試したい場合は API/MCP 経由の方が簡単です）

ヘルプ:

```bash
uv --directory ./app run coding-agent-util --help
```

実行:

```bash
uv --directory ./app run coding-agent-util --config ./ai-chat-util-config.yml run \
  "Hello. Please respond with a single word and exit." \
  --wait
```

非同期（task_id を返して終了）:

```bash
uv --directory ./app run coding-agent-util --config ./ai-chat-util-config.yml run \
  "Hello" \
  --no-wait
```

状態確認:

```bash
uv --directory ./app run coding-agent-util --config ./ai-chat-util-config.yml status <task_id>
```

---

## トラブルシュート

### `LLM API key が未設定` で失敗する

- `.env` または環境変数で API キー（例: `LLM_API_KEY`）を設定してください
- `ai-chat-util-config.yml` に `ai_chat_util_config.llm.api_key` を書く場合は、必ず `os.environ/LLM_API_KEY` 形式で参照してください

### 設定ファイルが見つからない

- `--config` を指定する
- もしくは `AI_CHAT_UTIL_CONFIG` を設定する
- もしくは `./ai-chat-util-config.yml` をカレント/プロジェクトルートに置く

### `workspace_path must be an absolute path`

- API/MCP の `execute` には絶対パスを渡してください（例: `/tmp/coding_agent_tasks/demo`）

---

## 開発メモ

- 設定ローダは `src/ai_chat_util/common/config/runtime.py` にあります（ai-chat-util / coding-agent-util を同梱）
- 設定は `ai-chat-util-config.yml` へ統合可能で、環境変数は secrets（`.env` 含む）の供給にのみ使う設計です
