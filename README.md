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
pip install -e .
```

または、`pyproject.toml` を利用して依存関係を管理します。

---

## Docker / Docker Compose で起動（MCPサーバ）

`.env_template` を参考に `ai-chat-util/.env` を用意した上で、以下を実行してください。

```bash
cd ai-chat-util
docker compose up --build
```

起動後、MCPサーバは `http://localhost:${HOST_PORT:-5001}` で待ち受けます。

- ホスト側ポートを変更したい場合は `ai-chat-util/.env` に `HOST_PORT` を設定してください（例: `HOST_PORT=9000`）。
- コンテナ内の待受ポートは 5001 固定です。

> 補足: docker compose は同ディレクトリの `.env` を自動で読み込みますが、
> それは compose ファイル内の `${VAR}` 置換用途です。
> コンテナへ環境変数として渡すために、`docker-compose.yml` 側で `env_file: .env` を指定しています。

---

## 依存関係

主要な依存パッケージは `requirements.txt` に記載されています。  
例：
```
openai
pydantic
requests
```

---

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

## 使用例

## CLI（コマンドライン）

このリポジトリには `ai_chat_util` の簡易CLIが含まれています。

### インストール（開発モード）

`uv` を利用する前提:

```bash
uv sync
uv pip install -e .
```

### ヘルプ

```bash
ai-chat-util --help
```

### チャット

```bash
ai-chat-util chat -p "ping"
```

### 画像解析

```bash
ai-chat-util analyze_image_files -i sample.png -p "内容を説明して" --detail auto
```

### MCPサーバ起動

```bash
ai-chat-util mcp_server --mode stdio
```

### チャットクライアントの利用例

```python
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

llm_config = LLMConfig()
client = LLMClient.create_llm_client(llm_config)
response = client.simple_chat("こんにちは、今日の天気は？")
print(response)
```

### バッチ処理の利用例

```python
from ai_chat_util.batch.batch_client import BatchClient
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

llm_config = LLMConfig()
client = LLMClient.create_llm_client(llm_config)

batch = BatchClient(client)
results = batch.run(["要約して", "翻訳して", "説明して"])
for r in results:
    print(r)
```
### 画像解析の利用例（simple_image_analysis）

```python
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

llm_config = LLMConfig()
client = LLMClient.create_llm_client(llm_config)

result = client.simple_image_analysis(
    ["sample_image.jpg"],
    prompt="この画像の内容を説明してください。"
)
print(result)
```

### PDF解析の利用例（simple_pdf_analysis）

```python
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

llm_config = LLMConfig()
client = LLMClient.create_llm_client(llm_config)

result = client.simple_pdf_analysis(
    ["document.pdf"],
    prompt="このPDFの要約を作成してください。"
)
print(result)
```

### Officeドキュメント解析の利用例（simple_office_analysis）

```python
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

llm_config = LLMConfig()
client = LLMClient.create_llm_client(llm_config)

result = client.simple_office_document_analysis(
    "data.xlsx",
    prompt="これらのドキュメントの内容を要約してください。"
)
print(result)
```

---

## ライセンス

このプロジェクトは [MIT License](LICENSE) のもとで公開されています。

---

## リポジトリ

GitHub: [https://github.com/knd3dayo/ai_chat_util](https://github.com/knd3dayo/ai_chat_util)
