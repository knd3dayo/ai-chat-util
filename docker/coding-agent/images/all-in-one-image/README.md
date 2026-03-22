# all-in-one-image MCP workflow

このディレクトリには、host 側で管理する MCP プロジェクトを
`${HOME}/data/mcps` へ同期し、all-in-one コンテナから read-only bind mount で利用するためのスクリプトが含まれます。

## 前提

- Docker / `docker compose` が利用可能であること
- `all-in-one-code-executor-image` を build できること
- host 側で Python と `uv` が利用可能であること
- `ai-chat-util/app` の `.venv` を host 上で用意済みであること

## 初回セットアップ

1. image を build

```sh
cd docker/coding-agent/images/all-in-one-image/build-image
./build.sh
```

2. `${HOME}/data/mcps` を populate

```sh
cd docker/coding-agent/images/all-in-one-image
./run-mcp-server.sh prepare
```

この処理は、source repos から `${HOME}/data/mcps` へ実ディレクトリを rsync し、
各 Python プロジェクトで `uv sync --no-dev` を実行して `.venv` をコピー先のパスへ再整合します。

## 差分同期

通常の差分同期:

```sh
./run-mcp-server.sh prepare
```

変更内容だけ確認する dry-run:

```sh
./run-mcp-server.sh prepare --dry-run
```

特定プロジェクトだけ同期:

```sh
./run-mcp-server.sh prepare --project ai-chat-util
./run-mcp-server.sh prepare --project deonodo-log-util
```

依存更新が不要で、ファイル同期だけ行いたい場合:

```sh
./run-mcp-server.sh prepare --skip-uv-sync
```

## MCP サーバー起動

起動:

```sh
./run-mcp-server.sh up
```

ログ確認:

```sh
./run-mcp-server.sh logs -f
```

bash ログイン:

```sh
./run-mcp-server.sh bash
```

停止:

```sh
./run-mcp-server.sh down
```

## ディレクトリ構成

- host 側 MCP root: `${HOME}/data/mcps`
- host 側ログ: `${HOME}/data/mcp-logs/ai-chat-util`
- コンテナ内ログ: `/home/codeuser/logs/ai-chat-util-mcp.log`

既定では次のプロジェクトを同期対象とします。

- `${HOME}/data/mcps/ai-chat-util/app`
- `${HOME}/data/mcps/deonodo-log-util`
- `${HOME}/data/mcps/denodo-vql-client`

## 注意点

- `${HOME}/data/mcps` 配下は symlink ではなく実ディレクトリで運用してください。
- Python minor version が変わった場合は、host 側 `.venv` を再作成したうえで `prepare` を再実行してください。
- 起動時は `uv run --no-sync python -m ...` を使うため、`.venv/bin` の console script を直接呼ぶ運用は想定していません。