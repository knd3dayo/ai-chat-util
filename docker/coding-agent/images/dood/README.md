# dood MCP workflow

このディレクトリは、MCP サーバーを bundle コンテナ内で動かしつつ、
実タスクの backend は docker.sock 経由で host daemon に起動する DooD 構成です。

all-in-one-image との違いは次の通りです。

- MCP サーバー本体は [docker-compose.yml](docker-compose.yml) の bundle コンテナで起動する
- コーディング backend は [backend-compose.yml](backend-compose.yml) を bundle 内から `docker compose run` して起動する
- workspace は host と bundle コンテナで同一絶対パス `/srv/ai_platform/workspaces` を共有する

## 前提

- Docker / `docker compose` が利用可能であること
- `all-in-one-code-executor-image` が build 済みであること
- host 側に `/srv/ai_platform/workspaces` を用意できること
- LLM の秘密情報は shell 環境変数または `.env` で与えること

## 主なファイル

- [docker-compose.yml](docker-compose.yml)
  DooD bundle の compose。MCP サーバーを公開する
- [backend-compose.yml](backend-compose.yml)
  backend 実行用 compose。bundle 内の docker CLI がこれを使って task container を起動する
- [ai-chat-util-config.yml](ai-chat-util-config.yml)
  bundle 内 MCP サーバーが読む coding 設定
- [env_compose](env_compose)
  非秘匿の既定値。`LLM_API_KEY` はここには置かない

## 初回 build

```sh
cd docker/coding-agent/images/dood
./run-mcp-server.sh build
```

## 起動

秘密情報は shell 環境変数または `.env` で渡します。

```sh
cd docker/coding-agent/images/dood
export LLM_API_KEY="<your-api-key>"
./run-mcp-server.sh up
```

`LLM_API_KEY` は必須です。未設定のまま `docker compose up` すると失敗します。

`.env` を使う場合の最小例:

```dotenv
LLM_API_KEY=<your-api-key>
```

## 状態確認

```sh
./run-mcp-server.sh ps
./run-mcp-server.sh status
./run-mcp-server.sh logs -f
```

正常起動時は logs に次のような行が出ます。

- `Server: coding_agent_executor`
- `transport 'streamable-http'`
- `http://0.0.0.0:7101/mcp`

## MCP 疎通確認

`GET /mcp` は streamable-http の都合で `406 Not Acceptable` でも正常です。

```sh
curl -i http://127.0.0.1:7101/mcp
```

## execute の確認

host 側からテストクライアントで execute できます。

```sh
mkdir -p /srv/ai_platform/workspaces/e2e_dood_ws_1

cd app
.venv/bin/python -m ai_chat_util.agent.coding._test_.mcp_client \
  --url http://127.0.0.1:7101/mcp \
  --workspace-path /srv/ai_platform/workspaces/e2e_dood_ws_1 \
  --prompt "コーディングエージェントMCPツールとして、ワークスペース直下に done.txt を作成し、内容を dood-ok にしてください。最後に実施内容を短く報告してください。" \
  --wait --poll-interval 1.0 --max-polls 90 --tail 80
```

成功すると、workspace 配下に `done.txt` が生成されます。

```sh
cat /srv/ai_platform/workspaces/e2e_dood_ws_1/done.txt
```

期待値:

```text
dood-ok
```

## 停止

```sh
cd docker/coding-agent/images/dood
./run-mcp-server.sh down
```

停止済み task container も片付けたい場合:

```sh
cd docker/coding-agent/images/dood
./run-mcp-server.sh clean
```

## シェル

```sh
cd docker/coding-agent/images/dood
./run-mcp-server.sh sh
```

## 注意点

- backend container は bundle 内ではなく host daemon 上に起動する
- そのため bind mount の source path は host から見える絶対パスである必要がある
- [backend-compose.yml](backend-compose.yml) は DooD 用に最小構成へ寄せており、all-in-one-image の script bind mount は使わない