# syntax=docker/dockerfile:1

# ベースイメージ
FROM python:3.12-slim-bookworm AS base

WORKDIR /app

# uv をインストール（pip 経由）
RUN pip install uv

# ソースコードをコピー
COPY . /app

# venv は /app/.venv に作成される
RUN uv sync

# 5001 で Streamable HTTP を listen する
EXPOSE 5001

# デフォルトコマンド
# NOTE: ポートを変えたい場合は `-p` を追加
CMD ["uv", "run", "-m", "ai_chat_util.mcp.mcp_server", "-m", "http", "-p", "5001"]
