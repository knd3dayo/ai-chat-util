from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ContainerInfo(BaseModel):
    """Docker コンテナの情報を表すモデル。"""

    id: str = Field(description="コンテナ ID（フル形式）")
    short_id: str = Field(description="コンテナ ID（短縮形式、12文字）")
    name: str = Field(description="コンテナ名")
    status: str = Field(description="コンテナの状態（running / exited / created 等）")
    image: str = Field(description="使用しているイメージ名")
    labels: dict[str, str] = Field(default_factory=dict, description="コンテナに付与されたラベルの辞書")
    created_at: Optional[datetime] = Field(default=None, description="コンテナの作成日時")


class ImageInfo(BaseModel):
    """Docker イメージの情報を表すモデル。"""

    id: str = Field(description="イメージ ID（フル形式）")
    short_id: str = Field(description="イメージ ID（短縮形式、12文字）")
    repo_tags: list[str] = Field(default_factory=list, description="イメージのリポジトリ:タグ一覧")
    size: Optional[int] = Field(default=None, description="イメージサイズ（bytes）")
    created_at: Optional[datetime] = Field(default=None, description="イメージの作成日時")


class ComposeOperationResult(BaseModel):
    """docker-compose 操作の結果を表すモデル。"""

    success: bool = Field(description="操作が正常に完了したかどうか")
    project_name: str = Field(description="docker-compose プロジェクト名")
    output: str = Field(default="", description="操作の標準出力")
    error: str = Field(default="", description="操作の標準エラー出力またはエラーメッセージ")
    container_names: list[str] = Field(default_factory=list, description="関連するコンテナ名の一覧")
    image_names: list[str] = Field(default_factory=list, description="関連するイメージ名の一覧")


class DockerGenerateResult(BaseModel):
    """AI による Dockerfile / docker-compose.yml 生成結果を表すモデル。"""

    content: str = Field(description="生成されたファイルの内容（Dockerfile または docker-compose.yml）")
    explanation: str = Field(default="", description="生成内容の説明・補足")
