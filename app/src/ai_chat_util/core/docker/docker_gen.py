from __future__ import annotations

import time
from typing import Annotated

from pydantic import Field

from ai_chat_util.core.docker.docker_gen_util import DockerGenUtil
import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


async def docker_generate_dockerfile(
    instructions: Annotated[
        str,
        Field(description="Dockerfile に対する自然言語の指示。例: 'Python 3.12 + FastAPI アプリ、ポート 8000 で起動'"),
    ],
    base_image: Annotated[
        str | None,
        Field(description="ベースイメージの指定（例: 'python:3.12-slim'）。省略時は AI が判断します。"),
    ] = None,
    language: Annotated[
        str | None,
        Field(description="言語/フレームワークのヒント（例: 'Python/FastAPI'、'Node.js/Express'）。"),
    ] = None,
    additional_requirements: Annotated[
        str | None,
        Field(description="追加要件の説明（例: 'GPU 対応、CUDA 12.0 必要'、'非 root ユーザーで実行'）。"),
    ] = None,
) -> Annotated[str, Field(description="生成された Dockerfile の内容と説明を JSON 文字列で返します。")]:
    """
    指示に基づいて Dockerfile を AI で生成します。

    生成された Dockerfile はそのまま使用できる形式で返されます。
    必要に応じて base_image や language を指定することで、より精度の高い結果が得られます。
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=docker_generate_dockerfile instructions_len=%d",
        len(instructions or ""),
    )
    try:
        result = await DockerGenUtil.generate_dockerfile(
            instructions=instructions,
            base_image=base_image,
            language=language,
            additional_requirements=additional_requirements,
        )
        return result.model_dump_json()
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_generate_dockerfile")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_generate_dockerfile elapsed_ms=%s", elapsed_ms)


async def docker_generate_compose(
    instructions: Annotated[
        str,
        Field(
            description=(
                "docker-compose.yml に対する自然言語の指示。"
                "例: 'nginx をリバースプロキシとし、FastAPI アプリと PostgreSQL を起動する構成'"
            )
        ),
    ],
    environment_description: Annotated[
        str | None,
        Field(
            description=(
                "環境の説明（例: '本番環境: nginx + FastAPI + PostgreSQL + Redis'）。"
                "指定すると生成精度が向上します。"
            )
        ),
    ] = None,
) -> Annotated[str, Field(description="生成された docker-compose.yml の内容と説明を JSON 文字列で返します。")]:
    """
    指示に基づいて docker-compose.yml を AI で生成します。

    生成された docker-compose.yml はそのまま使用できる形式で返されます。
    environment_description を指定することで、より具体的な構成が生成されます。
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=docker_generate_compose instructions_len=%d",
        len(instructions or ""),
    )
    try:
        result = await DockerGenUtil.generate_compose(
            instructions=instructions,
            environment_description=environment_description,
        )
        return result.model_dump_json()
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_generate_compose")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_generate_compose elapsed_ms=%s", elapsed_ms)
