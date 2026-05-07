from __future__ import annotations

import re
import time
from textwrap import dedent

from ai_chat_util.core.chat import create_llm_client
from ai_chat_util.core.docker.model import DockerGenerateResult
import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

_DOCKERFILE_SYSTEM_PROMPT = dedent("""\
    あなたは Docker の専門家です。
    ユーザーの指示に基づき、セキュアで実用的な Dockerfile を作成してください。
    以下のルールに従ってください:
    - FROM, RUN, COPY, CMD/ENTRYPOINT など必要な命令を含める
    - 不要なパッケージはインストールしない（最小構成を心がける）
    - root ユーザーで実行しないよう USER 命令を入れる（可能な場合）
    - .dockerignore の記述が推奨される場合は説明に含める
    - レスポンスは以下の形式で返すこと:

    <DOCKERFILE>
    （ここに Dockerfile の内容）
    </DOCKERFILE>
    <EXPLANATION>
    （ここに生成内容の説明・補足）
    </EXPLANATION>
""")

_COMPOSE_SYSTEM_PROMPT = dedent("""\
    あなたは Docker Compose の専門家です。
    ユーザーの指示に基づき、実用的な docker-compose.yml を作成してください。
    以下のルールに従ってください:
    - services, networks, volumes を適切に定義する
    - 環境変数はハードコードせず ${VARIABLE} 形式で記述する（機密情報の場合）
    - depends_on, healthcheck などを適切に設定する
    - レスポンスは以下の形式で返すこと:

    <COMPOSE>
    （ここに docker-compose.yml の内容）
    </COMPOSE>
    <EXPLANATION>
    （ここに生成内容の説明・補足）
    </EXPLANATION>
""")


def _extract_tagged(text: str, tag: str) -> tuple[str, str]:
    """<TAG>...</TAG> 形式のブロックと説明を抽出する。"""
    pattern = rf"<{tag}>\s*([\s\S]*?)\s*</{tag}>"
    explanation_pattern = r"<EXPLANATION>\s*([\s\S]*?)\s*</EXPLANATION>"

    content_match = re.search(pattern, text, re.IGNORECASE)
    explanation_match = re.search(explanation_pattern, text, re.IGNORECASE)

    content = content_match.group(1).strip() if content_match else text.strip()
    explanation = explanation_match.group(1).strip() if explanation_match else ""
    return content, explanation


class DockerGenUtil:
    """生成 AI を使用して Dockerfile / docker-compose.yml を生成するユーティリティクラス。"""

    @classmethod
    async def generate_dockerfile(
        cls,
        instructions: str,
        *,
        base_image: str | None = None,
        language: str | None = None,
        additional_requirements: str | None = None,
    ) -> DockerGenerateResult:
        """指示から Dockerfile を生成します。

        Args:
            instructions: Dockerfile に対する自然言語の指示。
            base_image: ベースイメージの指定（例: "python:3.12-slim"）。省略時は AI が判断。
            language: 言語/フレームワークのヒント（例: "Python/FastAPI"）。
            additional_requirements: 追加要件の説明（例: "GPU 対応、CUDA 12.0 必要"）。

        Returns:
            DockerGenerateResult
        """
        started = time.perf_counter()
        prompt_parts = [f"以下の指示に基づいて Dockerfile を作成してください:\n\n{instructions}"]
        if base_image:
            prompt_parts.append(f"\nベースイメージ: {base_image}")
        if language:
            prompt_parts.append(f"\n言語/フレームワーク: {language}")
        if additional_requirements:
            prompt_parts.append(f"\n追加要件: {additional_requirements}")

        full_prompt = "\n".join(prompt_parts)
        logger.info("DOCKER_GEN_START type=dockerfile instructions_len=%d", len(instructions))

        # システムプロンプトをユーザープロンプトの前に置く（simple_chat はユーザーメッセージのみ）
        combined = f"{_DOCKERFILE_SYSTEM_PROMPT}\n\n---\n\n{full_prompt}"
        llm = create_llm_client()
        raw = await llm.simple_chat(combined)

        content, explanation = _extract_tagged(raw, "DOCKERFILE")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("DOCKER_GEN_END type=dockerfile elapsed_ms=%d content_len=%d", elapsed_ms, len(content))
        return DockerGenerateResult(content=content, explanation=explanation)

    @classmethod
    async def generate_compose(
        cls,
        instructions: str,
        *,
        environment_description: str | None = None,
    ) -> DockerGenerateResult:
        """指示から docker-compose.yml を生成します。

        Args:
            instructions: docker-compose.yml に対する自然言語の指示。
            environment_description: 環境の説明（例: "本番環境: nginx + FastAPI + PostgreSQL + Redis"）。

        Returns:
            DockerGenerateResult
        """
        started = time.perf_counter()
        prompt_parts = [f"以下の指示に基づいて docker-compose.yml を作成してください:\n\n{instructions}"]
        if environment_description:
            prompt_parts.append(f"\n環境の説明: {environment_description}")

        full_prompt = "\n".join(prompt_parts)
        logger.info("DOCKER_GEN_START type=compose instructions_len=%d", len(instructions))

        combined = f"{_COMPOSE_SYSTEM_PROMPT}\n\n---\n\n{full_prompt}"
        llm = create_llm_client()
        raw = await llm.simple_chat(combined)

        content, explanation = _extract_tagged(raw, "COMPOSE")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("DOCKER_GEN_END type=compose elapsed_ms=%d content_len=%d", elapsed_ms, len(content))
        return DockerGenerateResult(content=content, explanation=explanation)
