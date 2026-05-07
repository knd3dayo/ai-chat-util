from __future__ import annotations

import json
import time
from typing import Annotated

from pydantic import Field

from ai_chat_util.core.docker.docker_ops_util import DockerOpsUtil
import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


async def docker_compose_up(
    project_name: Annotated[str, Field(description="docker-compose プロジェクト名。コンテナの識別に使用します。")],
    compose_path: Annotated[
        str | None,
        Field(description="docker-compose.yml のファイルパス。compose_content と排他（compose_content 優先）。"),
    ] = None,
    compose_content: Annotated[
        str | None,
        Field(description="docker-compose.yml の内容（YAML 文字列）。指定時はファイルパスより優先されます。"),
    ] = None,
    project_directory: Annotated[
        str | None,
        Field(description="compose_content 内の相対 build context や volume パス解決に使う基準ディレクトリ。compose_content 利用時に指定可能。"),
    ] = None,
    env_vars: Annotated[
        str | None,
        Field(
            description=(
                'compose ファイル内の変数置換に使用する環境変数の JSON 辞書文字列。'
                '例: {"WORKSPACE": "/tmp/workspace", "PORT": "8080"}'
            )
        ),
    ] = None,
    service_names: Annotated[
        list[str] | None,
        Field(description="起動するサービス名のリスト。省略時は全サービスを起動します。"),
    ] = None,
    detach: Annotated[
        bool,
        Field(description="バックグラウンドで実行するかどうか。デフォルト True。"),
    ] = True,
    build: Annotated[
        bool,
        Field(description="起動前にイメージをビルドするかどうか。デフォルト False。"),
    ] = False,
) -> Annotated[str, Field(description="操作結果を JSON 文字列で返します。")]:
    """
    docker compose up を実行してサービスを起動します。

    compose_content（YAML 文字列）または compose_path（ファイルパス）のいずれかを指定してください。
    compose_content が指定された場合は一時ファイルに書き出して処理します。

    [MCP_META]
    requires_approval=true
    action_kind=write
    usage_guidance=コンテナを起動します。compose_content または compose_path を指定し、project_name でプロジェクトを識別してください。停止には docker_compose_down を使用してください。
    """
    started = time.perf_counter()
    logger.info("MCP_TOOL_START tool=docker_compose_up project=%s", project_name)
    env_dict: dict[str, str] | None = None
    if env_vars:
        try:
            env_dict = json.loads(env_vars)
        except Exception:
            return json.dumps({"success": False, "project_name": project_name, "error": "env_vars が不正な JSON です。"})
    try:
        result = DockerOpsUtil.compose_up(
            project_name=project_name,
            compose_path=compose_path,
            compose_content=compose_content,
            project_directory=project_directory,
            env_vars=env_dict,
            service_names=service_names,
            detach=detach,
            build=build,
        )
        return result.model_dump_json()
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_compose_up")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_compose_up elapsed_ms=%s", elapsed_ms)


async def docker_compose_down(
    project_name: Annotated[str, Field(description="停止する docker-compose プロジェクト名。")],
    compose_path: Annotated[
        str | None,
        Field(description="docker-compose.yml のファイルパス。compose_content と排他（compose_content 優先）。"),
    ] = None,
    compose_content: Annotated[
        str | None,
        Field(description="docker-compose.yml の内容（YAML 文字列）。指定時はファイルパスより優先されます。"),
    ] = None,
    project_directory: Annotated[
        str | None,
        Field(description="compose_content 内の相対パス解決に使う基準ディレクトリ。compose_content 利用時に指定可能。"),
    ] = None,
    env_vars: Annotated[
        str | None,
        Field(description='環境変数の JSON 辞書文字列。例: {"KEY": "value"}'),
    ] = None,
    remove_volumes: Annotated[
        bool,
        Field(description="ボリュームも削除するかどうか。デフォルト False。"),
    ] = False,
) -> Annotated[str, Field(description="操作結果を JSON 文字列で返します。")]:
    """
    docker compose down を実行してサービスを停止・削除します。

    [MCP_META]
    requires_approval=true
    action_kind=write
    usage_guidance=コンテナを停止して削除します。remove_volumes=true を指定するとボリュームも削除されます。
    """
    started = time.perf_counter()
    logger.info("MCP_TOOL_START tool=docker_compose_down project=%s", project_name)
    env_dict: dict[str, str] | None = None
    if env_vars:
        try:
            env_dict = json.loads(env_vars)
        except Exception:
            return json.dumps({"success": False, "project_name": project_name, "error": "env_vars が不正な JSON です。"})
    try:
        result = DockerOpsUtil.compose_down(
            project_name=project_name,
            compose_path=compose_path,
            compose_content=compose_content,
            project_directory=project_directory,
            env_vars=env_dict,
            remove_volumes=remove_volumes,
        )
        return result.model_dump_json()
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_compose_down")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_compose_down elapsed_ms=%s", elapsed_ms)


async def docker_compose_restart(
    project_name: Annotated[str, Field(description="再起動する docker-compose プロジェクト名。")],
    compose_path: Annotated[
        str | None,
        Field(description="docker-compose.yml のファイルパス。compose_content と排他（compose_content 優先）。"),
    ] = None,
    compose_content: Annotated[
        str | None,
        Field(description="docker-compose.yml の内容（YAML 文字列）。指定時はファイルパスより優先されます。"),
    ] = None,
    project_directory: Annotated[
        str | None,
        Field(description="compose_content 内の相対パス解決に使う基準ディレクトリ。compose_content 利用時に指定可能。"),
    ] = None,
    env_vars: Annotated[
        str | None,
        Field(description='環境変数の JSON 辞書文字列。例: {"KEY": "value"}'),
    ] = None,
    service_names: Annotated[
        list[str] | None,
        Field(description="再起動するサービス名のリスト。省略時は全サービスを再起動します。"),
    ] = None,
) -> Annotated[str, Field(description="操作結果を JSON 文字列で返します。")]:
    """
    docker compose restart を実行してサービスを再起動します。

    [MCP_META]
    requires_approval=true
    action_kind=write
    usage_guidance=コンテナを再起動します。特定サービスのみ再起動する場合は service_names を指定してください。
    """
    started = time.perf_counter()
    logger.info("MCP_TOOL_START tool=docker_compose_restart project=%s", project_name)
    env_dict: dict[str, str] | None = None
    if env_vars:
        try:
            env_dict = json.loads(env_vars)
        except Exception:
            return json.dumps({"success": False, "project_name": project_name, "error": "env_vars が不正な JSON です。"})
    try:
        result = DockerOpsUtil.compose_restart(
            project_name=project_name,
            compose_path=compose_path,
            compose_content=compose_content,
            project_directory=project_directory,
            env_vars=env_dict,
            service_names=service_names,
        )
        return result.model_dump_json()
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_compose_restart")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_compose_restart elapsed_ms=%s", elapsed_ms)


async def docker_compose_logs(
    project_name: Annotated[str, Field(description="ログを取得する docker-compose プロジェクト名。")],
    compose_path: Annotated[
        str | None,
        Field(description="docker-compose.yml のファイルパス。compose_content と排他（compose_content 優先）。"),
    ] = None,
    compose_content: Annotated[
        str | None,
        Field(description="docker-compose.yml の内容（YAML 文字列）。指定時はファイルパスより優先されます。"),
    ] = None,
    project_directory: Annotated[
        str | None,
        Field(description="compose_content 内の相対パス解決に使う基準ディレクトリ。compose_content 利用時に指定可能。"),
    ] = None,
    env_vars: Annotated[
        str | None,
        Field(description='環境変数の JSON 辞書文字列。例: {"KEY": "value"}'),
    ] = None,
    service_names: Annotated[
        list[str] | None,
        Field(description="ログを取得するサービス名のリスト。省略時は全サービスのログを取得します。"),
    ] = None,
    tail: Annotated[
        int,
        Field(description="取得する末尾の行数。デフォルト 200。", ge=1, le=10000),
    ] = 200,
) -> Annotated[str, Field(description="docker compose のログ文字列。")]:
    """
    docker compose logs を取得します。

    compose_content（YAML 文字列）または compose_path（ファイルパス）のいずれかを指定してください。
    """
    started = time.perf_counter()
    logger.info("MCP_TOOL_START tool=docker_compose_logs project=%s", project_name)
    env_dict: dict[str, str] | None = None
    if env_vars:
        try:
            env_dict = json.loads(env_vars)
        except Exception:
            return "ERROR: env_vars が不正な JSON です。"
    try:
        return DockerOpsUtil.compose_logs(
            project_name=project_name,
            compose_path=compose_path,
            compose_content=compose_content,
            project_directory=project_directory,
            env_vars=env_dict,
            service_names=service_names,
            tail=tail,
        )
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_compose_logs")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_compose_logs elapsed_ms=%s", elapsed_ms)


async def docker_list_containers(
    label_filter: Annotated[
        str | None,
        Field(description='ラベルフィルター（"key=value" 形式）。例: "managed_by=myapp"。省略時はフィルターなし。'),
    ] = None,
    name_filter: Annotated[
        str | None,
        Field(description="コンテナ名の部分一致フィルター。省略時はフィルターなし。"),
    ] = None,
    show_all: Annotated[
        bool,
        Field(description="停止中のコンテナも含めるかどうか。デフォルト True。"),
    ] = True,
) -> Annotated[str, Field(description="コンテナ情報の JSON 配列文字列。")]:
    """
    Docker コンテナの一覧を取得します。

    label_filter または name_filter でコンテナを絞り込むことができます。
    show_all=False にすると実行中のコンテナのみ返します。
    """
    started = time.perf_counter()
    logger.info("MCP_TOOL_START tool=docker_list_containers label=%s name=%s", label_filter, name_filter)
    try:
        containers = DockerOpsUtil.list_containers(
            label_filter=label_filter,
            name_filter=name_filter,
            show_all=show_all,
        )
        return json.dumps([c.model_dump(mode="json") for c in containers], ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_list_containers")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_list_containers elapsed_ms=%s", elapsed_ms)


async def docker_list_images(
    name_filter: Annotated[
        str | None,
        Field(description="repository:tag の部分一致フィルター。省略時は全イメージを返します。"),
    ] = None,
) -> Annotated[str, Field(description="イメージ情報の JSON 配列文字列。")]:
    """
    Docker イメージの一覧を取得します。

    name_filter で repository:tag を部分一致検索できます。
    """
    started = time.perf_counter()
    logger.info("MCP_TOOL_START tool=docker_list_images name=%s", name_filter)
    try:
        images = DockerOpsUtil.list_images(name_filter=name_filter)
        return json.dumps([image.model_dump(mode="json") for image in images], ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_list_images")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_list_images elapsed_ms=%s", elapsed_ms)


async def docker_remove_containers(
    container_ids: Annotated[
        list[str] | None,
        Field(description="削除するコンテナ ID のリスト。label_filter と併用可能。"),
    ] = None,
    label_filter: Annotated[
        str | None,
        Field(description='ラベルフィルター（"key=value" 形式）にマッチするコンテナを全て削除。container_ids と併用可能。'),
    ] = None,
    force: Annotated[
        bool,
        Field(description="実行中のコンテナも強制削除するかどうか。デフォルト True。"),
    ] = True,
) -> Annotated[str, Field(description="削除結果を JSON 文字列で返します。")]:
    """
    Docker コンテナを削除します。

    container_ids と label_filter の両方または一方を指定してください。
    label_filter にマッチする全コンテナを一括削除できます。

    [MCP_META]
    requires_approval=true
    action_kind=write
    usage_guidance=コンテナを削除します。label_filter を使用すると特定ラベルを持つコンテナを一括削除できます。削除前に docker_list_containers でコンテナを確認することを推奨します。
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=docker_remove_containers ids=%s label=%s",
        len(container_ids or []),
        label_filter,
    )
    try:
        result = DockerOpsUtil.remove_containers(
            container_ids=container_ids,
            label_filter=label_filter,
            force=force,
        )
        return result.model_dump_json()
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_remove_containers")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_remove_containers elapsed_ms=%s", elapsed_ms)


async def docker_remove_images(
    image_names: Annotated[
        list[str],
        Field(description="削除するイメージ名またはイメージ ID のリスト。例: ['myapp:latest']"),
    ],
    force: Annotated[
        bool,
        Field(description="使用中イメージも強制削除するかどうか。デフォルト False。"),
    ] = False,
) -> Annotated[str, Field(description="削除結果を JSON 文字列で返します。")]:
    """
    Docker イメージを削除します。

    image_names を 1 つ以上指定してください。

    [MCP_META]
    requires_approval=true
    action_kind=write
    usage_guidance=Docker イメージを削除します。削除前にコンテナが停止済みであることを確認してください。
    """
    started = time.perf_counter()
    logger.info("MCP_TOOL_START tool=docker_remove_images count=%s", len(image_names or []))
    try:
        result = DockerOpsUtil.remove_images(
            image_names=image_names,
            force=force,
        )
        return result.model_dump_json()
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=docker_remove_images")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("MCP_TOOL_END tool=docker_remove_images elapsed_ms=%s", elapsed_ms)
