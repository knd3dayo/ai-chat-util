from __future__ import annotations

import tempfile
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from python_on_whales import DockerClient
from python_on_whales import docker as whales

from ai_chat_util.core.docker.model import ComposeOperationResult, ContainerInfo, ImageInfo
import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


@contextmanager
def _compose_client(
    *,
    compose_path: Optional[str | Path] = None,
    compose_content: Optional[str] = None,
    project_name: str,
    env_vars: Optional[dict[str, str]] = None,
    project_directory: Optional[str | Path] = None,
) -> Generator[DockerClient, None, None]:
    """DockerClient (compose) を生成するコンテキストマネージャー。

    compose_content が指定された場合は一時ファイルに書き出し、
    compose_path が指定された場合はそのパスを直接使用します。
    両方指定された場合は compose_content が優先されます。
    compose_content に相対パスが含まれる場合は project_directory を指定してください。
    """
    env_file_path: str | None = None
    compose_project_directory: str | None = None

    if env_vars:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".env",
            delete=False,
            encoding="utf-8",
        ) as env_file:
            env_file.write("\n".join(f"{key}={value}" for key, value in env_vars.items()))
            env_file.write("\n")
            env_file_path = env_file.name

    if compose_content is not None:
        # 一時ファイルに書き出して使用
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yml",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(compose_content)
            tmp_path = f.name
        try:
            client = DockerClient(
                compose_files=[tmp_path],
                compose_project_directory=str(Path(project_directory) if project_directory is not None else Path(tmp_path).parent),
                compose_env_files=([env_file_path] if env_file_path else None),
                compose_project_name=project_name,
            )
            yield client
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if env_file_path:
                try:
                    os.unlink(env_file_path)
                except OSError:
                    pass
    elif compose_path is not None:
        compose_path = Path(compose_path)
        compose_project_directory = str(compose_path.parent)
        client = DockerClient(
            compose_files=[str(compose_path)],
            compose_project_directory=compose_project_directory,
            compose_env_files=([env_file_path] if env_file_path else None),
            compose_project_name=project_name,
        )
        try:
            yield client
        finally:
            if env_file_path:
                try:
                    os.unlink(env_file_path)
                except OSError:
                    pass
    else:
        if env_file_path:
            try:
                os.unlink(env_file_path)
            except OSError:
                pass
        raise ValueError("compose_path または compose_content のいずれかを指定してください。")


class DockerOpsUtil:
    """Docker コンテナ・compose 操作の汎用ユーティリティクラス。

    coding agent などの固有ロジックには依存せず、python_on_whales のみを使用します。
    """

    # ------------------------------------------------------------------
    # compose 操作
    # ------------------------------------------------------------------

    @classmethod
    def compose_up(
        cls,
        *,
        project_name: str,
        compose_path: Optional[str | Path] = None,
        compose_content: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
        project_directory: Optional[str | Path] = None,
        service_names: Optional[list[str]] = None,
        detach: bool = True,
        build: bool = False,
    ) -> ComposeOperationResult:
        """docker compose up を実行します。

        Args:
            project_name: compose プロジェクト名（コンテナの識別に使用）。
            compose_path: docker-compose.yml のファイルパス。
            compose_content: docker-compose.yml の内容（文字列）。compose_path より優先。
            env_vars: compose ファイル内の変数置換に使用する環境変数の辞書。
            project_directory: compose_content 内の相対 build context / volume 解決に使う基準ディレクトリ。
            service_names: 起動するサービス名のリスト（省略時は全サービス）。
            detach: バックグラウンドで実行するかどうか。デフォルト True。
            build: 起動前にイメージをビルドするかどうか。デフォルト False。

        Returns:
            ComposeOperationResult
        """
        try:
            with _compose_client(
                compose_path=compose_path,
                compose_content=compose_content,
                project_name=project_name,
                env_vars=env_vars,
                project_directory=project_directory,
            ) as client:
                kwargs: dict = {"detach": detach, "build": build}
                if service_names:
                    kwargs["services"] = service_names
                client.compose.up(**kwargs)
            containers = cls.list_containers(
                label_filter=f"com.docker.compose.project={project_name}",
                show_all=True,
            )
            container_names = [container.name for container in containers]
            image_names = sorted({container.image for container in containers if container.image})
            logger.info("compose_up project=%s detach=%s", project_name, detach)
            return ComposeOperationResult(
                success=True,
                project_name=project_name,
                container_names=container_names,
                image_names=image_names,
            )
        except Exception as exc:
            logger.exception("compose_up failed project=%s", project_name)
            return ComposeOperationResult(
                success=False,
                project_name=project_name,
                error=str(exc),
            )

    @classmethod
    def compose_down(
        cls,
        *,
        project_name: str,
        compose_path: Optional[str | Path] = None,
        compose_content: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
        project_directory: Optional[str | Path] = None,
        remove_volumes: bool = False,
    ) -> ComposeOperationResult:
        """docker compose down を実行します。

        Args:
            project_name: 停止する compose プロジェクト名。
            compose_path: docker-compose.yml のファイルパス。
            compose_content: docker-compose.yml の内容（文字列）。compose_path より優先。
            env_vars: 環境変数の辞書。
            project_directory: compose_content 内の相対パス解決に使う基準ディレクトリ。
            remove_volumes: ボリュームも削除するかどうか。デフォルト False。

        Returns:
            ComposeOperationResult
        """
        try:
            with _compose_client(
                compose_path=compose_path,
                compose_content=compose_content,
                project_name=project_name,
                env_vars=env_vars,
                project_directory=project_directory,
            ) as client:
                client.compose.down(volumes=remove_volumes)
            logger.info("compose_down project=%s remove_volumes=%s", project_name, remove_volumes)
            return ComposeOperationResult(success=True, project_name=project_name)
        except Exception as exc:
            logger.exception("compose_down failed project=%s", project_name)
            return ComposeOperationResult(
                success=False,
                project_name=project_name,
                error=str(exc),
            )

    @classmethod
    def compose_restart(
        cls,
        *,
        project_name: str,
        compose_path: Optional[str | Path] = None,
        compose_content: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
        project_directory: Optional[str | Path] = None,
        service_names: Optional[list[str]] = None,
    ) -> ComposeOperationResult:
        """docker compose restart を実行します。

        Args:
            project_name: 再起動する compose プロジェクト名。
            compose_path: docker-compose.yml のファイルパス。
            compose_content: docker-compose.yml の内容（文字列）。compose_path より優先。
            env_vars: 環境変数の辞書。
            project_directory: compose_content 内の相対パス解決に使う基準ディレクトリ。
            service_names: 再起動するサービス名のリスト（省略時は全サービス）。

        Returns:
            ComposeOperationResult
        """
        try:
            with _compose_client(
                compose_path=compose_path,
                compose_content=compose_content,
                project_name=project_name,
                env_vars=env_vars,
                project_directory=project_directory,
            ) as client:
                kwargs: dict = {}
                if service_names:
                    kwargs["services"] = service_names
                client.compose.restart(**kwargs)
            logger.info("compose_restart project=%s", project_name)
            return ComposeOperationResult(success=True, project_name=project_name)
        except Exception as exc:
            logger.exception("compose_restart failed project=%s", project_name)
            return ComposeOperationResult(
                success=False,
                project_name=project_name,
                error=str(exc),
            )

    @classmethod
    def compose_logs(
        cls,
        *,
        project_name: str,
        compose_path: Optional[str | Path] = None,
        compose_content: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
        project_directory: Optional[str | Path] = None,
        service_names: Optional[list[str]] = None,
        tail: Optional[int] = 200,
    ) -> str:
        """docker compose logs を取得します。

        Args:
            project_name: ログを取得する compose プロジェクト名。
            compose_path: docker-compose.yml のファイルパス。
            compose_content: docker-compose.yml の内容（文字列）。compose_path より優先。
            env_vars: 環境変数の辞書。
            project_directory: compose_content 内の相対パス解決に使う基準ディレクトリ。
            service_names: ログを取得するサービス名のリスト（省略時は全サービス）。
            tail: 取得する末尾の行数。デフォルト 200。

        Returns:
            ログ文字列
        """
        with _compose_client(
            compose_path=compose_path,
            compose_content=compose_content,
            project_name=project_name,
            env_vars=env_vars,
            project_directory=project_directory,
        ) as client:
            kwargs: dict = {}
            if service_names:
                kwargs["services"] = service_names
            if tail is not None:
                kwargs["tail"] = str(tail)
            raw = client.compose.logs(**kwargs)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw) if raw is not None else ""

    # ------------------------------------------------------------------
    # コンテナ一覧・削除
    # ------------------------------------------------------------------

    @classmethod
    def list_containers(
        cls,
        *,
        label_filter: Optional[str] = None,
        name_filter: Optional[str] = None,
        show_all: bool = True,
    ) -> list[ContainerInfo]:
        """Docker コンテナの一覧を取得します。

        Args:
            label_filter: "key=value" 形式のラベルフィルター（例: "managed_by=myapp"）。
            name_filter: コンテナ名の部分一致フィルター。
            show_all: 停止中のコンテナも含めるかどうか。デフォルト True。

        Returns:
            ContainerInfo のリスト
        """
        filters: dict[str, str] = {}
        if label_filter:
            filters["label"] = label_filter
        if name_filter:
            filters["name"] = name_filter

        containers = whales.container.list(all=show_all, filters=filters if filters else {})
        result: list[ContainerInfo] = []
        for c in containers:
            try:
                cfg = getattr(c, "config", None)
                image_name = getattr(cfg, "image", "") if cfg is not None else ""
                labels_raw = getattr(cfg, "labels", {}) if cfg is not None else {}
                labels: dict[str, str] = {str(k): str(v) for k, v in (labels_raw or {}).items()}
                created = getattr(c, "created", None)
                result.append(
                    ContainerInfo(
                        id=str(c.id),
                        short_id=str(c.id)[:12],
                        name=str(c.name),
                        status=str(getattr(getattr(c, "state", None), "status", "unknown")),
                        image=str(image_name),
                        labels=labels,
                        created_at=created,
                    )
                )
            except Exception:
                logger.debug("Failed to parse container info for %s", getattr(c, "id", "?"), exc_info=True)
        return result

    @classmethod
    def remove_containers(
        cls,
        *,
        container_ids: Optional[list[str]] = None,
        label_filter: Optional[str] = None,
        force: bool = True,
    ) -> ComposeOperationResult:
        """Docker コンテナを削除します。

        container_ids と label_filter の両方が指定された場合、両方の条件にマッチしたコンテナを削除します。
        label_filter のみ指定した場合は、そのラベルを持つ全コンテナを削除します。

        Args:
            container_ids: 削除するコンテナ ID のリスト。
            label_filter: "key=value" 形式のラベルフィルター。
            force: 実行中のコンテナも強制削除するかどうか。デフォルト True。

        Returns:
            ComposeOperationResult
        """
        if not container_ids and not label_filter:
            return ComposeOperationResult(
                success=False,
                project_name="",
                error="container_ids または label_filter のいずれかを指定してください。",
            )
        removed: list[str] = []
        errors: list[str] = []

        target_ids: set[str] = set(container_ids or [])
        if label_filter:
            labeled = whales.container.list(all=True, filters={"label": label_filter})
            for c in labeled:
                target_ids.add(str(c.id))

        for cid in target_ids:
            try:
                whales.container.remove(cid, force=force)
                removed.append(cid[:12])
                logger.info("container removed id=%s", cid[:12])
            except Exception as exc:
                errors.append(f"{cid[:12]}: {exc}")
                logger.warning("container remove failed id=%s err=%s", cid[:12], exc)

        return ComposeOperationResult(
            success=len(errors) == 0,
            project_name="",
            output=f"Removed: {', '.join(removed)}" if removed else "",
            error="; ".join(errors),
        )

    @classmethod
    def remove_images(
        cls,
        *,
        image_names: Optional[list[str]] = None,
        force: bool = False,
    ) -> ComposeOperationResult:
        """Docker イメージを削除します。

        Args:
            image_names: 削除するイメージ名または ID のリスト。
            force: 強制削除するかどうか。デフォルト False。

        Returns:
            ComposeOperationResult
        """
        if not image_names:
            return ComposeOperationResult(
                success=False,
                project_name="",
                error="image_names を 1 つ以上指定してください。",
            )

        removed: list[str] = []
        errors: list[str] = []
        for image_name in image_names:
            try:
                whales.image.remove(image_name, force=force)
                removed.append(image_name)
                logger.info("image removed name=%s", image_name)
            except Exception as exc:
                errors.append(f"{image_name}: {exc}")
                logger.warning("image remove failed name=%s err=%s", image_name, exc)

        return ComposeOperationResult(
            success=len(errors) == 0,
            project_name="",
            output=f"Removed: {', '.join(removed)}" if removed else "",
            error="; ".join(errors),
        )

    @classmethod
    def list_images(
        cls,
        *,
        name_filter: Optional[str] = None,
    ) -> list[ImageInfo]:
        """Docker イメージの一覧を取得します。

        Args:
            name_filter: repository:tag の部分一致フィルター。

        Returns:
            ImageInfo のリスト
        """
        images = whales.image.list()
        result: list[ImageInfo] = []
        for image in images:
            try:
                repo_tags_raw = getattr(image, "repo_tags", None) or getattr(image, "tags", None) or []
                repo_tags = [str(tag) for tag in repo_tags_raw if tag]
                if name_filter and not any(name_filter in tag for tag in repo_tags):
                    continue
                image_id = str(getattr(image, "id", ""))
                result.append(
                    ImageInfo(
                        id=image_id,
                        short_id=image_id[:12],
                        repo_tags=repo_tags,
                        size=getattr(image, "size", None),
                        created_at=getattr(image, "created", None),
                    )
                )
            except Exception:
                logger.debug("Failed to parse image info for %s", getattr(image, "id", "?"), exc_info=True)
        return result

    # ------------------------------------------------------------------
    # コンテナログ（単一コンテナ）
    # ------------------------------------------------------------------

    @classmethod
    def get_container_logs(cls, container, tail: int = 200) -> tuple[str, str]:
        """docker コンテナの stdout/stderr を取得して (stdout, stderr) を返します。

        Args:
            container: python_on_whales の Container オブジェクト。
            tail: 取得する末尾の行数。デフォルト 200。

        Returns:
            (stdout, stderr) の文字列タプル
        """
        out = container.logs(stdout=True, stderr=False, tail=tail)
        err = container.logs(stdout=False, stderr=True, tail=tail)
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        return str(out), str(err)

    # ------------------------------------------------------------------
    # 単一コンテナ操作
    # ------------------------------------------------------------------

    @classmethod
    def kill_container(cls, container_id: str) -> None:
        """コンテナを強制終了します。

        Args:
            container_id: 終了するコンテナの ID。
        """
        whales.container.kill(container_id)
        logger.info("container killed id=%s", container_id[:12])
