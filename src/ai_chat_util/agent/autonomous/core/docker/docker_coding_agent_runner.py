from __future__ import annotations

from typing import Optional, cast, Union
import uuid
import pathlib
from pathlib import Path
import shlex
import os
from fastapi import UploadFile
from python_on_whales import docker as whales, Container, DockerClient
from ai_chat_util_base.model.agent_util_models import ComposeConfig, CodingAgentConfig, TaskStatus
from ..abstract_agent_runner import AbstractAgentRunner
from ..process_utils import get_host_uid_gid

from ..utils import ExecutorUtil

from ai_chat_util_base.config.runtime import get_autonomous_runtime_config

from ...util.logging import get_application_logger

logger = get_application_logger()

class CodingAgentRunner(AbstractAgentRunner):
    """
    コーディングエージェント用のdocker-compose.yml から設定を動的に読み取り、コンテナを実行するクラス
    
    
    """

    def __init__(
        self,
        compose_config: ComposeConfig,
        coding_agent_config: CodingAgentConfig,
        task_id: Optional[str] = None,
        workspace_path: Optional[Union[str, pathlib.Path]] = None,
        extra_env: Optional[dict[str, str]] = None,
    ):

        self.compose_config = compose_config
        self.coding_agent_config = coding_agent_config
        self.task_id = task_id or str(uuid.uuid4())  # タスクごとに一意のIDを生成
        if workspace_path is not None:
            self.workspace = pathlib.Path(workspace_path)
        else:
            self.workspace = pathlib.Path(self.coding_agent_config.workspace_root) / self.task_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        workspace_path = self.workspace.resolve().as_posix()
        self.task_status = TaskStatus.create(task_id=self.task_id, workspace_path=workspace_path)
        # 共有workspaceを使う場合に、後段（/status や artifacts算出）で参照できるよう保存しておく
        self.task_status.workspace_path = workspace_path
        self.task_status.metadata["workspace_path"] = workspace_path
        self.task_status.metadata["backend"] = "docker"

        self.command = shlex.split(self.compose_config.compose_command)
        self.detach = True  # デフォルトはバックグラウンド実行
        self.container = None

        self.extra_env: dict[str, str] = {
            str(k): str(v) for k, v in (extra_env or {}).items() if v is not None
        }


    def get_task_status(self) -> TaskStatus:
        """現在の TaskStatus を返す。"""
        return self.task_status

    def get_workspace_path(self) -> Path:
        """ワークスペースのパスを返す。"""
        return self.workspace.resolve()

    def prepare_workspace(self, 
                          data: Optional[dict[str, str]] = None, 
                          zip_file: Optional[UploadFile] = None, 
                          source_paths: Optional[list[pathlib.Path]] = None):
        """入力ソースに関わらずワークスペースを準備する（共通化）"""
        if zip_file:
            ExecutorUtil.add_zip_file(zip_file, self.workspace)

        if data:
            ExecutorUtil.add_data(data, self.workspace)

        if source_paths:
            ExecutorUtil.add_files(source_paths, self.workspace)
                    
    def start(self) -> Container:
        """
        コンテナを起動し、task_id を返します。
        volumes: [(ホストパス, コンテナパス, モード), ...] のリスト
        """
        params = {
            "service": self.compose_config.compose_service_name,
            "detach": self.detach,
            # "remove": True, # 終了時に自動削除
            "tty": False,   # ★明示的に False を指定（あるいは省略）
        }
        
        # 自然言語プロンプトは空白を含むため、呼び出し側で配列にして渡された場合はそのまま使う。
        # 文字列で渡された場合のみ shlex.split で分解する。
        params["command"] =  (shlex.split(self.command) if isinstance(self.command, str) else list(self.command))
        # WORKSPACE、USER_ID、GROUP_IDを設定
        # DoOD（docker.sock）利用時は、バンドルコンテナ内の UID/GID とホスト側の所有者が
        # 一致しないことがあるため、環境変数で上書きできるようにする。
        runtime_cfg = get_autonomous_runtime_config()
        uid, gid = get_host_uid_gid(
            default_uid=(runtime_cfg.host.uid if runtime_cfg.host.uid is not None else 1000),
            default_gid=(runtime_cfg.host.gid if runtime_cfg.host.gid is not None else 1000),
        )
        params["envs"] = {
            "WORKSPACE": self.workspace.as_posix(),
            "USER_ID": str(uid),
            "GROUP_ID": str(gid),
        }

        # Per-task environment variables (e.g., Authorization) for downstream tools.
        for k, v in self.extra_env.items():
            if v:
                params["envs"][str(k)] = str(v)

        # NOTE:
        # This runner intentionally does NOT inject LLM_* environment variables into the
        # container command (e.g., opencode). The external runner should manage its own
        # model/provider/base_url/credentials.

        # docker-compose.yml の volumes で ${WORKSPACE} を使っているため、
        # compose 側の変数置換に効く env-file をタスクごとに生成して渡す。
        # (compose.run(envs=...) は `--env` であり、変数置換には影響しない)
        compose_env_file = self.workspace / ".compose.env"
        compose_env_file.write_text(
            "\n".join(
                [
                    f"WORKSPACE={self.workspace.as_posix()}",
                    f"USER_ID={uid}",
                    f"GROUP_ID={gid}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        # クライアントは一度作れば使い回せます
        self.docker = DockerClient(
            compose_files=self.compose_config.get_compose_paths(), # type: ignore
            compose_project_directory=self.compose_config.compose_directory,
            compose_env_files=[compose_env_file],
            # service_name ではなく task_id をベースにしたユニークな名前に
            compose_project_name=f"task_{self.task_id}"
        )

        # コンテナを起動（Container オブジェクトが返る）
        self.container = self.docker.compose.run(**params)

        if not self.container or isinstance(self.container, str):
            raise RuntimeError("Failed to start container as an object")

        self.container = cast(Container, self.container)  # 明示的に Container 型にキャスト
        logger.info(
            f"Started container {self.container.name} for task {self.task_id} with command: {params['command']}"
        )

        return self.container


    @classmethod
    async def create_runner(
        cls, 
        prompt: str,
        initial_files: Optional[dict[str, str]] = None,
        zip_file: Optional[UploadFile] = None,
        source_paths: Optional[list[pathlib.Path]] = None,
        task_id: Optional[str] = None,
        detach: bool = True,
        workspace_path: Optional[Union[str, pathlib.Path]] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> "CodingAgentRunner":
        """
        インスタンス生成からコンテナ起動までを一括で行うエントリーポイント
        """
        # 1. Runnerの準備（非秘匿は ai-chat-util-config.yml から取得）
        runtime_cfg = get_autonomous_runtime_config()
        compose_config = ComposeConfig(
            compose_directory=runtime_cfg.compose.directory,
            compose_file=runtime_cfg.compose.file,
            compose_service_name=runtime_cfg.compose.service_name,
            compose_command=runtime_cfg.compose.command,
        )
        coding_agent_config = CodingAgentConfig(
            workspace_root=runtime_cfg.paths.workspace_root,
        )
        runner = cls(
            task_id=task_id, 
            compose_config=compose_config,
            coding_agent_config=coding_agent_config,
            workspace_path=workspace_path,
            extra_env=extra_env,
        )

        runner.detach = detach

        runner.prepare_workspace(
            data=initial_files,
            zip_file=zip_file,
            source_paths=source_paths,
        )        # 2. ファイルの配置（ZIPまたは初期ファイル）

        # 3. コマンドと実行設定
        command_base = compose_config.compose_command
        runner.command = shlex.split(command_base)
        if prompt:
            runner.command.append(prompt)
        
        return runner
