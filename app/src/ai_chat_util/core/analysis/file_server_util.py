from __future__ import annotations

import os
import posixpath
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from ai_chat_util.core.common.config.runtime import FileServerAllowedRoot, FileServerSMBSection
from ai_chat_util.core.analysis.model import (
    FileServerEntry,
    FileServerEntryType,
    FileServerListResponse,
    FileServerProvider,
    FileUtilDocumentType,
    FileUtilDocument,
)


def _normalize_relative_path(path: str | None) -> str:
    raw = (path or ".").strip().replace("\\", "/")
    if not raw:
        return "."

    normalized = PurePosixPath(raw)
    if normalized.is_absolute():
        raise ValueError("path は設定ルート配下の相対パスで指定してください")

    parts: list[str] = []
    for part in normalized.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("path に '..' は指定できません")
        parts.append(part)
    return "/".join(parts) or "."


def _document_type_from_mime_type(mime_type: str | None) -> FileUtilDocumentType | None:
    if not mime_type:
        return None
    if mime_type.startswith("text/"):
        return FileUtilDocumentType.TEXT
    if mime_type == "application/pdf":
        return FileUtilDocumentType.PDF
    if mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return FileUtilDocumentType.EXCEL
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return FileUtilDocumentType.WORD
    if mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        return FileUtilDocumentType.PPT
    if mime_type.startswith("image/"):
        return FileUtilDocumentType.IMAGE
    return FileUtilDocumentType.UNSUPPORTED


def _isoformat_timestamp(timestamp: float | int | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def _sort_key(entry: FileServerEntry) -> tuple[int, str]:
    return (0 if entry.entry_type == FileServerEntryType.DIRECTORY else 1, entry.name.lower())


@dataclass
class _EntryCounter:
    max_entries: int
    total_entries: int = 0

    def consume(self) -> None:
        self.total_entries += 1
        if self.total_entries > self.max_entries:
            raise ValueError(f"返却件数が上限を超えました: max_entries={self.max_entries}")


class FileServerUtil:
    @classmethod
    def list_local_entries(
        cls,
        *,
        root: FileServerAllowedRoot,
        root_path: Path,
        relative_path: str,
        recursive: bool,
        max_depth: int,
        max_entries: int,
        include_hidden: bool,
        include_mime: bool,
        follow_symlinks: bool,
    ) -> FileServerListResponse:
        normalized_path = _normalize_relative_path(relative_path)
        resolved_root = root_path.expanduser().resolve()
        target = cls._resolve_local_target(
            resolved_root=resolved_root,
            relative_path=normalized_path,
            follow_symlinks=follow_symlinks,
        )
        if not target.exists():
            raise ValueError(f"指定パスが存在しません: {normalized_path}")
        if not target.is_dir():
            raise ValueError(f"指定パスはディレクトリではありません: {normalized_path}")

        counter = _EntryCounter(max_entries=max_entries)
        entries = cls._walk_local_directory(
            resolved_root=resolved_root,
            current_dir=target,
            current_relative_path=normalized_path,
            depth=0,
            recursive=recursive,
            max_depth=max_depth,
            include_hidden=include_hidden,
            include_mime=include_mime,
            follow_symlinks=follow_symlinks,
            counter=counter,
        )
        return FileServerListResponse(
            provider=FileServerProvider.LOCAL,
            root_name=root.name,
            root_path=str(resolved_root),
            path=normalized_path,
            recursive=recursive,
            max_depth=max_depth,
            total_entries=counter.total_entries,
            entries=entries,
        )

    @classmethod
    def _resolve_local_target(
        cls,
        *,
        resolved_root: Path,
        relative_path: str,
        follow_symlinks: bool,
    ) -> Path:
        target = resolved_root if relative_path == "." else resolved_root.joinpath(*relative_path.split("/"))
        resolved_target = target.resolve()
        try:
            resolved_target.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("要求された path が許可ルートの外を指しています") from exc

        if target.is_symlink() and not follow_symlinks:
            raise ValueError("シンボリックリンクのディレクトリ走査は無効です")
        return target

    @classmethod
    def _walk_local_directory(
        cls,
        *,
        resolved_root: Path,
        current_dir: Path,
        current_relative_path: str,
        depth: int,
        recursive: bool,
        max_depth: int,
        include_hidden: bool,
        include_mime: bool,
        follow_symlinks: bool,
        counter: _EntryCounter,
    ) -> list[FileServerEntry]:
        entries: list[FileServerEntry] = []
        with os.scandir(current_dir) as iterator:
            for item in iterator:
                if not include_hidden and item.name.startswith("."):
                    continue

                is_symlink = item.is_symlink()
                try:
                    is_dir = item.is_dir(follow_symlinks=follow_symlinks)
                    stat_result = item.stat(follow_symlinks=follow_symlinks)
                except OSError as exc:
                    raise ValueError(f"ディレクトリ項目の取得に失敗しました: {item.path}: {exc}") from exc

                child_relative_path = item.name if current_relative_path == "." else f"{current_relative_path}/{item.name}"
                mime_type = None
                document_type = None
                if include_mime and not is_dir:
                    mime_type, _ = FileUtilDocument.identify_file_type(item.path)
                    document_type = _document_type_from_mime_type(mime_type)

                counter.consume()
                child_entry = FileServerEntry(
                    name=item.name,
                    path=child_relative_path,
                    entry_type=FileServerEntryType.DIRECTORY if is_dir else FileServerEntryType.FILE,
                    size=None if is_dir else int(stat_result.st_size),
                    modified_at=_isoformat_timestamp(stat_result.st_mtime),
                    mime_type=mime_type,
                    document_type=document_type,
                    is_hidden=item.name.startswith("."),
                    is_symlink=is_symlink,
                )

                should_descend = (
                    recursive
                    and is_dir
                    and depth < max_depth
                    and (follow_symlinks or not is_symlink)
                )
                if should_descend:
                    child_dir = Path(item.path)
                    resolved_child = child_dir.resolve()
                    try:
                        resolved_child.relative_to(resolved_root)
                    except ValueError as exc:
                        raise ValueError(
                            f"許可ルート外のシンボリックリンクを検出しました: {child_relative_path}"
                        ) from exc
                    child_entry.children = cls._walk_local_directory(
                        resolved_root=resolved_root,
                        current_dir=child_dir,
                        current_relative_path=child_relative_path,
                        depth=depth + 1,
                        recursive=recursive,
                        max_depth=max_depth,
                        include_hidden=include_hidden,
                        include_mime=include_mime,
                        follow_symlinks=follow_symlinks,
                        counter=counter,
                    )
                entries.append(child_entry)

        entries.sort(key=_sort_key)
        return entries

    @classmethod
    def list_smb_entries(
        cls,
        *,
        root: FileServerAllowedRoot,
        smb: FileServerSMBSection,
        relative_path: str,
        recursive: bool,
        max_depth: int,
        max_entries: int,
        include_hidden: bool,
    ) -> FileServerListResponse:
        if not smb.enabled:
            raise ValueError("SMB 一覧機能が無効です。file_server.smb.enabled を true にしてください")
        if not smb.server or not smb.share:
            raise ValueError("SMB 利用時は file_server.smb.server と file_server.smb.share が必要です")
        if not smb.username or not smb.password:
            raise ValueError("SMB 利用時は file_server.smb.username と file_server.smb.password が必要です")

        normalized_path = _normalize_relative_path(relative_path)
        normalized_root_path = cls._normalize_smb_root_path(root.path)
        counter = _EntryCounter(max_entries=max_entries)

        try:
            from smb.SMBConnection import SMBConnection
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pysmb が利用できません") from exc

        connection = SMBConnection(
            smb.username,
            smb.password,
            socket.gethostname() or "ai-chat-util",
            smb.server,
            domain=smb.domain or "",
            use_ntlm_v2=True,
            is_direct_tcp=smb.port == 445,
        )
        if not connection.connect(smb.server, smb.port):
            raise ValueError("SMB サーバーへ接続できませんでした")

        try:
            requested_path = cls._join_smb_path(normalized_root_path, normalized_path)
            entries = cls._walk_smb_directory(
                connection=connection,
                share=smb.share,
                directory_path=requested_path,
                current_relative_path=normalized_path,
                depth=0,
                recursive=recursive,
                max_depth=max_depth,
                include_hidden=include_hidden,
                counter=counter,
            )
        finally:
            connection.close()

        logical_root = f"//{smb.server}/{smb.share}/{normalized_root_path}".rstrip("/")
        return FileServerListResponse(
            provider=FileServerProvider.SMB,
            root_name=root.name,
            root_path=logical_root or f"//{smb.server}/{smb.share}",
            path=normalized_path,
            recursive=recursive,
            max_depth=max_depth,
            total_entries=counter.total_entries,
            entries=entries,
        )

    @classmethod
    def _walk_smb_directory(
        cls,
        *,
        connection,
        share: str,
        directory_path: str,
        current_relative_path: str,
        depth: int,
        recursive: bool,
        max_depth: int,
        include_hidden: bool,
        counter: _EntryCounter,
    ) -> list[FileServerEntry]:
        try:
            raw_entries = connection.listPath(share, cls._to_smb_api_path(directory_path))
        except Exception as exc:
            raise ValueError(f"SMB ディレクトリの列挙に失敗しました: {current_relative_path}: {exc}") from exc

        entries: list[FileServerEntry] = []
        for item in raw_entries:
            name = str(getattr(item, "filename", ""))
            if name in ("", ".", ".."):
                continue
            if not include_hidden and name.startswith("."):
                continue

            is_dir = bool(getattr(item, "isDirectory", False))
            child_relative_path = name if current_relative_path == "." else f"{current_relative_path}/{name}"
            child_directory_path = cls._join_smb_path(directory_path, name)
            counter.consume()

            child_entry = FileServerEntry(
                name=name,
                path=child_relative_path,
                entry_type=FileServerEntryType.DIRECTORY if is_dir else FileServerEntryType.FILE,
                size=None if is_dir else int(getattr(item, "file_size", 0)),
                modified_at=_isoformat_timestamp(getattr(item, "last_write_time", None)),
                is_hidden=name.startswith("."),
                is_symlink=False,
            )
            if recursive and is_dir and depth < max_depth:
                child_entry.children = cls._walk_smb_directory(
                    connection=connection,
                    share=share,
                    directory_path=child_directory_path,
                    current_relative_path=child_relative_path,
                    depth=depth + 1,
                    recursive=recursive,
                    max_depth=max_depth,
                    include_hidden=include_hidden,
                    counter=counter,
                )
            entries.append(child_entry)

        entries.sort(key=_sort_key)
        return entries

    @staticmethod
    def _normalize_smb_root_path(path: str | None) -> str:
        raw = (path or "").strip().replace("\\", "/")
        if not raw:
            return ""
        normalized = PurePosixPath(raw)
        if normalized.is_absolute():
            raw = raw.lstrip("/")
            normalized = PurePosixPath(raw)
        parts: list[str] = []
        for part in normalized.parts:
            if part in ("", "."):
                continue
            if part == "..":
                raise ValueError("file_server.allowed_roots[].path に '..' は指定できません")
            parts.append(part)
        return "/".join(parts)

    @classmethod
    def _join_smb_path(cls, base_path: str, relative_path: str) -> str:
        normalized_base = cls._normalize_smb_root_path(base_path)
        normalized_relative = _normalize_relative_path(relative_path)
        if normalized_relative == ".":
            return normalized_base
        return posixpath.normpath(posixpath.join(normalized_base, normalized_relative)).lstrip("/")

    @staticmethod
    def _to_smb_api_path(path: str) -> str:
        normalized = path.strip("/")
        return "/" if not normalized else f"/{normalized}"