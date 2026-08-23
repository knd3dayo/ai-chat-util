"""file_util.util.file_path_resolver

MCP/CLI/ローカル実行のどの形態でも、ユーザーが渡したファイルパスを
できるだけ「実在するパス」に解決するためのユーティリティ。

背景:
- Windows ホストの絶対パス (例: C:\\Users\\...\\a.pdf) を、Docker(Linux) 内の
  MCP サーバへ渡すとコンテナ側には存在しないため FileNotFoundError になる。
- docker-compose.yml では ./work を /app/work へ bind mount しているため、
  コンテナ内で解析したいファイルは通常 /app/work 配下に置く必要がある。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import ntpath
from pathlib import Path
from typing import Iterable, Literal


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "\"") or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def looks_like_windows_abs_path(path: str) -> bool:
    """C:\\... のような Windows 絶対パスに見えるか"""
    p = path.strip()
    return len(p) >= 3 and p[1] == ":" and (p[2] in ("\\", "/"))


def _maybe_parse_file_uri(path: str) -> str:
    p = path.strip()
    if p.lower().startswith("file://"):
        p = p[7:]
        p = p.lstrip("/")
    return p


def _iter_unique(candidates: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _find_repo_root(start: Path) -> Path | None:
    """pyproject.toml がある場所をリポジトリルートとみなして探索"""
    cur = start
    for _ in range(10):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


@dataclass(frozen=True)
class PathResolutionResult:
    resolved_path: str
    tried_candidates: list[str]
    path_kind: Literal["file", "directory"] = "file"


def resolve_existing_path(
    input_path: str,
    *,
    working_directory: str | None = None,
    extra_search_dirs: list[str] | None = None,
    allow_directory: bool = False,
) -> PathResolutionResult:
    """入力パスを、存在するファイルまたは directory パスへ解決する。"""

    raw = _strip_quotes(_maybe_parse_file_uri(input_path))
    expanded = os.path.expandvars(os.path.expanduser(raw))
    p = expanded

    cwd = Path.cwd()
    basename = ntpath.basename(p)

    repo_root = _find_repo_root(cwd)
    repo_work = (repo_root / "work") if repo_root else None
    docker_work = Path("/app/work")
    is_absolute_input = Path(p).is_absolute()

    candidates: list[str] = []
    if is_absolute_input:
        candidates.append(p)

    if working_directory and not is_absolute_input:
        wd = Path(working_directory).expanduser()
        candidates.append(str((wd / p)))
        candidates.append(str((wd / basename)))

    try:
        if not is_absolute_input:
            candidates.append(str((cwd / p).resolve()))
    except Exception:
        pass

    if not is_absolute_input:
        candidates.append(p)

    if repo_work is not None:
        candidates.append(str(repo_work / p))
        candidates.append(str(repo_work / basename))

    candidates.append(str(docker_work / p))
    candidates.append(str(docker_work / basename))

    if extra_search_dirs:
        for d in extra_search_dirs:
            dd = Path(d)
            candidates.append(str(dd / p))
            candidates.append(str(dd / basename))

    candidates = _iter_unique([c for c in candidates if c])

    for c in candidates:
        try:
            candidate = Path(c)
            if candidate.exists() and candidate.is_file():
                return PathResolutionResult(
                    resolved_path=str(candidate.resolve()),
                    tried_candidates=candidates,
                )
            if allow_directory and candidate.exists() and candidate.is_dir():
                return PathResolutionResult(
                    resolved_path=str(candidate.resolve()),
                    tried_candidates=candidates,
                    path_kind="directory",
                )
        except Exception:
            continue

    os_name = os.name
    is_win_path = looks_like_windows_abs_path(raw)
    hints: list[str] = []

    if is_win_path and os_name != "nt":
        hints.append(
            "入力が Windows の絶対パスに見えますが、実行環境が Windows ではありません。"
        )
        hints.append(
            "Docker コンテナ内で MCP サーバを動かしている場合、ホストの C:\\... は参照できません。"
        )
        hints.append(
            "対処: 対象ファイルをこのリポジトリの ./work に置き、コンテナ内パス /app/work/<ファイル名> を渡してください。"
        )

    if repo_work is not None:
        hints.append(f"参考: ホスト側の work/ ディレクトリ: {repo_work}")
    hints.append(f"CWD: {cwd}")
    if working_directory:
        hints.append(f"WORKING_DIRECTORY: {working_directory}")

    tried_preview = "\n".join([f"- {c}" for c in candidates[:20]])
    hint_text = "\n".join([f"- {h}" for h in hints])
    allowed_types = "file or directory" if allow_directory else "file"

    raise FileNotFoundError(
        "Path not found. Path resolution failed.\n"
        f"input: {input_path!r}\n"
        f"expanded: {expanded!r}\n"
        f"allowed_types: {allowed_types}\n"
        f"os.name: {os_name!r}\n"
        "tried candidates (first 20):\n"
        f"{tried_preview}\n"
        "hints:\n"
        f"{hint_text}"
    )


def resolve_existing_file_path(
    input_path: str,
    *,
    working_directory: str | None = None,
    extra_search_dirs: list[str] | None = None,
) -> PathResolutionResult:
    """入力パスを、存在するファイルパスへ解決する。"""

    return resolve_existing_path(
        input_path,
        working_directory=working_directory,
        extra_search_dirs=extra_search_dirs,
        allow_directory=False,
    )