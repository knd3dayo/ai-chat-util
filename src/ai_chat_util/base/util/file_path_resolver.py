"""ai_chat_util.util.file_path_resolver

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
from typing import Iterable


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "\"") or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def looks_like_windows_abs_path(path: str) -> bool:
    """C:\\... のような Windows 絶対パスに見えるか"""
    p = path.strip()
    # `C:\` or `C:/`
    return len(p) >= 3 and p[1] == ":" and (p[2] in ("\\", "/"))


def _maybe_parse_file_uri(path: str) -> str:
    # file:///C:/... などが来たときに備える（厳密でなくてよい）
    p = path.strip()
    if p.lower().startswith("file://"):
        p = p[7:]
        # file:///C:/... -> /C:/... のように先頭に / が付く場合がある
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


def resolve_existing_file_path(
    input_path: str,
    *,
    working_directory: str | None = None,
    extra_search_dirs: list[str] | None = None,
) -> PathResolutionResult:
    """入力パスを、存在するファイルパスへ解決する。

    Args:
        input_path: ユーザー入力のパス（絶対/相対/Windowsパス/ file:// URI など）
        working_directory: 環境変数 WORKING_DIRECTORY 等で指定される検索基点（任意）
        extra_search_dirs: 追加の探索ディレクトリ（任意）

    Returns:
        PathResolutionResult(resolved_path=..., tried_candidates=[...])

    Raises:
        FileNotFoundError: 候補のいずれにも存在しない場合
    """

    raw = _strip_quotes(_maybe_parse_file_uri(input_path))
    expanded = os.path.expandvars(os.path.expanduser(raw))
    p = expanded

    cwd = Path.cwd()
    # POSIX 環境で Windows パスが来た場合でも file name を取れるよう ntpath を使う
    # 例: "C:\\a\\b\\c.pdf" -> "c.pdf"
    basename = ntpath.basename(p)

    repo_root = _find_repo_root(cwd)
    repo_work = (repo_root / "work") if repo_root else None

    # docker-compose の bind mount 想定
    docker_work = Path("/app/work")

    candidates: list[str] = []

    # 1) 入力をそのまま
    candidates.append(p)

    # 2) 相対パスなら CWD 基準も試す
    try:
        if not Path(p).is_absolute():
            candidates.append(str((cwd / p).resolve()))
    except Exception:
        # `Path("C:\\...").is_absolute()` は OS により挙動がブレるので握りつぶす
        pass

    # 3) WORKING_DIRECTORY 基準
    if working_directory:
        wd = Path(working_directory)
        candidates.append(str((wd / p)))
        candidates.append(str((wd / basename)))

    # 4) repo の ./work 基準
    if repo_work is not None:
        candidates.append(str(repo_work / p))
        candidates.append(str(repo_work / basename))

    # 5) /app/work 基準（Docker内）
    candidates.append(str(docker_work / p))
    candidates.append(str(docker_work / basename))

    # 6) 追加探索
    if extra_search_dirs:
        for d in extra_search_dirs:
            dd = Path(d)
            candidates.append(str(dd / p))
            candidates.append(str(dd / basename))

    candidates = _iter_unique([c for c in candidates if c])

    for c in candidates:
        try:
            if Path(c).exists() and Path(c).is_file():
                return PathResolutionResult(resolved_path=str(Path(c).resolve()), tried_candidates=candidates)
        except Exception:
            # 変な文字列などで Path が死ぬケースを考慮してスキップ
            continue

    # 見つからなかった場合は、原因が分かるように情報を盛った例外を投げる
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

    raise FileNotFoundError(
        "File not found. Path resolution failed.\n"
        f"input: {input_path!r}\n"
        f"expanded: {expanded!r}\n"
        f"os.name: {os_name!r}\n"
        "tried candidates (first 20):\n"
        f"{tried_preview}\n"
        "hints:\n"
        f"{hint_text}"
    )
