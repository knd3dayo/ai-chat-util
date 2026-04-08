from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from deepagents import create_deep_agent as _create_deep_agent
except Exception:  # pragma: no cover
    _create_deep_agent = None


def deepagents_available() -> bool:
    return _create_deep_agent is not None


def require_create_deep_agent() -> Any:
    if _create_deep_agent is None:
        raise RuntimeError(
            "deep_agent route requires the deepagents package. "
            "Install it with `pip install deepagents` or add it to dependencies before enabling features.enable_deep_agent."
        )
    return _create_deep_agent


def _normalize_explicit_paths(paths: Sequence[str] | None) -> list[str]:
    return [
        str(path).strip()
        for path in (paths or [])
        if isinstance(path, str) and str(path).strip()
    ]


def build_deep_agent_system_prompt(
    working_directory: str | None = None,
    *,
    explicit_user_file_paths: Sequence[str] | None = None,
    explicit_user_directory_paths: Sequence[str] | None = None,
) -> str:
    working_directory_instruction = (
        f"- working_directory が与えられている場合、相対パスはまず {working_directory} 基準で解釈してください。\n"
        if working_directory
        else "- working_directory が与えられている場合、相対パスはまずその working_directory 基準で解釈してください。\n"
    )
    normalized_file_paths = _normalize_explicit_paths(explicit_user_file_paths)
    normalized_directory_paths = _normalize_explicit_paths(explicit_user_directory_paths)
    explicit_target_lines: list[str] = []
    if normalized_file_paths:
        explicit_target_lines.append("- 明示された file path: " + ", ".join(normalized_file_paths))
    if normalized_directory_paths:
        explicit_target_lines.append("- 明示された directory path: " + ", ".join(normalized_directory_paths))
        explicit_target_lines.append(
            "- 明示された directory path は、その path 自体を analyze_files 系ツールへ渡してください。存在確認前に docs/** のような glob や child path へ勝手に変形しないでください。"
        )
    explicit_targets_instruction = (
        "\n[この run で明示されている対象]\n" + "\n".join(explicit_target_lines) + "\n"
        if explicit_target_lines
        else ""
    )

    return f"""
あなたは deep_agent ルートの実行エージェントです。
このルートでは、複雑な調査や分解は行ってよいですが、既存 coding_agent の非同期ジョブ経路は担当しません。

[重要な制約]
- execute / status / get_result / workspace_path / cancel は使わないでください。
- 同一目的で同じツールを繰り返し呼ばないでください。
- ユーザーがローカルファイルパスを指定した場合、アクセス不能と決めつけず、利用可能なファイル系ツールや MCP ツールで確認してください。
- 絶対パスが存在する場合はそのまま使ってください。
{working_directory_instruction}- directory パスが指定された場合、file 指定として却下せず、存在確認のうえで配下を探索し、対象ファイルへ展開してから解析してください。
- directory path を扱う際は、根拠となる tool 結果を得る前に「空」や「未検出」と断定しないでください。まず analyze_files 系ツールで確認してください。
- パス、見出し、設定値などの具体値は、要約よりも原文の保持を優先してください。
- 文書見出しは Markdown 原文行をそのまま返してください。必要なら `HEADING_LINE_EXACT: <exact line>` の値をそのまま使ってください。
- 不足情報や承認が必要な場合は question で返して構いません。
{explicit_targets_instruction}

出力フォーマットは必ずXML形式で返してください。
<OUTPUT>
    <TEXT>ユーザーへ返す本文</TEXT>
    <RESPONSE_TYPE>complete|question|reject</RESPONSE_TYPE>
</OUTPUT>
""".strip()