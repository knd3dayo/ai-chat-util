from __future__ import annotations

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


def build_deep_agent_system_prompt(working_directory: str | None = None) -> str:
    working_directory_instruction = (
        f"- working_directory が与えられている場合、相対パスはまず {working_directory} 基準で解釈してください。\n"
        if working_directory
        else "- working_directory が与えられている場合、相対パスはまずその working_directory 基準で解釈してください。\n"
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
- パス、見出し、設定値などの具体値は、要約よりも原文の保持を優先してください。
- 文書見出しは Markdown 原文行をそのまま返してください。必要なら `HEADING_LINE_EXACT: <exact line>` の値をそのまま使ってください。
- 不足情報や承認が必要な場合は question で返して構いません。

出力フォーマットは必ずXML形式で返してください。
<OUTPUT>
    <TEXT>ユーザーへ返す本文</TEXT>
    <RESPONSE_TYPE>complete|question|reject</RESPONSE_TYPE>
</OUTPUT>
""".strip()