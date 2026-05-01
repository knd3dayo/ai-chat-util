from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Iterable, cast
from ai_chat_util.app.agent.core.agent_client_factory import AgentFactory
from ai_chat_util.core.analysis_service import AnalysisService
from ai_chat_util.core.chat import create_llm_client
from ai_chat_util.app.agent.hitl import create_stdio_hitl_client
from ai_chat_util.core.common.config.runtime import init_runtime, apply_logging_overrides, get_runtime_config_info
from ai_chat_util.core.chat.model import ChatRequestContext
from ai_chat_util.app.agent.core.app import run_mermaid_workflow_from_file
from ai_chat_util.app.workflow import WorkflowChatClient


def _add_common_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loglevel",
        type=str,
        default="",
        help="ログレベルを上書きします（例: DEBUG, INFO）。未指定の場合は ai-chat-util-config.yml の設定を使用します。",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="",
        help="ログのファイル出力先を上書きします。未指定の場合は ai-chat-util-config.yml の設定を使用します。",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ai_chat_util CLI")
    _add_common_logging_args(parser)

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "設定ファイル(ai-chat-util-config.yml)のパス。指定時は環境変数 AI_CHAT_UTIL_CONFIG にも反映し、"
            "後続処理に伝播します。未指定の場合は AI_CHAT_UTIL_CONFIG / カレント / プロジェクトルートの順で探索します。"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # chat
    chat_parser = subparsers.add_parser("chat", help="LLM へテキストでチャットします")
    chat_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="送信するプロンプト文字列",
    )
    agent_chat_parser = subparsers.add_parser("agent_chat", help="MCP を使用してテキストでチャットします")
    agent_chat_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="送信するプロンプト文字列",
    )
    agent_chat_parser.add_argument(
        "--workflow-file",
        type=str,
        default="",
        help="workflow backend で実行する Markdown workflow ファイルのパス",
    )
    agent_chat_parser.add_argument(
        "--workflow-plan-mode",
        action="store_true",
        help="workflow backend を plan mode で起動します",
    )
    agent_chat_parser.add_argument(
        "--workflow-non-durable",
        action="store_true",
        help="workflow backend を durable pause/resume なしで起動します",
    )
    agent_chat_parser.add_argument(
        "--workflow-max-node-visits",
        type=int,
        default=8,
        help="workflow 実行時の単一ノード訪問回数上限",
    )
    agent_chat_parser.add_argument(
        "--predictability",
        choices=["low", "medium", "high"],
        default="",
        help="要求の予見性ヒント",
    )
    agent_chat_parser.add_argument(
        "--approval-frequency",
        choices=["low", "medium", "high"],
        default="",
        help="承認頻度ヒント",
    )
    agent_chat_parser.add_argument(
        "--exploration-level",
        choices=["low", "medium", "high"],
        default="",
        help="探索性ヒント",
    )
    agent_chat_parser.add_argument(
        "--has-side-effects",
        action="store_true",
        help="副作用ありの処理として扱います",
    )
    deepagent_chat_parser = subparsers.add_parser("run_deepagent_chat", help="DeepAgents を使用してテキストでチャットします")
    deepagent_chat_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="送信するプロンプト文字列",
    )
    # batch_chat
    batch_chat_parser = subparsers.add_parser(
        "batch_chat", help="LLM へテキストでバッチチャットします"
    )
    batch_chat_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="送信するプロンプトテンプレート文字列",
    )
    agent_batch_chat_parser = subparsers.add_parser(
        "agent_batch_chat", help="MCP を使用してテキストでバッチチャットします"
    )
    agent_batch_chat_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="送信するプロンプトテンプレート文字列",
    )
    deepagent_batch_chat_parser = subparsers.add_parser(
        "run_deepagent_batch_chat", help="DeepAgents を使用してテキストでバッチチャットします"
    )
    deepagent_batch_chat_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="送信するプロンプトテンプレート文字列",
    )
    deepagent_batch_alias_parser = subparsers.add_parser(
        "deepagent_batch_chat", help="DeepAgents を使用してテキストでバッチチャットします"
    )
    deepagent_batch_alias_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="送信するプロンプトテンプレート文字列",
    )

    for current_batch_parser in (batch_chat_parser, agent_batch_chat_parser, deepagent_batch_chat_parser, deepagent_batch_alias_parser):
        current_batch_parser.add_argument(
            "-i",
            "--input_excel_path",
            type=str,
            required=True,
            help="処理対象のメッセージとファイルパスを記載したExcelファイルのパス",
        )
        current_batch_parser.add_argument(
            "-o",
            "--output_excel_path",
            type=str,
            default="output.xlsx",
            required=False,
            help="結果を出力するExcelファイルのパス",
        )
        current_batch_parser.add_argument(
            "--concurrency",
            type=int,
            default=16,
            required=False,
            help="同時実行数の上限（デフォルト: 16）",
        )
        current_batch_parser.add_argument(
            "--content_column",
            type=str,
            default="content",
            help="入力Excelファイル内のメッセージを含む列名（デフォルト: content）",
        )
        current_batch_parser.add_argument(
            "--file_path_column",
            type=str,
            default="file_path",
            help="入力Excelファイル内のファイルパスを含む列名（デフォルト: file_path）",
        )
        current_batch_parser.add_argument(
            "--output_column",
            type=str,
            default="output",
            help="出力Excelファイル内のLLM応答を含む列名（デフォルト: output）",
        )
        current_batch_parser.add_argument(
            "--image_detail",
            type=str,
            default="auto",
            help="画像解析のdetail（low/high/auto）。既定は auto",
        )
    
    # analyze_image_files
    image_parser = subparsers.add_parser(
        "analyze_image_files", help="画像ファイルを解析します"
    )
    image_parser.add_argument(
        "-i",
        "--image_path_list",
        type=str,
        nargs="+",
        required=True,
        help="画像ファイルパス（複数指定可）",
    )
    image_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="解析指示プロンプト",
    )
    image_parser.add_argument(
        "--detail",
        type=str,
        default="auto",
        help="画像解析のdetail（low/high/auto）。既定は auto",
    )

    # analyze_pdf_files
    pdf_parser = subparsers.add_parser("analyze_pdf_files", help="PDFファイルを解析します")
    pdf_parser.add_argument(
        "-i",
        "--pdf_path_list",
        type=str,
        nargs="+",
        required=True,
        help="PDFファイルパス（複数指定可）",
    )
    pdf_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="解析指示プロンプト",
    )
    pdf_parser.add_argument(
        "--detail",
        type=str,
        default="auto",
        help=(
            "features.use_custom_pdf_analyzer=true の場合に使われる detail（low/high/auto）。既定は auto"
        ),
    )

    # analyze_office_files
    office_parser = subparsers.add_parser(
        "analyze_office_files", help="Officeドキュメント（Word/Excel/PowerPoint等）をPDF化した後、解析します"
    )
    office_parser.add_argument(
        "-i",
        "--office_path_list",
        type=str,
        nargs="+",
        required=True,
        help="Officeドキュメントファイルパス（複数指定可）",
    )
    office_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="解析指示プロンプト",
    )
    office_parser.add_argument(
        "--detail",
        type=str,
        default="auto",
        help=(
            "features.use_custom_pdf_analyzer=true の場合に使われる detail（low/high/auto）。既定は auto"
        ),
    )

    # analyze_multi_format_files
    multi_parser = subparsers.add_parser(
        "analyze_files",
        help="複数形式（テキスト/画像/PDF/Office）ファイルをまとめて解析します",
    )
    multi_parser.add_argument(
        "-i",
        "--file_path_list",
        type=str,
        nargs="+",
        required=True,
        help="ファイルパス（複数指定可）",
    )
    multi_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="解析指示プロンプト",
    )
    multi_parser.add_argument(
        "--detail",
        type=str,
        default="auto",
        help=(
            "features.use_custom_pdf_analyzer=true の場合に使われる detail（low/high/auto）。既定は auto"
        ),
    )

    subparsers.add_parser(
        "show_config",
        help="実際に読み込まれた設定ファイルのパスと内容を表示します",
    )

    workflow_parser = subparsers.add_parser(
        "run_workflow",
        help="Markdown で定義されたWF型ワークフローを同期ワンショットで実行します",
    )
    workflow_parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="mermaid ブロックをちょうど1つ含む Markdown ファイルのパス",
    )
    workflow_parser.add_argument(
        "-m",
        "--message",
        type=str,
        default="",
        help="ワークフローへ渡す初期入力",
    )
    workflow_parser.add_argument(
        "--max-node-visits",
        type=int,
        default=8,
        help="ループ安全弁として同一ノードの最大実行回数を指定します",
    )
    durable_workflow_parser = subparsers.add_parser(
        "run_workflow_durable",
        help="Markdown で定義されたWF型ワークフローを durable pause/resume 付きで実行します",
    )
    durable_workflow_parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="mermaid ブロックをちょうど1つ含む Markdown ファイルのパス",
    )
    durable_workflow_parser.add_argument(
        "-m",
        "--message",
        type=str,
        default="",
        help="ワークフローへ渡す初期入力",
    )
    durable_workflow_parser.add_argument(
        "--max-node-visits",
        type=int,
        default=8,
        help="ループ安全弁として同一ノードの最大実行回数を指定します",
    )
    durable_workflow_parser.add_argument(
        "--plan-mode",
        action="store_true",
        help="実行前に Markdown と Mermaid を補正し、承認待ちで停止します",
    )

    return parser


def _validate_non_empty(text: str, parser: argparse.ArgumentParser) -> str:
    if not text.strip():
        parser.print_help()
        raise SystemExit(1)
    return text

def _print_header(command: str) -> None:
    print(f"Executing command: {command}")


def _build_agent_request_context(args: argparse.Namespace) -> ChatRequestContext | None:
    workflow_file = str(getattr(args, "workflow_file", "") or "").strip()
    predictability = str(getattr(args, "predictability", "") or "").strip()
    approval_frequency = str(getattr(args, "approval_frequency", "") or "").strip()
    exploration_level = str(getattr(args, "exploration_level", "") or "").strip()
    has_side_effects = bool(getattr(args, "has_side_effects", False))
    workflow_plan_mode = bool(getattr(args, "workflow_plan_mode", False))
    workflow_durable = not bool(getattr(args, "workflow_non_durable", False))
    workflow_max_node_visits = int(getattr(args, "workflow_max_node_visits", 8) or 8)

    if not any(
        [
            workflow_file,
            predictability,
            approval_frequency,
            exploration_level,
            has_side_effects,
            workflow_plan_mode,
            not workflow_durable,
            workflow_max_node_visits != 8,
        ]
    ):
        return None

    return ChatRequestContext(
        workflow_file_path=(workflow_file or None),
        workflow_plan_mode=workflow_plan_mode,
        workflow_durable=workflow_durable,
        workflow_max_node_visits=workflow_max_node_visits,
        predictability=cast(Any, predictability or None),
        approval_frequency=cast(Any, approval_frequency or None),
        exploration_level=cast(Any, exploration_level or None),
        has_side_effects=(True if has_side_effects else None),
    )

async def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Initialize runtime config first (ai-chat-util-config.yml required)
    init_runtime(args.config or None)

    # Optional logging overrides (process-local; does not touch env)
    apply_logging_overrides(level=args.loglevel or None, file=args.logfile or None)

    _print_header(args.command)

    if args.command == "chat":
        _validate_non_empty(args.prompt, parser)
        llm_client = create_llm_client()
        trace_id: str | None = None
        return await create_stdio_hitl_client(llm_client, trace_id=trace_id).run(args.prompt)

    if args.command == "agent_chat":
        _validate_non_empty(args.prompt, parser)
        llm_client = AgentFactory.create_mcp_client(default_request_context=_build_agent_request_context(args))
        trace_id: str | None = None
        return await create_stdio_hitl_client(llm_client, trace_id=trace_id).run(args.prompt)

    if args.command == "run_deepagent_chat":
        _validate_non_empty(args.prompt, parser)
        llm_client = AgentFactory.create_deepagent_client()
        trace_id: str | None = None
        return await create_stdio_hitl_client(llm_client, trace_id=trace_id).run(args.prompt)
    
    if args.command == "batch_chat":
        _validate_non_empty(args.prompt, parser)
        # Heavy deps (e.g., pandas) are only needed for batch_chat.
        from ai_chat_util.core.chat.batch_client import BatchClient

        llm_batch_client = BatchClient()
        await llm_batch_client.run_batch_chat_from_excel(
            input_excel_path=args.input_excel_path,
            output_excel_path=args.output_excel_path,
            prompt=args.prompt,
            content_column=args.content_column,
            file_path_column=args.file_path_column,
            output_column=args.output_column,
            concurrency=args.concurrency,
            detail=args.image_detail,
        )
        print(f"Batch chat completed. Results saved to {args.output_excel_path}")
        return

    if args.command == "agent_batch_chat":
        _validate_non_empty(args.prompt, parser)
        from ai_chat_util.app.agent.core import MCPBatchClient

        llm_batch_client = MCPBatchClient()
        await llm_batch_client.run_batch_chat_from_excel(
            input_excel_path=args.input_excel_path,
            output_excel_path=args.output_excel_path,
            prompt=args.prompt,
            content_column=args.content_column,
            file_path_column=args.file_path_column,
            output_column=args.output_column,
            concurrency=args.concurrency,
            detail=args.image_detail,
        )
        print(f"Batch chat completed. Results saved to {args.output_excel_path}")
        return

    if args.command in {"run_deepagent_batch_chat", "deepagent_batch_chat"}:
        _validate_non_empty(args.prompt, parser)
        from ai_chat_util.app.agent.core import DeepAgentBatchClient

        llm_batch_client = DeepAgentBatchClient()
        await llm_batch_client.run_batch_chat_from_excel(
            input_excel_path=args.input_excel_path,
            output_excel_path=args.output_excel_path,
            prompt=args.prompt,
            content_column=args.content_column,
            file_path_column=args.file_path_column,
            output_column=args.output_column,
            concurrency=args.concurrency,
            detail=args.image_detail,
        )
        print(f"Batch chat completed. Results saved to {args.output_excel_path}")
        return

    if args.command == "analyze_image_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = create_llm_client()
        response = await AnalysisService.analyze_image_files(llm_client, args.image_path_list, args.prompt, args.detail)
        print(response.output)
        return

    if args.command == "analyze_pdf_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = create_llm_client()
        response = await AnalysisService.analyze_pdf_files(llm_client, args.pdf_path_list, args.prompt, args.detail)
        print(response.output)
        return

    if args.command == "analyze_office_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = create_llm_client()
        response = await AnalysisService.analyze_office_files(llm_client, args.office_path_list, args.prompt, args.detail)
        print(response.output)
        return

    if args.command == "analyze_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = create_llm_client()
        response = await AnalysisService.analyze_files(llm_client, args.file_path_list, args.prompt, args.detail)
        print(response.output)
        return

    if args.command == "show_config":
        print(json.dumps(get_runtime_config_info(), ensure_ascii=False, indent=2))
        return

    if args.command == "run_workflow":
        response = await run_mermaid_workflow_from_file(
            workflow_file_path=args.file,
            message=args.message,
            max_node_visits=args.max_node_visits,
            durable=False,
            enable_tool_approval_nodes=False,
        )
        print(response.final_output)
        return

    if args.command == "run_workflow_durable":
        workflow_client = WorkflowChatClient(
            args.file,
            max_node_visits=args.max_node_visits,
            plan_mode=args.plan_mode,
            durable=True,
        )
        trace_id: str | None = None
        return await create_stdio_hitl_client(workflow_client, trace_id=trace_id).run(args.message)

    parser.print_help()
    raise SystemExit(1)

def cli_main() -> None:
    """console_scripts 用の同期エントリポイント。

    `[project.scripts]` から呼ばれる関数は同期関数である必要があるため、
    ここで asyncio.run して async main() を起動する。
    """

    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        import sys
        import ai_chat_util.core.log.log_settings as log_settings

        logger = log_settings.getLogger(__name__)
        logger.exception("Unhandled CLI error")
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    cli_main()
