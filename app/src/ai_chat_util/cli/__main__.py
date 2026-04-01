from __future__ import annotations

import argparse
import asyncio
import json
from typing import Iterable
from ai_chat_util.base.agent.agent_client_factory import AgentFactory
from ai_chat_util.base.llm.llm_client_factory import LLMFactory
from ..base.llm.llm_client_util import LLMClientUtil
from ai_chat_util.common.config.runtime import init_runtime, apply_logging_overrides, get_runtime_config_info


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

    return parser


def _validate_non_empty(text: str, parser: argparse.ArgumentParser) -> str:
    if not text.strip():
        parser.print_help()
        raise SystemExit(1)
    return text

def _print_header(command: str) -> None:
    print(f"Executing command: {command}")

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
        llm_client = LLMFactory.create_llm_client()
        trace_id: str | None = None
        return await LLMFactory.create_stdio_hitl_client(llm_client, trace_id=trace_id).run(args.prompt)

    if args.command == "agent_chat":
        _validate_non_empty(args.prompt, parser)
        llm_client = AgentFactory.create_mcp_client()
        trace_id: str | None = None
        return await LLMFactory.create_stdio_hitl_client(llm_client, trace_id=trace_id).run(args.prompt)

    if args.command == "run_deepagent_chat":
        _validate_non_empty(args.prompt, parser)
        llm_client = AgentFactory.create_deepagent_client()
        trace_id: str | None = None
        return await LLMFactory.create_stdio_hitl_client(llm_client, trace_id=trace_id).run(args.prompt)
    
    if args.command == "batch_chat":
        _validate_non_empty(args.prompt, parser)
        # Heavy deps (e.g., pandas) are only needed for batch_chat.
        from ai_chat_util.base.llm.llm_batch_client import LLMBatchClient

        llm_batch_client = LLMBatchClient()
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
        from ai_chat_util.base.agent.agent_batch_client import MCPBatchClient

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
        from ai_chat_util.base.agent.agent_batch_client import DeepAgentBatchClient

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
        llm_client = LLMFactory.create_llm_client()
        response = await LLMClientUtil.analyze_image_files(llm_client, args.image_path_list, args.prompt, args.detail)
        print(response.output)
        return

    if args.command == "analyze_pdf_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = LLMFactory.create_llm_client()
        response = await LLMClientUtil.analyze_pdf_files(llm_client, args.pdf_path_list, args.prompt, args.detail)
        print(response.output)
        return

    if args.command == "analyze_office_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = LLMFactory.create_llm_client()
        response = await LLMClientUtil.analyze_office_files(llm_client, args.office_path_list, args.prompt, args.detail)
        print(response.output)
        return

    if args.command == "analyze_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = LLMFactory.create_llm_client()
        response = await LLMClientUtil.analyze_files(llm_client, args.file_path_list, args.prompt, args.detail)
        print(response.output)
        return

    if args.command == "show_config":
        print(json.dumps(get_runtime_config_info(), ensure_ascii=False, indent=2))
        return

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
        import ai_chat_util.log.log_settings as log_settings

        logger = log_settings.getLogger(__name__)
        logger.exception("Unhandled CLI error")
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    cli_main()
