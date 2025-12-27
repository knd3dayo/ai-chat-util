"""ai_chat_util CLI.

`vector-search-util` の `__main__.py` と同様に、argparse + subcommand で実装する。

実行例:

- ヘルプ
    python -m ai_chat_util.cli --help

- チャット
    python -m ai_chat_util.cli chat -p "こんにちは"

- ファイル解析
    python -m ai_chat_util.cli analyze_files -i a.png b.jpg -p "内容を説明して" --detail auto

"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Iterable
from ai_chat_util.llm.model import ChatMessage, ChatHistory, ChatContent, ChatResponse
from ai_chat_util.llm.llm_client import LLMClient

def _set_env_if_provided(name: str, value: str) -> None:
    if value:
        os.environ[name] = value


def _add_common_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loglevel",
        type=str,
        default="",
        help="LOGLEVEL 環境変数を設定します（例: DEBUG, INFO）。指定しない場合は既存設定を使用します。",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="",
        help="LOGFILE 環境変数を設定します（ログをファイル出力）。指定しない場合は既存設定を使用します。",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ai_chat_util CLI")
    _add_common_logging_args(parser)

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
            "USE_CUSTOM_PDF_ANALYZER=true の場合に使われる detail（low/high/auto）。既定は auto"
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
            "USE_CUSTOM_PDF_ANALYZER=true の場合に使われる detail（low/high/auto）。既定は auto"
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
            "USE_CUSTOM_PDF_ANALYZER=true の場合に使われる detail（low/high/auto）。既定は auto"
        ),
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
    # NOTE: dotenv は各機能側でも読み込むが、CLI起動時点でも読み込んでおく
    from dotenv import load_dotenv

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    _set_env_if_provided("LOGLEVEL", args.loglevel)
    _set_env_if_provided("LOGFILE", args.logfile)
    load_dotenv()

    _print_header(args.command)

    if args.command == "chat":
        _validate_non_empty(args.prompt, parser)
        llm_client = LLMClient.create_llm_client()
        response = await llm_client.simple_chat(args.prompt)
        print(response)
        return

    if args.command == "analyze_image_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = LLMClient.create_llm_client()
        response = await llm_client.simple_image_analysis(args.image_path_list, args.prompt, args.detail)
        print(response)
        return

    if args.command == "analyze_pdf_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = LLMClient.create_llm_client()
        response = await llm_client.simple_pdf_analysis(args.pdf_path_list, args.prompt, args.detail)
        print(response)
        return

    if args.command == "analyze_office_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = LLMClient.create_llm_client()
        response = await llm_client.simple_office_document_analysis(args.office_path_list, args.prompt, args.detail)
        print(response)
        return

    if args.command == "analyze_files":
        _validate_non_empty(args.prompt, parser)
        llm_client = LLMClient.create_llm_client()
        response = await llm_client.simple_multi_format_document_analysis(args.file_path_list, args.prompt, args.detail)
        print(response)
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


if __name__ == "__main__":
    cli_main()
