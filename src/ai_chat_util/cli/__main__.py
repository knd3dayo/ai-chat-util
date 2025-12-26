"""ai_chat_util CLI.

`vector-search-util` の `__main__.py` と同様に、argparse + subcommand で実装する。

実行例:

- ヘルプ
    python -m ai_chat_util.cli --help

- チャット
    python -m ai_chat_util.cli chat -p "こんにちは"

- 画像解析
    python -m ai_chat_util.cli analyze_image_files -i a.png b.jpg -p "内容を説明して" --detail auto

- MCPサーバ起動
    python -m ai_chat_util.cli mcp_server --mode stdio
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Iterable


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
        "analyze_office_files", help="Officeドキュメント（Word/Excel/PowerPoint等）を解析します"
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
        "analyze_multi_format_files",
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

    # mcp_server
    mcp_parser = subparsers.add_parser("mcp_server", help="MCPサーバを起動します")
    mcp_parser.add_argument(
        "-m",
        "--mode",
        choices=["sse", "http", "stdio"],
        default="stdio",
        help="起動モード（stdio/sse/http）。既定は stdio",
    )
    mcp_parser.add_argument(
        "-t",
        "--tools",
        type=str,
        default="",
        help="登録するツール名（カンマ区切り）。未指定の場合はデフォルトツール一式を登録",
    )
    mcp_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5001,
        help="サーバ起動ポート（sse/http のとき使用）。既定は 5001",
    )
    mcp_parser.add_argument(
        "-v",
        "--server_log_level",
        type=str,
        default="",
        help="サーバのログレベル（環境変数 LOGLEVEL を上書きしたい場合に使用）",
    )

    # api_server
    api_parser = subparsers.add_parser("api_server", help="FastAPI サーバを起動します")
    api_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="バインドするホスト。既定は 0.0.0.0",
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="バインドするポート。既定は 8000",
    )

    return parser


def _validate_non_empty(text: str, parser: argparse.ArgumentParser) -> str:
    if not text.strip():
        parser.print_help()
        raise SystemExit(1)
    return text


def _print_header(command: str) -> None:
    print(f"Executing command: {command}")


def _ensure_api_key() -> None:
    """LLM呼び出し前に最低限の設定チェックを行う。

    - 現状の実装では OpenAI/Azure OpenAI ともに OPENAI_API_KEY を利用するため、
      未設定の場合は早期にエラー終了する。
    """

    from ai_chat_util.llm.llm_config import LLMConfig

    cfg = LLMConfig()
    if not cfg.api_key:
        raise SystemExit(
            "OPENAI_API_KEY が未設定です。.env を用意するか環境変数を設定してください。"
        )


async def _run_chat(prompt: str) -> None:
    from ai_chat_util.llm.llm_client import LLMClient
    from ai_chat_util.llm.llm_config import LLMConfig

    _ensure_api_key()
    client = LLMClient.create_llm_client(LLMConfig())
    resp = await client.simple_chat(prompt)
    print(resp.output)


async def _analyze_image_files(image_path_list: list[str], prompt: str, detail: str) -> None:
    from ai_chat_util.llm.llm_client import LLMClient
    from ai_chat_util.llm.llm_config import LLMConfig

    _ensure_api_key()
    client = LLMClient.create_llm_client(LLMConfig())
    text = await client.analyze_image_files(image_path_list, prompt, detail)
    print(text)


async def _analyze_pdf_files(pdf_path_list: list[str], prompt: str, detail: str) -> None:
    from ai_chat_util.core.app import use_custom_pdf_analyzer
    from ai_chat_util.llm.llm_client import LLMClient
    from ai_chat_util.llm.llm_config import LLMConfig

    _ensure_api_key()
    client = LLMClient.create_llm_client(LLMConfig())
    if use_custom_pdf_analyzer():
        text = await client.analyze_pdf_files_custom(pdf_path_list, prompt, detail=detail)
    else:
        text = await client.analyze_pdf_files(pdf_path_list, prompt)
    print(text)


async def _analyze_office_files(office_path_list: list[str], prompt: str, detail: str) -> None:
    from ai_chat_util.core.app import use_custom_pdf_analyzer
    from ai_chat_util.llm.llm_client import LLMClient
    from ai_chat_util.llm.llm_config import LLMConfig

    _ensure_api_key()
    client = LLMClient.create_llm_client(LLMConfig())
    if use_custom_pdf_analyzer():
        text = await client.analyze_office_document_files_custom(
            office_path_list, prompt, detail=detail
        )
    else:
        text = await client.analyze_office_document_files(office_path_list, prompt)
    print(text)


async def _analyze_multi_format_files(file_path_list: list[str], prompt: str, detail: str) -> None:
    from ai_chat_util.core.app import use_custom_pdf_analyzer
    from ai_chat_util.llm.llm_client import LLMClient
    from ai_chat_util.llm.llm_config import LLMConfig

    _ensure_api_key()
    client = LLMClient.create_llm_client(LLMConfig())
    if use_custom_pdf_analyzer():
        text = await client.analyze_multi_format_files_custom(file_path_list, prompt, detail)
    else:
        text = await client.analyze_multi_format_files(file_path_list, prompt, detail)
    print(text)


async def _run_mcp_server(mode: str, tools: str, port: int, server_log_level: str) -> None:
    from dotenv import load_dotenv
    from fastmcp import FastMCP

    from ai_chat_util.core.app import (
        run_chat,
        analyze_image_files,
        analyze_pdf_files,
        analyze_office_files,
        analyze_image_urls,
        analyze_pdf_urls,
        analyze_office_urls,
    )

    # MCPサーバ側が環境変数に依存するため、ここでも dotenv を読み込む
    load_dotenv()
    if server_log_level:
        os.environ["LOGLEVEL"] = server_log_level

    mcp = FastMCP()

    # tools 指定がある場合は、それのみ登録（vector-search-util と同様の挙動）
    if tools:
        tool_names = [t.strip() for t in tools.split(",") if t.strip()]
        namespace = {
            "run_chat": run_chat,
            "analyze_image_files": analyze_image_files,
            "analyze_pdf_files": analyze_pdf_files,
            "analyze_office_files": analyze_office_files,
            "analyze_image_urls": analyze_image_urls,
            "analyze_pdf_urls": analyze_pdf_urls,
            "analyze_office_urls": analyze_office_urls,
        }
        for name in tool_names:
            if name not in namespace:
                raise ValueError(f"Unknown tool name: {name}")
            mcp.tool()(namespace[name])
    else:
        # デフォルトツールを登録
        mcp.tool()(run_chat)
        mcp.tool()(analyze_image_files)
        mcp.tool()(analyze_pdf_files)
        mcp.tool()(analyze_office_files)
        mcp.tool()(analyze_image_urls)
        mcp.tool()(analyze_pdf_urls)
        mcp.tool()(analyze_office_urls)

    if mode == "stdio":
        await mcp.run_async()
        return

    if mode == "sse":
        await mcp.run_async(transport="sse", host="0.0.0.0", port=port)
        return

    if mode == "http":
        await mcp.run_async(transport="streamable-http", host="0.0.0.0", port=port)
        return

    raise ValueError(f"Unsupported mode: {mode}")


def _run_api_server(host: str, port: int) -> None:
    try:
        import uvicorn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "uvicorn が見つかりません。APIサーバを起動するには `pip install uvicorn` が必要です。"
        ) from e

    from dotenv import load_dotenv
    from ai_chat_util.api.api_server import app

    load_dotenv()
    uvicorn.run(app, host=host, port=port)


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
        await _run_chat(args.prompt)
        return

    if args.command == "analyze_image_files":
        _validate_non_empty(args.prompt, parser)
        await _analyze_image_files(args.image_path_list, args.prompt, args.detail)
        return

    if args.command == "analyze_pdf_files":
        _validate_non_empty(args.prompt, parser)
        await _analyze_pdf_files(args.pdf_path_list, args.prompt, args.detail)
        return

    if args.command == "analyze_office_files":
        _validate_non_empty(args.prompt, parser)
        await _analyze_office_files(args.office_path_list, args.prompt, args.detail)
        return

    if args.command == "analyze_multi_format_files":
        _validate_non_empty(args.prompt, parser)
        await _analyze_multi_format_files(args.file_path_list, args.prompt, args.detail)
        return

    if args.command == "mcp_server":
        await _run_mcp_server(args.mode, args.tools, args.port, args.server_log_level)
        return

    if args.command == "api_server":
        _run_api_server(args.host, args.port)
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
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    cli_main()
