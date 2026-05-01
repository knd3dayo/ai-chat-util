from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from ai_chat_util.common.config.runtime import init_runtime
from ai_chat_util.ai_chat_util_base.request_headers import RequestHeaders, bind_current_request_headers

from ai_chat_util.ai_chat_util_agent.core.resource_app import (
    use_custom_pdf_analyzer,
    get_completion_model,
    get_loaded_config_info,
    create_user_message,
    create_system_message,
    create_assistant_message,
    create_text_content,
    create_pdf_content,
    create_pdf_content_from_file,
    create_image_content,
    create_image_content_from_file,
    create_office_content,
    create_office_content_from_file,
    create_multi_format_contents_from_file,
)

from ai_chat_util.ai_chat_util_agent.core.app import (
    run_chat,
    run_agent_chat,
    run_deepagent_chat,
    run_simple_chat,
    run_batch_chat,
    run_agent_batch_chat,
    run_deepagent_batch_chat,
    run_simple_batch_chat,
    run_batch_chat_from_excel,
    run_agent_batch_chat_from_excel,
    run_deepagent_batch_chat_from_excel,
)

from ai_chat_util.ai_chat_util_base.analyze_pdf_util.core import (
    analyze_image_files,
    analyze_pdf_files,
    analyze_office_files,
    analyze_files,
    convert_office_files_to_pdf,
    convert_pdf_files_to_images,
    detect_log_format_and_search,
    infer_log_header_pattern,
    extract_log_time_range,
    analyze_documents_data,
    analyze_image_urls,
    analyze_pdf_urls,
    analyze_office_urls,
    analyze_urls
)

router = APIRouter()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Ensure config is loaded (uvicorn direct import path).
    init_runtime(None)
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def _capture_request_headers(request: Request, call_next):
    headers = {str(k).lower(): str(v) for k, v in request.headers.items()}
    with bind_current_request_headers(RequestHeaders.from_mapping(headers)):
        return await call_next(request)

# 複数の画像の分析を行う
router.add_api_route(path="/analyze_image_files", endpoint=analyze_image_files, methods=["POST"])

# 複数のPDFの分析を行う
router.add_api_route(path="/analyze_pdf_files", endpoint=analyze_pdf_files, methods=["POST"])

# 複数のOfficeドキュメントの分析を行う
router.add_api_route(path="/analyze_office_files", endpoint=analyze_office_files, methods=["POST"])

# 複数の形式のドキュメントの分析を行う
router.add_api_route(path="/analyze_files", endpoint=analyze_files, methods=["POST"])

# 複数の形式のドキュメントの分析を行う
router.add_api_route(path="/analyze_documents_data", endpoint=analyze_documents_data, methods=["POST"])

# 複数の画像の分析を行う URLから画像をダウンロードして分析する 
router.add_api_route(path="/analyze_image_urls", endpoint=analyze_image_urls, methods=["POST"])

# 複数のPDFの分析を行う URLからPDFをダウンロードして分析する
router.add_api_route(path="/analyze_pdf_urls", endpoint=analyze_pdf_urls, methods=["POST"])

# 複数のOfficeドキュメントの分析を行う URLからOfficeドキュメントをダウンロードして分析する
router.add_api_route(path="/analyze_office_urls", endpoint=analyze_office_urls, methods=["POST"])

# 複数の形式のドキュメントの分析を行う URLから形式のドキュメントをダウンロードして分析する
router.add_api_route(path="/analyze_urls", endpoint=analyze_urls, methods=["POST"])

# ドキュメント変換ツール
router.add_api_route(path="/convert_office_files_to_pdf", endpoint=convert_office_files_to_pdf, methods=["POST"])
router.add_api_route(path="/convert_pdf_files_to_images", endpoint=convert_pdf_files_to_images, methods=["POST"])
router.add_api_route(path="/detect_log_format_and_search", endpoint=detect_log_format_and_search, methods=["POST"])
router.add_api_route(path="/infer_log_header_pattern", endpoint=infer_log_header_pattern, methods=["POST"])
router.add_api_route(path="/extract_log_time_range", endpoint=extract_log_time_range, methods=["POST"])

# resource_app の関数をデフォルトで公開する
router.add_api_route(path="/use_custom_pdf_analyzer", endpoint=use_custom_pdf_analyzer, methods=["GET"])
router.add_api_route(path="/get_completion_model", endpoint=get_completion_model, methods=["GET"])
router.add_api_route(path="/get_loaded_config_info", endpoint=get_loaded_config_info, methods=["GET"])
router.add_api_route(
    path="/chat",
    endpoint=run_chat,
    methods=["POST"],
    summary="Run chat",
    description=(
        "Run a chat request via the standard LLM client."
    ),
)
router.add_api_route(
    path="/agent_chat",
    endpoint=run_agent_chat,
    methods=["POST"],
    summary="Run agent chat",
    description=(
        "Run a chat request via the MCP-backed agent client. "
        "If chat_request_context.workflow_file_path is provided, the same endpoint may route to the workflow backend. "
        "The response may return status='paused' with hitl and trace_id, "
        "and the client can resume by sending another ChatRequest with the same trace_id."
    ),
)
router.add_api_route(
    path="/run_deepagent_chat",
    endpoint=run_deepagent_chat,
    methods=["POST"],
    summary="Run DeepAgent chat",
    description=(
        "Run a chat request via the MCP-backed DeepAgent client. "
        "The response may return status='paused' with hitl and trace_id, "
        "and the client can resume by sending another ChatRequest with the same trace_id."
    ),
)
router.add_api_route(path="/batch_chat", endpoint=run_batch_chat, methods=["POST"])
router.add_api_route(path="/agent_batch_chat", endpoint=run_agent_batch_chat, methods=["POST"])
router.add_api_route(path="/run_deepagent_batch_chat", endpoint=run_deepagent_batch_chat, methods=["POST"])
router.add_api_route(path="/deepagent_batch_chat", endpoint=run_deepagent_batch_chat, methods=["POST"])
router.add_api_route(path="/batch_chat_from_excel", endpoint=run_batch_chat_from_excel, methods=["POST"])
router.add_api_route(path="/agent_batch_chat_from_excel", endpoint=run_agent_batch_chat_from_excel, methods=["POST"])
router.add_api_route(path="/run_deepagent_batch_chat_from_excel", endpoint=run_deepagent_batch_chat_from_excel, methods=["POST"])
router.add_api_route(path="/deepagent_batch_chat_from_excel", endpoint=run_deepagent_batch_chat_from_excel, methods=["POST"])
router.add_api_route(path="/create_user_message", endpoint=create_user_message, methods=["POST"])
router.add_api_route(path="/create_assistant_message", endpoint=create_assistant_message, methods=["POST"])
router.add_api_route(path="/create_system_message", endpoint=create_system_message, methods=["POST"])
router.add_api_route(path="/create_text_content", endpoint=create_text_content, methods=["POST"])
router.add_api_route(path="/create_image_content", endpoint=create_image_content, methods=["POST"])
router.add_api_route(path="/create_image_content_from_file", endpoint=create_image_content_from_file, methods=["POST"])
router.add_api_route(path="/create_pdf_content", endpoint=create_pdf_content, methods=["POST"])
router.add_api_route(path="/create_pdf_content_from_file", endpoint=create_pdf_content_from_file, methods=["POST"])
router.add_api_route(path="/create_office_content", endpoint=create_office_content, methods=["POST"])
router.add_api_route(path="/create_office_content_from_file", endpoint=create_office_content_from_file, methods=["POST"])
router.add_api_route(path="/create_multi_format_contents_from_file", endpoint=create_multi_format_contents_from_file, methods=["POST"])

# NOTE: include_router は、ルート定義が揃ってから呼ぶ（呼び出し時点の router.routes が登録される）
app.include_router(prefix="/api/ai_chat_util", router=router)

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="ai_chat_util API server")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "設定ファイル(ai-chat-util-config.yml)のパス。指定時は環境変数 AI_CHAT_UTIL_CONFIG にも反映し、"
            "後続処理に伝播します。未指定の場合は AI_CHAT_UTIL_CONFIG / カレント / プロジェクトルートの順で探索します。"
        ),
    )
    args = parser.parse_args()

    init_runtime(args.config or None)
    uvicorn.run(app, host="0.0.0.0", port=8000)
