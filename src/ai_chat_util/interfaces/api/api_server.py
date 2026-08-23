from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel, Field

from ai_chat_util.core.chat import create_llm_client
from ai_chat_util.core.chat.batch_client import BatchClient
from ai_chat_util.core.chat.model import ChatRequest, ChatResponse
from ai_chat_util.core.common.config.runtime import init_runtime
from ai_chat_util.core.request_headers import RequestHeaders, bind_current_request_headers

from ai_chat_util.core.resource_app import (
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

from ai_chat_util.core.analysis.analyze_pdf import (
    analyze_pdf_files,
    convert_office_files_to_pdf,
    convert_pdf_files_to_images,
    analyze_pdf_urls,
)
from ai_chat_util.core.analysis.analyze_image import (
    analyze_image_files,
    analyze_image_urls,
)
from ai_chat_util.core.analysis.analyze_office import (
    analyze_office_files,
    analyze_office_urls,
)
from ai_chat_util.core.analysis.analyze_office_xml import (
    analyze_office_xml_files,
)
from ai_chat_util.core.analysis.analyze_log import (
    extract_time_range_from_logfile,
    infer_log_header_pattern,
)
from ai_chat_util.core.analysis.analyze_file import (
    analyze_files,
    analyze_file_urls,
)


class SimpleChatRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text")


class BatchChatRequest(BaseModel):
    chat_requests: list[ChatRequest] = Field(..., description="List of chat requests")
    concurrency: int = Field(default=5, ge=1, description="Maximum concurrent tasks")


class BatchChatFromExcelRequest(BaseModel):
    prompt: str
    input_excel_path: str
    output_excel_path: str = "output.xlsx"
    content_column: str = "content"
    file_path_column: str = "file_path"
    output_column: str = "output"
    detail: str = "auto"
    concurrency: int = 16


router = APIRouter()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_runtime(None)
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def _capture_request_headers(request: Request, call_next):
    headers = {str(k).lower(): str(v) for k, v in request.headers.items()}
    with bind_current_request_headers(RequestHeaders.from_mapping(headers)):
        return await call_next(request)


async def run_chat(chat_request: ChatRequest) -> ChatResponse:
    llm_client = create_llm_client()
    return await llm_client.chat(chat_request)


async def run_simple_chat(request: SimpleChatRequest) -> str:
    llm_client = create_llm_client()
    return await llm_client.simple_chat(request.prompt)


async def run_batch_chat(request: BatchChatRequest) -> list[ChatResponse]:
    llm_batch_client = BatchClient()
    rows = await llm_batch_client.run_batch_chat(
        chat_requests=request.chat_requests,
        concurrency=request.concurrency,
    )
    return [response for _, response in rows]


async def run_batch_chat_from_excel(request: BatchChatFromExcelRequest) -> dict[str, str]:
    llm_batch_client = BatchClient()
    await llm_batch_client.run_batch_chat_from_excel(
        prompt=request.prompt,
        input_excel_path=request.input_excel_path,
        output_excel_path=request.output_excel_path,
        content_column=request.content_column,
        file_path_column=request.file_path_column,
        output_column=request.output_column,
        detail=request.detail,
        concurrency=request.concurrency,
    )
    return {"output_excel_path": request.output_excel_path}


router.add_api_route(path="/analyze_image_files", endpoint=analyze_image_files, methods=["POST"])
router.add_api_route(path="/analyze_pdf_files", endpoint=analyze_pdf_files, methods=["POST"])
router.add_api_route(path="/analyze_office_files", endpoint=analyze_office_files, methods=["POST"])
router.add_api_route(path="/analyze_office_xml_files", endpoint=analyze_office_xml_files, methods=["POST"])
router.add_api_route(path="/analyze_files", endpoint=analyze_files, methods=["POST"])
router.add_api_route(path="/analyze_image_urls", endpoint=analyze_image_urls, methods=["POST"])
router.add_api_route(path="/analyze_pdf_urls", endpoint=analyze_pdf_urls, methods=["POST"])
router.add_api_route(path="/analyze_office_urls", endpoint=analyze_office_urls, methods=["POST"])
router.add_api_route(path="/analyze_file_urls", endpoint=analyze_file_urls, methods=["POST"])

router.add_api_route(path="/convert_office_files_to_pdf", endpoint=convert_office_files_to_pdf, methods=["POST"])
router.add_api_route(path="/convert_pdf_files_to_images", endpoint=convert_pdf_files_to_images, methods=["POST"])
router.add_api_route(path="/extract_time_range_from_logfile", endpoint=extract_time_range_from_logfile, methods=["POST"])
router.add_api_route(path="/infer_log_header_pattern", endpoint=infer_log_header_pattern, methods=["POST"])

router.add_api_route(path="/use_custom_pdf_analyzer", endpoint=use_custom_pdf_analyzer, methods=["GET"])
router.add_api_route(path="/get_completion_model", endpoint=get_completion_model, methods=["GET"])
router.add_api_route(path="/get_loaded_config_info", endpoint=get_loaded_config_info, methods=["GET"])

router.add_api_route(path="/chat", endpoint=run_chat, methods=["POST"])
router.add_api_route(path="/simple_chat", endpoint=run_simple_chat, methods=["POST"])
router.add_api_route(path="/batch_chat", endpoint=run_batch_chat, methods=["POST"])
router.add_api_route(path="/batch_chat_from_excel", endpoint=run_batch_chat_from_excel, methods=["POST"])

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