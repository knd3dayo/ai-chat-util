from fastapi import APIRouter, FastAPI
from ai_chat_util.config.runtime import init_runtime

from ai_chat_util.core.resource_app import (
    use_custom_pdf_analyzer,
    get_completion_model,
    create_user_message,
    create_system_message,
    create_assistant_message,
    create_text_content,
    create_pdf_content_from_file,
    create_image_content,
    create_image_content_from_file,
    create_office_content_from_file,
    create_multi_format_contents_from_file,
)

from ai_chat_util.core.app import (
    run_chat,
    run_simple_chat,
    run_batch_chat,
    run_simple_batch_chat,
    run_batch_chat_from_excel,
)

from ai_chat_util.core.tool_app import (
    analyze_image_files,
    analyze_pdf_files,
    analyze_office_files,
    analyze_files,
    analyze_documents_data,
    analyze_image_urls,
    analyze_pdf_urls,
    analyze_office_urls,
    analyze_urls
)

router = APIRouter()

app = FastAPI()


@app.on_event("startup")
async def _startup_init_runtime() -> None:
    # Ensure config is loaded (uvicorn direct import path).
    init_runtime(None)

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
            "設定ファイル(config.yml)のパス。指定時は環境変数 AI_CHAT_UTIL_CONFIG にも反映し、"
            "後続処理に伝播します。未指定の場合は AI_CHAT_UTIL_CONFIG / カレント / プロジェクトルートの順で探索します。"
        ),
    )
    args = parser.parse_args()

    init_runtime(args.config or None)
    uvicorn.run(app, host="0.0.0.0", port=8000)
