from fastapi import APIRouter, FastAPI
from ai_chat_util.core.app import (
    use_custom_pdf_analyzer,
    run_chat,
    analyze_image_files,
    analyze_pdf_files,
    analyze_office_files,
    analyze_multi_format_files,
    analyze_image_urls,
    analyze_pdf_urls,
    analyze_office_urls,
    analyze_multi_format_urls
)
router = APIRouter()

app = FastAPI()

router.add_api_route(path="/use_custom_pdf_analyzer", endpoint=use_custom_pdf_analyzer, methods=["GET"])

# chat_utilのrun_chat_asyncを呼び出すラッパー関数を定義
router.add_api_route(path="/run_chat", endpoint=run_chat, methods=["POST"])

# 複数の画像の分析を行う
router.add_api_route(path="/analyze_image_files", endpoint=analyze_image_files, methods=["POST"])

# 複数のPDFの分析を行う
router.add_api_route(path="/analyze_pdf_files", endpoint=analyze_pdf_files, methods=["POST"])

# 複数のOfficeドキュメントの分析を行う
router.add_api_route(path="/analyze_office_files", endpoint=analyze_office_files, methods=["POST"])

# 複数の形式のドキュメントの分析を行う
router.add_api_route(path="/analyze_multi_format_files", endpoint=analyze_multi_format_files, methods=["POST"])

# 複数の画像の分析を行う URLから画像をダウンロードして分析する 
router.add_api_route(path="/analyze_image_urls", endpoint=analyze_image_urls, methods=["POST"])

# 複数のPDFの分析を行う URLからPDFをダウンロードして分析する
router.add_api_route(path="/analyze_pdf_urls", endpoint=analyze_pdf_urls, methods=["POST"])

# 複数のOfficeドキュメントの分析を行う URLからOfficeドキュメントをダウンロードして分析する
router.add_api_route(path="/analyze_office_urls", endpoint=analyze_office_urls, methods=["POST"])

# 複数の形式のドキュメントの分析を行う URLから形式のドキュメントをダウンロードして分析する
router.add_api_route(path="/analyze_multi_format_urls", endpoint=analyze_multi_format_urls, methods=["POST"])

# NOTE: include_router は、ルート定義が揃ってから呼ぶ（呼び出し時点の router.routes が登録される）
app.include_router(prefix="/api/ai_chat_util", router=router)

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
