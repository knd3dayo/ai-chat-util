from fastapi import APIRouter, FastAPI
from ai_chat_util.core.app import (
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
    run_chat,
    run_simple_chat,
    run_batch_chat,
    run_simple_batch_chat,
    run_batch_chat_from_excel,
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

router.add_api_route(path="/use_custom_pdf_analyzer", endpoint=use_custom_pdf_analyzer, methods=["GET"])

# chat_utilのget_completion_modelを呼び出すラッパー関数を定義
router.add_api_route(path="/get_completion_model", endpoint=get_completion_model, methods=["GET"])
# chat_utilのcreate_user_messageを呼び出すラッパー関数を定義
router.add_api_route(path="/create_user_message", endpoint=create_user_message, methods=["POST"])
# chat_utilのcreate_system_messageを呼び出すラッパー関数を定義
router.add_api_route(path="/create_system_message", endpoint=create_system_message, methods=["POST"])
# chat_utilのcreate_assistant_messageを呼び出すラッパー関数を定義
router.add_api_route(path="/create_assistant_message", endpoint=create_assistant_message, methods=["POST"])
# chat_utilのcreate_text_contentを呼び出すラッパー関数を定義
router.add_api_route(path="/create_text_content", endpoint=create_text_content, methods=["POST"])
# chat_utilのcreate_pdf_content_from_fileを呼び出すラッパー関数を定義
router.add_api_route(path="/create_pdf_content_from_file", endpoint=create_pdf_content_from_file, methods=["POST"])
# chat_utilのcreate_image_content_from_bytesを呼び出すラッパー関数を定義
router.add_api_route(path="/create_image_content_from_bytes", endpoint=create_image_content, methods=["POST"])
# chat_utilのcreate_image_content_from_fileを呼び出すラッパー関数を定義
router.add_api_route(path="/create_image_content_from_file", endpoint=create_image_content_from_file, methods=["POST"])
# chat_utilのcreate_office_content_from_fileを呼び出すラッパー関数を定義
router.add_api_route(path="/create_office_content_from_file", endpoint=create_office_content_from_file, methods=["POST"])
# chat_utilのcreate_multi_format_contents_from_fileを呼び出すラッパー関数を定義
router.add_api_route(path="/create_multi_format_contents_from_file", endpoint=create_multi_format_contents_from_file, methods=["POST"])
# chat_utilのrun_chat_asyncを呼び出すラッパー関数を定義
router.add_api_route(path="/run_chat", endpoint=run_chat, methods=["POST"])

# chat_utilのrun_simple_chatを呼び出すラッパー関数を定義
router.add_api_route(path="/run_simple_chat", endpoint=run_simple_chat, methods=["POST"])

# chat_utilのrun_simple_batch_chatを呼び出すラッパー関数を定義
router.add_api_route(path="/run_simple_batch_chat", endpoint=run_simple_batch_chat, methods=["POST"])  

# chat_utilのrun_batch_chatを呼び出すラッパー関数を定義
router.add_api_route(path="/run_batch_chat", endpoint=run_batch_chat, methods=["POST"])

# chat_utilのrun_batch_chat_from_excelを呼び出すラッパー関数を定義
router.add_api_route(path="/run_batch_chat_from_excel", endpoint=run_batch_chat_from_excel, methods=["POST"])

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
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
