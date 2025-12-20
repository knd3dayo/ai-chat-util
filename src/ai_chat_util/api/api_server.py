from typing import Annotated, Any
import os, tempfile
import requests
from pydantic import Field, BaseModel
from ai_chat_util.llm.model import ChatRequestContext, ChatHistory, ChatResponse
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig
from fastapi import FastAPI
from ai_chat_util.core.app import (
    use_custom_pdf_analyzer,
    run_chat,
    analyze_image_files,
    analyze_pdf_files,
    analyze_office_files,
    analyze_image_urls,
    analyze_pdf_urls,
    analyze_office_urls
)

app = FastAPI()

app.add_api_route(path="/use_custom_pdf_analyzer", endpoint=use_custom_pdf_analyzer, methods=["GET"])

# chat_utilのrun_chat_asyncを呼び出すラッパー関数を定義
app.add_api_route(path="/run_chat", endpoint=run_chat, methods=["POST"])

# 複数の画像の分析を行う
app.add_api_route(path="/analyze_image_files", endpoint=analyze_image_files, methods=["POST"])

# 複数の画像の分析を行う
app.add_api_route(path="/analyze_image_files", endpoint=analyze_image_files, methods=["POST"])

# 複数のPDFの分析を行う
app.add_api_route(path="/analyze_pdf_files", endpoint=analyze_pdf_files, methods=["POST"])

# 複数のOfficeドキュメントの分析を行う
app.add_api_route(path="/analyze_office_files", endpoint=analyze_office_files, methods=["POST"])

# 複数の画像の分析を行う URLから画像をダウンロードして分析する 
app.add_api_route(path="/analyze_image_urls", endpoint=analyze_image_urls, methods=["POST"])
# 複数のPDFの分析を行う URLからPDFをダウンロードして分析する
app.add_api_route(path="/analyze_pdf_urls", endpoint=analyze_pdf_urls, methods=["POST"])
# 複数のOfficeドキュメントの分析を行う URLからOfficeドキュメントをダウンロードして分析する
app.add_api_route(path="/analyze_office_urls", endpoint=analyze_office_urls, methods=["POST"])

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
