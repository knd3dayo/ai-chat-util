from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException

from ai_chat_util.ai_chat_util_base.core.common.config.runtime import init_runtime
from ai_chat_util.ai_chat_util_base.core.analyze_file_util.model import FileServerProvider

from ai_chat_util.ai_chat_util_base.app.analyze_file_util.core.base import (
    list_file_server_roots,
    list_file_server_entries,
    get_document_type,
    get_mime_type,
    get_sheet_names,
    extract_excel_sheet,
    extract_text_from_file,
    extract_base64_to_text,
    list_zip_contents,
    extract_zip,
    create_zip,
    export_data_to_excel,
    import_data_from_excel,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_runtime(None)
    yield


app = FastAPI(lifespan=lifespan)
router = APIRouter()


async def _list_file_server_entries_api(
    provider: Optional[FileServerProvider] = None,
    root_name: Optional[str] = None,
    path: str = ".",
    recursive: bool = False,
    max_depth: Optional[int] = None,
    include_hidden: Optional[bool] = None,
    include_mime: Optional[bool] = None,
):
    try:
        return await list_file_server_entries(
            provider=provider,
            root_name=root_name,
            path=path,
            recursive=recursive,
            max_depth=max_depth,
            include_hidden=include_hidden,
            include_mime=include_mime,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _list_file_server_roots_api():
    try:
        return await list_file_server_roots()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# get_document_type
router.add_api_route(path='/get_document_type', endpoint=get_document_type, methods=['GET'])
# get_mime_type
router.add_api_route(path='/get_mime_type', endpoint=get_mime_type, methods=['GET'])
 
# get_sheet_names
router.add_api_route(path='/get_sheet_names', endpoint=get_sheet_names, methods=['GET'])
# extract_excel_sheet
router.add_api_route(path='/extract_excel_sheet', endpoint=extract_excel_sheet, methods=['POST'])

# extract_text_from_file
router.add_api_route(path='/extract_text_from_file', endpoint=extract_text_from_file, methods=['POST'])

# extract_base64_to_text
router.add_api_route(path='/extract_base64_to_text', endpoint=extract_base64_to_text, methods=['GET'])

# ZIPファイルの内容をリストする関数
router.add_api_route(path='/list_zip_contents', endpoint=list_zip_contents, methods=['GET'])

# ZIPファイルを展開する関数
router.add_api_route(path='/extract_zip', endpoint=extract_zip, methods=['POST'])

# ZIPファイルを作成する関数
router.add_api_route(path='/create_zip', endpoint=create_zip, methods=['POST'])

# export_data_to_excel
router.add_api_route(path='/export_data_to_excel', endpoint=export_data_to_excel, methods=['POST'])

# import_data_from_excel
router.add_api_route(path='/import_data_from_excel', endpoint=import_data_from_excel, methods=['GET'])

# file server directory listing
router.add_api_route(path='/list_file_server_roots', endpoint=_list_file_server_roots_api, methods=['GET'])
router.add_api_route(path='/list_file_server_entries', endpoint=_list_file_server_entries_api, methods=['GET'])

app.include_router(router, prefix="/api/file_util")
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="file_util API server")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="ai-chat-util-config.yml のパス。未指定時は AI_CHAT_UTIL_CONFIG / カレント / プロジェクトルートから探索します。",
    )
    args = parser.parse_args()

    init_runtime(args.config or None)
    uvicorn.run(app, host="0.0.0.0", port=8000)