from fastapi import FastAPI, APIRouter

from file_util.core.app import (
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
app = FastAPI()
router = APIRouter()
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

app.include_router(router, prefix="/api/file_util")
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)