from typing import Annotated, Optional, Literal
from pydantic import Field
from file_util.util.file_util import FileUtil
from file_util.model import FileUtilDocumentType, FileUtilDocument
from file_util.util.excel_util import ExcelUtil
from file_util.util.zip_util import ZipUtil


async def get_document_type(
    file_path: Annotated[str, Field(description="Path to the file to get types for")]
    ) -> Annotated[FileUtilDocumentType, Field(description="Type of the document. None if undetectable")]:
    """
    This function gets the type of a file at the specified path.
    """
    document_type = FileUtilDocument.from_file(document_path=file_path)
    return document_type.get_document_type()

async def get_mime_type(
    file_path: Annotated[str, Field(description="Path to the file to get MIME type for")]
    ) -> Annotated[Optional[str], Field(description="MIME type of the file. None if undetectable")]:
    """
    This function gets the MIME type of a file at the specified path.
    """
    document_type = FileUtilDocument.from_file(document_path=file_path)
    return document_type.mime_type

# get_sheet_names
async def get_sheet_names(
    file_path: Annotated[str, Field(description="Path to the Excel file to get sheet names for")]
    ) -> Annotated[list[str], Field(description="List of sheet names in the Excel file")]:
    """
    This function gets the sheet names of an Excel file at the specified path.
    """
    response = ExcelUtil.get_sheet_names(file_path)
    return response

# extract_excel_sheet
async def extract_excel_sheet(
    file_path: Annotated[str, Field(description="Path to the Excel file to extract text from")],
    sheet_name: Annotated[str, Field(description="Name of the sheet to extract text from")]
    ) -> Annotated[str, Field(description="Extracted text from the specified Excel sheet")]:
    """
    This function extracts text from a specified sheet in an Excel file.
    """
    response = ExcelUtil.extract_text_from_sheet(file_path, sheet_name)
    return response

# extract_base64_to_text
async def extract_base64_to_text(
    extension: Annotated[str, Field(description="File extension of the base64 data")],
    base64_data: Annotated[str, Field(description="Base64 encoded data to extract text from")]
    ) -> Annotated[str, Field(description="Extracted text from the base64 data")]:
    """
    This function extracts text from base64 encoded data with a specified file extension.
    """
    response = await FileUtil.extract_base64_to_text(extension, base64_data)
    return response


async def extract_text_from_file(
    file_path: Annotated[str, Field(description="Path to the file to extract text from")]
    ) -> Annotated[str, Field(description="Extracted text from the file")]:
    """
    This function extracts text from a file at the specified path.
    """
    return await FileUtil.extract_text_from_file_async(file_path)

# ZIPファイルの内容をリストする関数
async def list_zip_contents(
    file_path: Annotated[str, Field(description="Path to the ZIP file to list contents from. **Absolute path required**")]
    ) -> Annotated[list[str], Field(description="List of file names in the ZIP archive")]:
    """
    This function lists the contents of a ZIP file at the specified path.
    """
    return ZipUtil.list_zip_contents(file_path)

# ZIPファイルを展開する関数
async def extract_zip(
    file_path: Annotated[str, Field(description="Path to the ZIP file to extract. **Absolute path required**")],
    extract_to: Annotated[str, Field(description="Directory to extract the ZIP contents to. **Absolute path required**")],
    password: Annotated[Optional[str], Field(description="Password for the ZIP file, if any")] = None
    ) -> Annotated[bool, Field(description="True if extraction was successful")]:

    """
    This function extracts a ZIP file at the specified path.
    """
    return ZipUtil.extract_zip(file_path, extract_to, password)

# ZIPファイルを作成する関数
async def create_zip(
    file_paths: Annotated[list[str], Field(description="List of file or directory paths to include in the ZIP. **Absolute paths required**")],
    output_zip: Annotated[str, Field(description="Path to the output ZIP file. **Absolute path required**")],
    password: Annotated[Optional[str], Field(description="Password for the ZIP file, if any")] = None
    ) -> Annotated[bool, Field(description="True if ZIP creation was successful")]:

    """
    This function creates a ZIP file at the specified path.
    """
    return ZipUtil.create_zip(file_paths, output_zip, password)

# export_data_to_excel
async def export_data_to_excel(
    data: Annotated[dict[str, list], Field(description="Data to export to Excel, with keys as column headers and values as lists of column data")],
    output_file: Annotated[str, Field(description="Path to the output Excel file")],
    sheet_name: Annotated[Optional[str], Field(description="Name of the sheet to create in the Excel file")] = "Sheet1"
    ) -> Annotated[bool, Field(description="True if data export was successful")]:

    """
    This function exports data to an Excel file at the specified path.
    """
    ExcelUtil.export_data_to_excel(data, output_file, sheet_name)
    return True

# import_data_from_excel
async def import_data_from_excel(
    input_file: Annotated[str, Field(description="Path to the Excel file to import data from")],
    sheet_name: Annotated[Optional[str], Field(description="Name of the sheet to import data from")] = "Sheet1"
    ) -> Annotated[dict[str, list], Field(description="Imported data from the Excel file, with keys as column headers and values as lists of column data")]:

    """
    This function imports data from an Excel file at the specified path.
    """
    data = ExcelUtil.import_data_from_excel(input_file, sheet_name)
    return data
