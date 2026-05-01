from pathlib import Path
from typing import Annotated, Optional

from pydantic import Field

from ai_chat_util.ai_chat_util_base.core.common.config.runtime import get_runtime_config, get_runtime_config_path
from ai_chat_util.ai_chat_util_base.util.analyze_file_util.file_util import FileUtil
from ai_chat_util.ai_chat_util_base.core.analyze_file_util.model import (
    FileServerListResponse,
    FileServerProvider,
    FileServerRootInfo,
    FileServerRootListResponse,
    FileUtilDocument,
    FileUtilDocumentType,
)
from ai_chat_util.ai_chat_util_base.analyze_file_util.util.document_text_util import DocumentTextUtil
from ai_chat_util.ai_chat_util_base.analyze_file_util.util.excel_util import ExcelUtil
from ai_chat_util.ai_chat_util_base.util.analyze_file_util.file_server_util import FileServerUtil
from ai_chat_util.ai_chat_util_base.analyze_file_util.util.zip_util import ZipUtil

def tool_timeout_seconds() -> float:
    runtime_config = get_runtime_config()
    tool_timeout_cfg = getattr(runtime_config.features, "mcp_tool_timeout_seconds", None)
    try:
        timeout = (
            float(tool_timeout_cfg)
            if tool_timeout_cfg is not None
            else float(runtime_config.llm.timeout_seconds)
        )
    except (TypeError, ValueError):
        timeout = float(runtime_config.llm.timeout_seconds)
    if timeout <= 0:
        timeout = float(runtime_config.llm.timeout_seconds)
    return timeout

def tool_timeout_retries() -> int:
    runtime_config = get_runtime_config()
    try:
        retries_raw = int(getattr(runtime_config.features, "mcp_tool_timeout_retries", 1) or 0)
    except (TypeError, ValueError):
        retries_raw = 1
    return max(0, min(5, retries_raw))


def _get_network_download_options() -> tuple[bool, str | None]:
    cfg = get_runtime_config()
    return cfg.network.requests_verify, cfg.network.ca_bundle


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
    response = DocumentTextUtil.extract_base64_to_text(extension, base64_data)
    return response


async def extract_text_from_file(
    file_path: Annotated[str, Field(description="Path to the file to extract text from")]
    ) -> Annotated[str, Field(description="Extracted text from the file")]:
    """
    This function extracts text from a file at the specified path.
    """
    return DocumentTextUtil.extract_text_from_path(file_path)

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


def _resolve_file_server_root_path(root_path: str) -> Path:
    candidate = Path(root_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    config_dir = get_runtime_config_path().parent
    resolved_from_config_dir = (config_dir / candidate).resolve()
    if resolved_from_config_dir.exists():
        return resolved_from_config_dir

    resolved_from_cwd = (Path.cwd() / candidate).resolve()
    if resolved_from_cwd.exists():
        return resolved_from_cwd

    return resolved_from_config_dir


def _select_file_server_root(*, provider: FileServerProvider | None, root_name: str | None):
    runtime_config = get_runtime_config()
    file_server = runtime_config.file_server

    if not file_server.enabled:
        raise ValueError("file_server.enabled が false のため、この機能は利用できません")
    if not file_server.allowed_roots:
        raise ValueError("file_server.allowed_roots が未設定です")

    selected_root = None
    if root_name:
        for allowed_root in file_server.allowed_roots:
            if allowed_root.name == root_name:
                selected_root = allowed_root
                break
        if selected_root is None:
            raise ValueError(f"指定された root_name が見つかりません: {root_name}")
    elif provider is not None:
        for allowed_root in file_server.allowed_roots:
            if allowed_root.provider == provider.value:
                selected_root = allowed_root
                break
    elif file_server.default_root:
        for allowed_root in file_server.allowed_roots:
            if allowed_root.name == file_server.default_root:
                selected_root = allowed_root
                break
    else:
        selected_root = file_server.allowed_roots[0]

    if selected_root is None:
        provider_name = provider.value if provider is not None else file_server.default_provider
        raise ValueError(f"利用可能な {provider_name} ルートが設定されていません")
    if provider is not None and selected_root.provider != provider.value:
        raise ValueError("provider と root_name の組み合わせが一致しません")
    return selected_root, file_server


async def list_file_server_entries(
    provider: Annotated[Optional[FileServerProvider], Field(description="Storage provider to use. Omit to use default_root")] = None,
    root_name: Annotated[Optional[str], Field(description="Configured root name to browse")] = None,
    path: Annotated[str, Field(description="Path relative to the configured root")] = ".",
    recursive: Annotated[bool, Field(description="Whether to return child directories recursively")] = False,
    max_depth: Annotated[Optional[int], Field(description="Maximum child depth when recursive is true")] = None,
    include_hidden: Annotated[Optional[bool], Field(description="Whether to include dotfiles and hidden entries")] = None,
    include_mime: Annotated[Optional[bool], Field(description="Whether to detect MIME type for files")] = None,
) -> FileServerListResponse:
    selected_root, file_server = _select_file_server_root(provider=provider, root_name=root_name)

    effective_max_depth = file_server.max_depth if max_depth is None else max_depth
    if effective_max_depth < 0:
        raise ValueError("max_depth は 0 以上である必要があります")
    effective_include_hidden = file_server.include_hidden_default if include_hidden is None else include_hidden
    effective_include_mime = file_server.include_mime_default if include_mime is None else include_mime

    if selected_root.provider == FileServerProvider.LOCAL.value:
        return FileServerUtil.list_local_entries(
            root=selected_root,
            root_path=_resolve_file_server_root_path(selected_root.path),
            relative_path=path,
            recursive=recursive,
            max_depth=effective_max_depth,
            max_entries=file_server.max_entries,
            include_hidden=effective_include_hidden,
            include_mime=effective_include_mime,
            follow_symlinks=file_server.follow_symlinks,
        )

    return FileServerUtil.list_smb_entries(
        root=selected_root,
        smb=file_server.smb,
        relative_path=path,
        recursive=recursive,
        max_depth=effective_max_depth,
        max_entries=file_server.max_entries,
        include_hidden=effective_include_hidden,
    )


async def list_file_server_roots() -> FileServerRootListResponse:
    runtime_config = get_runtime_config()
    file_server = runtime_config.file_server
    return FileServerRootListResponse(
        enabled=file_server.enabled,
        default_provider=FileServerProvider(file_server.default_provider),
        default_root=file_server.default_root,
        roots=[
            FileServerRootInfo(
                name=root.name,
                provider=FileServerProvider(root.provider),
                path=root.path,
                description=root.description,
                is_default=root.name == file_server.default_root,
            )
            for root in file_server.allowed_roots
        ],
    )
