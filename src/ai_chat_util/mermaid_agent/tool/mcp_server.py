
import asyncio
from typing import Annotated, Optional
from pydantic import Field
from fastmcp import FastMCP
from ai_chat_util.mermaid_agent.tool.file_tools import *


mcp = FastMCP("mermaid_workflow_sample") #type :ignore

@mcp.tool()
def list_files_mcp(
    directory_path: Annotated[str, Field(description="一覧表示するディレクトリの絶対パス。例: /path/to/directory")],
    filter: Annotated[Optional[str], "A filter string to match against file names. An asterisk matches any string."] = None
) -> Annotated[list[FileModel], Field(description="ディレクトリ内のファイルとサブディレクトリの一覧")]:

    """
    指定したディレクトリ内のファイルとサブディレクトリの一覧を取得します。
    filter: ワイルドカード（*）を使用してファイル名に一致するフィルタ文字列。
    """
    files = list_files(directory_path, filter)
    return files
        
@mcp.tool()
def read_file_mcp(
    file_path: Annotated[str, Field(description="読み込むファイルの絶対パス。例: /path/to/file.txt")],
    start_line: Annotated[Optional[int], Field(description="読み込みを開始する行番号（1から始まる）。Noneの場合はファイルの先頭から読み込みます。")] = None,
    end_line: Annotated[Optional[int], Field(description="読み込みを終了する行番号（1から始まる）。Noneの場合はファイルの最後まで読み込みます。")] = None,
) -> Annotated[str, Field(description="ファイルの内容")]:
    """
    指定したファイルの内容を読み込み、指定された行数分だけ返します。
    """
    content = read_file(file_path, start_line, end_line)
    return content

@mcp.tool()
def write_file_mcp(
    file_path: Annotated[str, Field(description="書き込むファイルの絶対パス。例: /path/to/file.txt")],
    content: Annotated[str, Field(description="書き込む内容")],
    append: Annotated[bool, Field(description="Trueの場合は追記、Falseの場合は上書き")] = False
) -> Annotated[bool, Field(description="書き込みが成功したかどうか")]:
    """
    指定したファイルに内容を書き込みます。appendがTrueの場合は追記、Falseの場合は上書きします。
    """
    result = write_file(file_path, content, append)
    return result

@mcp.tool()
def delete_file_mcp(
    file_path: Annotated[str, Field(description="削除するファイルの絶対パス。例: /path/to/file.txt")]
) -> Annotated[bool, Field(description="削除が成功したかどうか")]:
    """
    指定したファイルを削除します。
    """
    result = delete_file(file_path)
    return result

@mcp.tool()
def search_in_file_mcp(
    file_path: Annotated[str, Field(description="検索するファイルの絶対パス。例: /path/to/file.txt")],
    search_string: Annotated[str, Field(description="検索する文字列")],
    case_sensitive: Annotated[bool, Field(description="大文字と小文字を区別するかどうか")] = False
) -> Annotated[list[FileLineModel], Field(description="検索文字列が含まれる行のリスト")]:
    """
    指定したファイル内で検索文字列を検索し、その行をリストで返します。
    """
    lines = search_in_file(file_path, search_string, case_sensitive)
    return lines

@mcp.tool()
def create_directory_mcp(
    directory_path: Annotated[str, Field(description="作成するディレクトリの絶対パス。例: /path/to/directory")]
) -> Annotated[bool, Field(description="ディレクトリの作成が成功したかどうか")]:
    """
    指定したパスにディレクトリを作成します。
    """
    result = create_directory(directory_path)
    return result

@mcp.tool()
def delete_directory_mcp(
    directory_path: Annotated[str, Field(description="削除するディレクトリの絶対パス。例: /path/to/directory")]
) -> Annotated[bool, Field(description="ディレクトリの削除が成功したかどうか")]:
    """
    指定したディレクトリを削除します。
    """
    result = delete_directory(directory_path)
    return result

async def main():
    await mcp.run_async()


if __name__ == "__main__":
    asyncio.run(main())
