import os, fnmatch
import re
from typing import List, Optional, Annotated, Callable
from pydantic import BaseModel, Field
from datetime import datetime
import ai_chat_util.mermaid_agent.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

class FileModel(BaseModel):
    """
    Represents a file or directory with its metadata.
    """
    name: str = Field(..., description="The name of the file or directory")
    path: str = Field(..., description="The absolute path to the file")
    is_file: bool = Field(..., description="Indicates if the path is a file")
    is_directory: bool = Field(..., description="Indicates if the path is a directory")
    size: Optional[int] = Field(None, description="The size of the file in bytes, if loaded")
    last_modified: Optional[datetime] = Field(None, description="The last modified timestamp of the file, if loaded")

class FileLineModel(BaseModel):
    """
    Represents a line in a file with its metadata.
    """
    name: str = Field(..., description="The name of the file")
    path: str = Field(..., description="The absolute path to the file")
    line_number: int = Field(..., description="The line number in the file (1-based)")
    content: str = Field(..., description="The content of the line")


def check_allow_outside_modification(file_path: str):
    """
    Checks if outside file modification is allowed based on environment variable.
    Args:
    file_path (str): The path to the file to check.
    """
    allow_outside_modification = os.getenv("ALLOW_OUTSIDE_MODIFICATIONS", "false").lower()
    if not allow_outside_modification == "true":
        file_path = os.path.abspath(file_path)
        allowed_directory = os.path.abspath(os.getenv("PWD", "."))
        if not file_path.startswith(allowed_directory):
            logger.warning(f"Outside file modification is not allowed. Attempted to modify {file_path}.")
            raise ValueError(f"Outside file modification is not allowed. Attempted to modify {file_path}.")
            # return False

def list_files(
    directory_path: Annotated[str, "The path to the directory to list files from."] = ".", 
    filter: Annotated[Optional[str], "A filter string to match against file names. An asterisk matches any string."] = None
    ) -> List[FileModel]:
    """
    Lists files and directories in the specified directory path.

    Args:
    directory_path (str): The path to the directory to list files from.
    filter (Optional[str]): A filter string to match against file names. An asterisk matches any string.

    Returns:
    List[FileModel]: A list of FileModel instances representing files and directories.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"The path {directory_path} is not a valid directory.")

    file_models = []
    for entry in os.scandir(directory_path):
        if filter and not fnmatch.fnmatch(entry.name, filter):
            continue
        file_model = FileModel(
            name=entry.name,
            path=entry.path,
            is_file=entry.is_file(),
            is_directory=entry.is_dir(),
            size=None,
            last_modified=None,
        )

        if entry.is_file():
            file_model.size = os.path.getsize(entry.path)
            file_model.last_modified = datetime.fromtimestamp(os.path.getmtime(entry.path))

        file_models.append(file_model)

    return file_models

def read_file(
    file_path: Annotated[str, "The path to the file to read content from."], 
    start_line: Annotated[Optional[int], "The line number to start reading from (1-based). If None, starts from the beginning."] = None,
    end_line: Annotated[Optional[int], "The line number to stop reading at (1-based, inclusive). If None, reads to the end of the file."] = None,
) -> str:
    """
    Reads the content of a file.
    Args:
    file_path (str): The path to the file to read content from.
    start_line (Optional[int]): The line number to start reading from (1-based). If None, starts from the beginning.
    end_line (Optional[int]): The line number to stop reading at (1-based, inclusive). If None, reads to the end of the file.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"The path {file_path} is not a valid file.")

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Apply start_line and end_line filtering
    if start_line is not None:
        lines = lines[start_line - 1:]  # Convert to 0-based index
    if end_line is not None:
        lines = lines[:end_line]  # Slice up to end_line (inclusive)

    return '\n'.join(lines)

def search_in_file(
    file_path: Annotated[str, "The path to the file to search in."], 
    search_string: Annotated[str, "The string to search for in the file."], 
    case_sensitive: Annotated[bool, "Whether the search should be case sensitive."] = True
) -> List[FileLineModel]:
    """
    Searches for a string in a file and returns matching lines with their metadata.

    Args:
    file_path (str): The path to the file to search in.
    search_string (str): The string to search for in the file.
    case_sensitive (bool): Whether the search should be case sensitive.

    Returns:
    List[FileLineModel]: A list of FileLineModel instances representing matching lines.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"The path {file_path} is not a valid file.")

    matching_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            original_line = line
            if not case_sensitive:
                line = line.lower()
                search_string = search_string.lower()
            if re.search(re.escape(search_string), line):
                logger.info(f"Match found in {file_path} on line {line_number}: {original_line.strip()}")
                matching_lines.append(FileLineModel(
                    name=os.path.basename(file_path),
                    path=file_path,
                    line_number=line_number,
                    content=original_line
                ))
 
    return matching_lines

def write_file(
    file_path: Annotated[str, "The path to the file to write to."], 
    content: Annotated[str, "The content to write to the file."], 
    append: Annotated[bool, "Whether to append to the file (True) or overwrite it (False)."] = False
) -> bool:
    """
    Writes content to a file.

    Args:
    file_path (str): The path to the file to write to.
    content (str): The content to write to the file.
    append (bool): Whether to append to the file (True) or overwrite it (False).
    """
    check_allow_outside_modification(file_path)

    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8') as file:
        file.write(content)
    return True

def delete_file(
    file_path: Annotated[str, "The path to the file to delete."]
) -> bool:
    """
    Deletes a file.

    Args:
    file_path (str): The path to the file to delete.
    """
    check_allow_outside_modification(file_path)


    if not os.path.isfile(file_path):
        logger.warning(f"The path {file_path} is not a valid file.")
        return False

    os.remove(file_path)
    return True

def create_directory(
    directory_path: Annotated[str, "The path to the directory to create."]
) -> bool:
    """
    Creates a directory.

    Args:
    directory_path (str): The path to the directory to create.
    """
    check_allow_outside_modification(directory_path)


    if os.path.exists(directory_path):
        logger.warning(f"The path {directory_path} already exists.")
        return False

    os.makedirs(directory_path)
    return True

def delete_directory(
    directory_path: Annotated[str, "The path to the directory to delete."]
) -> bool:
    """
    Deletes a directory.

    Args:
    directory_path (str): The path to the directory to delete.
    """
    check_allow_outside_modification(directory_path)


    if not os.path.isdir(directory_path):
        logger.warning(f"The path {directory_path} is not a valid directory.")
        return False

    os.rmdir(directory_path)
    return True