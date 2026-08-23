from pathlib import Path

import pyzipper as zipfile

from ai_chat_util.util.analyze_file_util.zip_util import ZipUtil


def _create_cp932_filename_zip(zip_path: Path, filename: str, content: bytes) -> None:
    raw_name = filename.encode("cp932")
    placeholder_name = "a" * len(raw_name)
    source = zip_path.parent / "source.txt"
    source.write_bytes(content)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zip_ref:
        zip_ref.write(source, arcname=placeholder_name)

    archive_bytes = zip_path.read_bytes()
    placeholder_bytes = placeholder_name.encode("ascii")
    zip_path.write_bytes(archive_bytes.replace(placeholder_bytes, raw_name, 2))


def test_list_zip_contents_decodes_cp932_filename(tmp_path: Path) -> None:
    zip_path = tmp_path / "cp932.zip"
    expected_name = "日本語.txt"
    _create_cp932_filename_zip(zip_path, expected_name, b"hello")

    assert ZipUtil.list_zip_contents(str(zip_path)) == [expected_name]


def test_extract_zip_decodes_cp932_filename(tmp_path: Path) -> None:
    zip_path = tmp_path / "cp932.zip"
    expected_name = "日本語.txt"
    _create_cp932_filename_zip(zip_path, expected_name, b"hello")
    extract_to = tmp_path / "out"

    assert ZipUtil.extract_zip(str(zip_path), str(extract_to)) is True
    assert (extract_to / expected_name).read_bytes() == b"hello"


def test_list_zip_contents_keeps_utf8_filename(tmp_path: Path) -> None:
    zip_path = tmp_path / "utf8.zip"
    expected_name = "レポート.txt"
    source = tmp_path / expected_name
    source.write_bytes(b"hello")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zip_ref:
        zip_ref.write(source, arcname=expected_name)

    assert ZipUtil.list_zip_contents(str(zip_path)) == [expected_name]