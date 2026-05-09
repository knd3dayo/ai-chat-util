import locale
import os
from pathlib import Path

import chardet
import pyzipper as zipfile

class ZipUtil:

    _JAPANESE_RANGES = (
        (0x3040, 0x30FF),
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0xFF66, 0xFF9F),
    )

    @classmethod
    def __check_utf8_flag(cls, zip_ref: zipfile.ZipFile):
        # 1つでもUTF-8フラグが立っていればTrueを返す
        for info in zip_ref.infolist():
            if bool(info.flag_bits & 0x800):
                return True
        return False

    @classmethod
    def extract_zip(cls, file_path, extract_to, password=None) -> bool:
        # First pass: extract to a temporary location and detect structure
        temp_extract = Path(extract_to) / "_temp_zip_extract"
        temp_extract.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                system_encoding = cls.__get_system_encoding()
                is_utf = cls.__check_utf8_flag(zip_ref)
                if password:
                    zip_ref.setpassword(password.encode())
                for info in zip_ref.infolist():
                    name = info.filename
                    if not is_utf:
                        name = cls.__decode_legacy_filename(name, system_encoding)
                    info.filename = name
                    zip_ref.extract(info, path=str(temp_extract))

            # Check if all contents are in a single top-level directory
            items = list(temp_extract.iterdir())
            if len(items) == 1 and items[0].is_dir():
                # All files are in a single subdirectory - move its contents up
                subdir = items[0]
                for item in subdir.iterdir():
                    dest_path = Path(extract_to) / item.name
                    # Handle existing files/directories
                    if dest_path.exists():
                        if dest_path.is_dir():
                            import shutil
                            shutil.rmtree(dest_path)
                        else:
                            dest_path.unlink()
                    item.rename(dest_path)
                subdir.rmdir()
            else:
                # Multiple top-level items or files - keep as-is, move from temp
                for item in temp_extract.iterdir():
                    dest_path = Path(extract_to) / item.name
                    if dest_path.exists():
                        if dest_path.is_dir():
                            import shutil
                            shutil.rmtree(dest_path)
                        else:
                            dest_path.unlink()
                    item.rename(dest_path)
            
            # Clean up temp directory
            temp_extract.rmdir()
            return True
        except Exception:
            # On error, clean up temp and re-raise
            import shutil
            if temp_extract.exists():
                shutil.rmtree(temp_extract, ignore_errors=True)
            raise


    @classmethod
    def __get_system_encoding(cls):
        return locale.getpreferredencoding()

    @classmethod
    def __contains_japanese(cls, text: str) -> bool:
        for char in text:
            codepoint = ord(char)
            for start, end in cls._JAPANESE_RANGES:
                if start <= codepoint <= end:
                    return True
        return False

    @classmethod
    def __decode_legacy_filename(cls, name: str, system_encoding: str) -> str:
        raw_name = name.encode("cp437")
        if raw_name.isascii():
            return name

        detected = chardet.detect(raw_name)
        candidate_encodings: list[str] = []

        detected_encoding = detected.get("encoding")
        detected_confidence = float(detected.get("confidence") or 0.0)
        if detected_encoding and detected_confidence >= 0.5:
            candidate_encodings.append(str(detected_encoding))

        candidate_encodings.extend(["cp932", "shift_jis"])
        if system_encoding.lower() not in {"utf-8", "utf_8"}:
            candidate_encodings.append(system_encoding)

        seen: set[str] = set()
        for encoding in candidate_encodings:
            normalized = encoding.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            try:
                decoded = raw_name.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
            if cls.__contains_japanese(decoded):
                return decoded

        return name

    @classmethod
    def list_zip_contents(cls, file_path) -> list[str]:
        result = []
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            is_utf = cls.__check_utf8_flag(zip_ref)
            system_encoding = cls.__get_system_encoding()
            for info in zip_ref.infolist():
                name = info.filename
                if not is_utf:
                    name = cls.__decode_legacy_filename(name, system_encoding)
                result.append(name)
        return result

    @classmethod
    def create_zip(cls, file_paths: list[str], output_zip: str, password=None) -> bool:
        output_path = Path(output_zip).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if password:
            with zipfile.AESZipFile(
                output_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                encryption=zipfile.WZ_AES,
            ) as zip_ref:
                zip_ref.setpassword(password.encode())
                cls._write_zip_entries(zip_ref, file_paths)
            return True

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_ref:
            cls._write_zip_entries(zip_ref, file_paths)
        return True

    @classmethod
    def _write_zip_entries(cls, zip_ref: zipfile.ZipFile, file_paths: list[str]) -> None:
        for file_path in file_paths:
            if os.path.isdir(file_path):
                for root, _, files in os.walk(file_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, start=os.path.dirname(file_path))
                        zip_ref.write(full_path, arcname=arcname)
            else:
                zip_ref.write(file_path, arcname=os.path.basename(file_path))
