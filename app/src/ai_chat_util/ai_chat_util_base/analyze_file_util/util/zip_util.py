import locale
import os
from pathlib import Path

import pyzipper as zipfile

class ZipUtil:

    @classmethod
    def __check_utf8_flag(cls, zip_ref: zipfile.ZipFile):
        # 1つでもUTF-8フラグが立っていればTrueを返す
        for info in zip_ref.infolist():
            if bool(info.flag_bits & 0x800):
                return True
        return False

    @classmethod
    def extract_zip(cls, file_path, extract_to, password=None) -> bool:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            system_encoding = cls.__get_system_encoding()
            is_utf = cls.__check_utf8_flag(zip_ref)
            if password:
                zip_ref.setpassword(password.encode())
            for info in zip_ref.infolist():
                name = info.filename
                if not is_utf:
                    name = name.encode('cp437').decode(system_encoding, errors='replace')
                info.filename = name
                zip_ref.extract(info, path=extract_to)

        return True


    @classmethod
    def __get_system_encoding(cls):
        return locale.getpreferredencoding()

    @classmethod
    def list_zip_contents(cls, file_path) -> list[str]:
        result = []
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            is_utf = cls.__check_utf8_flag(zip_ref)
            system_encoding = cls.__get_system_encoding()
            for info in zip_ref.infolist():
                name = info.filename
                if not is_utf:
                    name = name.encode('cp437').decode(system_encoding, errors='replace')
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
