import os
import requests

from ai_chat_util_base.config.runtime import get_runtime_config
from ai_chat_util_base.model.ai_chatl_util_models import (
    WebRequestModel
)


class DownLoader:

    @classmethod
    def download_files(cls, urls: list[WebRequestModel], download_dir: str) -> list[str]:
        """
        Download files from the given URLs to the specified directory.
        Returns a list of file paths where the files are saved.
        """
        cfg = get_runtime_config()

        verify_enabled = cfg.network.requests_verify
        ca_bundle = cfg.network.ca_bundle
        # requestsのverifyには bool | str(=CA bundle path) を渡せる
        verify: bool | str
        if ca_bundle:
            verify = ca_bundle
        else:
            verify = verify_enabled

        def get_file_name_from_url(url: str) -> str:
            """
            URLからファイル名を抽出する。
            例: https://example.com/path/to/file.txt -> file.txt
            """
            from urllib.parse import urlparse
            import os
            parsed_url = urlparse(url)
            return os.path.basename(parsed_url.path)
        
        file_paths = []
        for item in urls:
            # タイムアウト無しだと接続/読み取りで無限待ちになり得る
            res = requests.get(url=item.url, headers=item.headers, verify=verify, timeout=(10, 60))
            res.raise_for_status()
            
            file_path = os.path.join(download_dir, get_file_name_from_url(item.url))
            with open(file_path, "wb") as f:
                f.write(res.content)
            file_paths.append(file_path)
        return file_paths

    @classmethod
    async def download_files_async(cls, urls: list[WebRequestModel], download_dir: str) -> list[str]:
        """Download files asynchronously (for use inside async flows).

        This avoids blocking the event loop (sync requests.get) and ensures
        HTTP resources are closed via `async with`.
        """
        cfg = get_runtime_config()

        verify_enabled = cfg.network.requests_verify
        ca_bundle = cfg.network.ca_bundle
        verify: bool | str
        if ca_bundle:
            verify = ca_bundle
        else:
            verify = verify_enabled

        def get_file_name_from_url(url: str) -> str:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            return os.path.basename(parsed_url.path)

        try:
            import httpx
        except Exception as e:
            raise RuntimeError("httpx が見つかりません。依存関係を確認してください。") from e

        timeout = httpx.Timeout(60.0, connect=10.0)

        file_paths: list[str] = []
        async with httpx.AsyncClient(verify=verify, timeout=timeout, follow_redirects=True) as client:
            for item in urls:
                resp = await client.get(item.url, headers=item.headers)
                resp.raise_for_status()

                file_path = os.path.join(download_dir, get_file_name_from_url(item.url))
                with open(file_path, "wb") as f:
                    f.write(resp.content)
                file_paths.append(file_path)

        return file_paths
