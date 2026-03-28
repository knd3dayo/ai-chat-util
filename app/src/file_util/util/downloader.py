from __future__ import annotations

import os
from typing import Any, Sequence

import requests
def _get_verify_option(*, requests_verify: bool = True, ca_bundle: str | None = None) -> bool | str:
    if ca_bundle:
        return ca_bundle
    return requests_verify


def _get_file_name_from_url(url: str) -> str:
    from urllib.parse import urlparse

    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)


def _get_headers(item: Any) -> dict[str, Any] | None:
    headers = getattr(item, "headers", None)
    if headers is None:
        return None
    return dict(headers)


class DownLoader:

    @classmethod
    def download_files(
        cls,
        urls: Sequence[Any],
        download_dir: str,
        *,
        requests_verify: bool = True,
        ca_bundle: str | None = None,
    ) -> list[str]:
        """Download files to the specified directory."""
        verify = _get_verify_option(requests_verify=requests_verify, ca_bundle=ca_bundle)

        file_paths: list[str] = []
        for item in urls:
            res = requests.get(
                url=item.url,
                headers=_get_headers(item),
                verify=verify,
                timeout=(10, 60),
            )
            res.raise_for_status()

            file_path = os.path.join(download_dir, _get_file_name_from_url(item.url))
            with open(file_path, "wb") as f:
                f.write(res.content)
            file_paths.append(file_path)
        return file_paths

    @classmethod
    async def download_files_async(
        cls,
        urls: Sequence[Any],
        download_dir: str,
        *,
        requests_verify: bool = True,
        ca_bundle: str | None = None,
    ) -> list[str]:
        """Download files asynchronously for async workflows."""
        try:
            import httpx
        except Exception as e:
            raise RuntimeError("httpx が見つかりません。依存関係を確認してください。") from e

        verify = _get_verify_option(requests_verify=requests_verify, ca_bundle=ca_bundle)
        timeout = httpx.Timeout(60.0, connect=10.0)

        file_paths: list[str] = []
        async with httpx.AsyncClient(verify=verify, timeout=timeout, follow_redirects=True) as client:
            for item in urls:
                resp = await client.get(item.url, headers=_get_headers(item))
                resp.raise_for_status()

                file_path = os.path.join(download_dir, _get_file_name_from_url(item.url))
                with open(file_path, "wb") as f:
                    f.write(resp.content)
                file_paths.append(file_path)

        return file_paths


__all__ = ["DownLoader"]