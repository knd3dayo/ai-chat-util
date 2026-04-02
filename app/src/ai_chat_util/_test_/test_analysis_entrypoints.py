from __future__ import annotations

import asyncio
from typing import Any

from ai_chat_util.analysis import AnalysisService
from ai_chat_util.api.api_server import router
from ai_chat_util.core import tool_app
from ai_chat_util.common.model.ai_chatl_util_models import ChatResponse
from ai_chat_util.mcp import mcp_server as mcp_server_mod


def test_api_router_registers_analysis_routes() -> None:
    route_paths = {route.path for route in router.routes}

    assert "/analyze_image_files" in route_paths
    assert "/analyze_pdf_files" in route_paths
    assert "/analyze_office_files" in route_paths
    assert "/analyze_files" in route_paths
    assert "/analyze_documents_data" in route_paths


def test_prepare_mcp_registers_analysis_tools() -> None:
    registered_names: list[str] = []

    class _FakeMCP:
        def tool(self):
            def _decorator(func):
                registered_names.append(func.__name__)
                return func

            return _decorator

    mcp_server_mod.prepare_mcp(
        _FakeMCP(),
        "analyze_image_files,analyze_pdf_files,analyze_office_files,analyze_files,analyze_documents_data",
    )

    assert registered_names == [
        "analyze_image_files",
        "analyze_pdf_files",
        "analyze_office_files",
        "analyze_files",
        "analyze_documents_data",
    ]


def test_tool_app_analyze_files_uses_analysis_service(monkeypatch) -> None:
    called: dict[str, Any] = {"resolved": None, "analyze": None}

    class _FakeClient:
        pass

    async def _fake_analyze_files(
        cls,
        llm_client: Any,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        called["analyze"] = {
            "client": llm_client,
            "file_path_list": list(file_path_list),
            "prompt": prompt,
            "detail": detail,
            "resolve_paths": resolve_paths,
        }
        return ChatResponse(status="completed", messages=[])

    monkeypatch.setattr(tool_app, "create_llm_client", lambda: _FakeClient())
    monkeypatch.setattr(AnalysisService, "resolve_existing_file_paths", lambda paths: ["/resolved/a.txt"])
    monkeypatch.setattr(AnalysisService, "analyze_files", classmethod(_fake_analyze_files))

    result = asyncio.run(tool_app.analyze_files(["a.txt"], "summarize", "auto"))

    assert result == ""
    assert called["analyze"] == {
        "client": called["analyze"]["client"],
        "file_path_list": ["/resolved/a.txt"],
        "prompt": "summarize",
        "detail": "auto",
        "resolve_paths": False,
    }


def test_tool_app_analyze_image_files_retries_timeout(monkeypatch) -> None:
    attempts = {"count": 0}

    class _FakeClient:
        pass

    async def _fake_analyze_image_files(
        cls,
        llm_client: Any,
        file_list: list[str],
        prompt: str,
        detail: str,
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise asyncio.TimeoutError()
        return ChatResponse(status="completed", messages=[])

    monkeypatch.setattr(tool_app, "create_llm_client", lambda: _FakeClient())
    monkeypatch.setattr(AnalysisService, "resolve_existing_file_paths", lambda paths: ["/resolved/image.png"])
    monkeypatch.setattr(AnalysisService, "tool_timeout_seconds", lambda: 0.01)
    monkeypatch.setattr(AnalysisService, "tool_timeout_retries", lambda: 1)
    monkeypatch.setattr(AnalysisService, "analyze_image_files", classmethod(_fake_analyze_image_files))

    result = asyncio.run(tool_app.analyze_image_files(["image.png"], "describe", "auto"))

    assert result == ""
    assert attempts["count"] == 2