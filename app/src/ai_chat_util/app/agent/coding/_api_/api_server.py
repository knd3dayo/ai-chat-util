from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, Request

from ..core.endpoint import EndPoint
from ai_chat_util.app.agent.agent_util_models import CancelResponse, ExecuteResponse, HealthzResponse, TaskStatus
from ai_chat_util.core.request_headers import RequestHeaders, bind_current_request_headers, get_current_request_headers
from ai_chat_util.core.common.config.runtime import init_coding_runtime


logger = logging.getLogger(__name__)
endpoint = EndPoint()

def _summarize_http_body(method: str, path: str, body: bytes, content_type: str | None) -> dict[str, Any] | None:
    if method not in ("POST", "PUT", "PATCH"):
        return None
    if not body:
        return {"size": 0}

    ct = (content_type or "").lower()
    if "application/json" not in ct:
        # Do not log arbitrary bodies.
        return {"size": len(body), "type": "non-json"}

    # Try parsing small JSON only.
    if len(body) > 20_000:
        return {"size": len(body), "type": "json", "note": "too_large"}

    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except Exception:
        return {"size": len(body), "type": "json", "note": "decode_failed"}

    if not isinstance(parsed, dict):
        return {"size": len(body), "type": "json", "shape": type(parsed).__name__}

    keys = sorted(str(k) for k in parsed.keys())
    summary: dict[str, Any] = {"type": "json", "size": len(body), "keys": keys}

    # Special-case known request shapes.
    if path.endswith("/execute"):
        prompt = parsed.get("prompt")
        if isinstance(prompt, str):
            summary["prompt_chars"] = len(prompt)
            summary["prompt_lines"] = prompt.count("\n") + 1 if prompt else 0
        ws = parsed.get("workspace_path")
        if isinstance(ws, str):
            summary["workspace_path"] = ws
        timeout = parsed.get("timeout")
        if isinstance(timeout, int):
            summary["timeout"] = timeout
        task_id = parsed.get("task_id")
        if isinstance(task_id, str) and task_id:
            summary["task_id"] = task_id
        trace_id = parsed.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            summary["trace_id"] = trace_id

    return summary


def _summarize_http_request(request: Request, *, body_summary: dict[str, Any] | None) -> dict[str, Any]:
    # Never log header values directly (Authorization etc). Only presence/keys.
    header_keys = sorted({str(k).lower() for k in request.headers.keys()})
    has_auth = "authorization" in header_keys
    query_keys = sorted({k for k in request.query_params.keys()})

    trace_id = None
    incoming = get_current_request_headers()
    if incoming and incoming.trace_id:
        trace_id = incoming.trace_id

    return {
        "method": request.method,
        "path": request.url.path,
        "query_keys": query_keys,
        "header_keys": header_keys,
        "has_authorization": has_auth,
        "trace_id": trace_id,
        "body": body_summary,
    }


def create_app(*, sync_mode: bool = False, init_config: bool = True) -> FastAPI:
    """Create FastAPI app.

    `sync_mode=False` (default): /execute returns immediately (async execution).
    `sync_mode=True`: /execute blocks until task completion (sync execution).
    """

    # NOTE:
    # Avoid initializing runtime at module import time. Some contexts (pytest collection,
    # uvicorn import-style usage) import this module before env/config is set.
    if init_config:
        init_coding_runtime(None)

    app = FastAPI(title="Coding Agent Executor API", version="0.1")

    @app.middleware("http")
    async def _capture_request_headers(request: Request, call_next):
        headers = {str(k).lower(): str(v) for k, v in request.headers.items()}
        with bind_current_request_headers(RequestHeaders.from_mapping(headers)):
            return await call_next(request)

    @app.middleware("http")
    async def _log_request_response(request: Request, call_next):
        start = time.perf_counter()

        body_bytes: bytes = b""
        body_summary: dict[str, Any] | None = None
        try:
            # Reading body is safe for JSON requests; Starlette caches it.
            # Avoid logging sensitive values; only compute summaries.
            if request.method in ("POST", "PUT", "PATCH"):
                body_bytes = await request.body()
                body_summary = _summarize_http_body(
                    request.method,
                    request.url.path,
                    body_bytes,
                    request.headers.get("content-type"),
                )
        except Exception:
            body_summary = {"note": "body_read_failed"}

        req_info = _summarize_http_request(request, body_summary=body_summary)
        logger.info("http.request %s", req_info)

        try:
            response = await call_next(request)
        except Exception:
            dt_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "http.error method=%s path=%s dt_ms=%s trace_id=%s",
                request.method,
                request.url.path,
                dt_ms,
                req_info.get("trace_id"),
            )
            raise

        dt_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "http.response method=%s path=%s status=%s dt_ms=%s trace_id=%s",
            request.method,
            request.url.path,
            getattr(response, "status_code", None),
            dt_ms,
            req_info.get("trace_id"),
        )
        return response

    app.get("/healthz", response_model=HealthzResponse)(endpoint.healthz)
    app.post("/execute", response_model=ExecuteResponse)(
        endpoint.execute_sync if sync_mode else endpoint.execute_async
    )
    app.get("/status/{task_id}", response_model=TaskStatus)(endpoint.status)
    # Result endpoint for clients that want stdout/stderr only.
    # Note: status already includes stdout/stderr; this is a convenience wrapper.
    app.get("/get_result/{task_id}")(endpoint.get_result)
    app.delete("/cancel/{task_id}", response_model=CancelResponse)(endpoint.cancel)

    return app


# Default app instance for uvicorn import-style usage.
# Runtime config is initialized on actual execution paths (CLI __main__ or explicit create_app(init_config=True)).
app = create_app(sync_mode=False, init_config=False)

def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run Coding Agent Executor API")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "Path to config YAML (ai-chat-util-config.yml). If omitted, resolved by env AI_CHAT_UTIL_CONFIG "
            "(with root-level coding_agent_util section), or searched from CWD/project root."
        ),
    )
    parser.add_argument("-p", "--port", type=int, default=7101)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--sync_mode",
        action="store_true",
        help="Run API server in synchronous mode (/execute blocks until completion).",
    )
    args = parser.parse_args()

    init_coding_runtime(args.config or None)

    uvicorn.run(create_app(sync_mode=args.sync_mode), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
