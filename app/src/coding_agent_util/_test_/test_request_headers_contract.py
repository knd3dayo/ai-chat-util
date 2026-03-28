from __future__ import annotations

from ai_chat_util_base.model.request_headers import RequestHeaders


def test_request_headers_trace_id_aliases() -> None:
    h1 = RequestHeaders.from_mapping({"X-Trace-Id": "t1"})
    assert h1.trace_id == "t1"

    h2 = RequestHeaders.from_mapping({"trace-id": "t2"})
    assert h2.trace_id == "t2"

    h3 = RequestHeaders.from_mapping({"trace_id": "t3"})
    assert h3.trace_id == "t3"


def test_request_headers_to_env_contract() -> None:
    hdr = RequestHeaders.from_values(authorization="Bearer abc", trace_id="trace-123")
    env = hdr.to_env()

    assert env["AI_PLATFORM_AUTHORIZATION"] == "Bearer abc"
    assert env["AUTHORIZATION"] == "Bearer abc"
    assert env["AI_PLATFORM_TRACE_ID"] == "trace-123"
    assert env["TRACE_ID"] == "trace-123"