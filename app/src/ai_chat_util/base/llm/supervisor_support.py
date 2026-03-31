from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field

from ai_chat_util_base.config.runtime import AiChatUtilConfig
from ai_chat_util_base.model.request_headers import RequestHeaders

import ai_chat_util.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


class RouteCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_name: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason_code: str
    tool_hints: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_route: str
    candidate_routes: list[RouteCandidate] = Field(default_factory=list)
    reason_code: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_information: list[str] = Field(default_factory=list)
    next_action: str = Field(default="execute_selected_route")
    requires_hitl: bool = Field(default=False)
    requires_clarification: bool = Field(default=False)
    notes: str | None = Field(default=None)


class EvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: str | None = None
    headings: list[str] = Field(default_factory=list)
    stdout_blocks: list[str] = Field(default_factory=list)
    raw_texts: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    successful_tools: list[str] = Field(default_factory=list)
    latest_task_id: str | None = None
    has_actionable_evidence: bool = Field(default=False)


class SufficiencyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str
    reason_code: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_facts: list[str] = Field(default_factory=list)
    recommended_next_action: str = Field(default="finalize_answer")
    requires_hitl: bool = Field(default=False)
    requires_approval: bool = Field(default=False)
    evidence_summary: EvidenceSummary = Field(default_factory=EvidenceSummary)
    user_prompt_hint: str | None = Field(default=None)


class SupervisorAuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str
    trace_id: str
    turn_id: str
    timestamp: str
    agent_name: str | None = None
    tool_name: str | None = None
    route_name: str | None = None
    reason_code: str | None = None
    confidence: float | None = None
    user_identity_hint: str | None = None
    target_system: str | None = None
    action_kind: str | None = None
    resource_identifier: str | None = None
    approval_status: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    final_status: str | None = None


class AuditSink(ABC):
    @abstractmethod
    def write_event(self, event: SupervisorAuditEvent) -> None:
        raise NotImplementedError


class NullAuditSink(AuditSink):
    def write_event(self, event: SupervisorAuditEvent) -> None:
        return


class JsonlAuditSink(AuditSink):
    def __init__(self, destination: Path):
        self.destination = destination.expanduser()

    def write_event(self, event: SupervisorAuditEvent) -> None:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")


@dataclass(slots=True)
class AuditContext:
    trace_id: str
    turn_id: str
    enabled: bool
    user_identity_hint: str | None = None
    sink: AuditSink = field(default_factory=NullAuditSink)

    def emit(
        self,
        event_type: str,
        *,
        agent_name: str | None = None,
        tool_name: str | None = None,
        route_name: str | None = None,
        reason_code: str | None = None,
        confidence: float | None = None,
        user_identity_hint: str | None = None,
        target_system: str | None = None,
        action_kind: str | None = None,
        resource_identifier: str | None = None,
        approval_status: str | None = None,
        payload: dict[str, Any] | None = None,
        final_status: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        event = SupervisorAuditEvent(
            event_type=event_type,
            trace_id=self.trace_id,
            turn_id=self.turn_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=agent_name,
            tool_name=tool_name,
            route_name=route_name,
            reason_code=reason_code,
            confidence=confidence,
            user_identity_hint=user_identity_hint or self.user_identity_hint,
            target_system=target_system,
            action_kind=action_kind,
            resource_identifier=resource_identifier,
            approval_status=approval_status,
            payload=payload or {},
            final_status=final_status,
        )
        try:
            self.sink.write_event(event)
        except Exception:
            logger.debug("Failed to write supervisor audit event", exc_info=True)


def build_audit_sink(runtime_config: AiChatUtilConfig) -> AuditSink:
    if not bool(getattr(runtime_config.features, "audit_log_enabled", False)):
        return NullAuditSink()

    raw_destination = getattr(runtime_config.features, "audit_log_path", None)
    destination = Path(str(raw_destination).strip()) if isinstance(raw_destination, str) and raw_destination.strip() else Path("work/supervisor_audit.jsonl")
    return JsonlAuditSink(destination)


def create_audit_context(
    runtime_config: AiChatUtilConfig,
    trace_id: str,
    *,
    request_headers: RequestHeaders | None = None,
) -> AuditContext:
    sink = build_audit_sink(runtime_config)
    return AuditContext(
        trace_id=trace_id,
        turn_id=uuid.uuid4().hex,
        enabled=not isinstance(sink, NullAuditSink),
        user_identity_hint=request_headers.user_identity_hint() if request_headers is not None else None,
        sink=sink,
    )