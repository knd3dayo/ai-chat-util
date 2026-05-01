from __future__ import annotations

import re
from typing import Any, Sequence
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from .prompts_base import PromptsBase
from .supervisor_support import AuditContext
from ai_chat_util.core.common.config.ai_chat_util_mcp_config import MCPServerConfig
from ai_chat_util.core.common.config.runtime import AiChatUtilConfig
from .tool_limits import ToolLimits
import ai_chat_util.core.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class AgentBuilder:
	_APPROVAL_METADATA_PATTERN = re.compile(r"requires_approval\s*=\s*true", re.IGNORECASE)

	@staticmethod
	def _slugify_agent_name_part(value: str) -> str:
		normalized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
		normalized = re.sub(r"_+", "_", normalized).strip("_")
		return normalized or "default"

	@classmethod
	def build_tool_agent_name(
		cls,
		*,
		group_label: str | None = None,
		server_names: Sequence[str] | None = None,
	) -> str:
		suffix_source = group_label
		if suffix_source is None:
			normalized_servers = [
				cls._slugify_agent_name_part(name)
				for name in (server_names or [])
				if isinstance(name, str) and name.strip()
			]
			if normalized_servers:
				suffix_source = "_".join(sorted(dict.fromkeys(normalized_servers)))

		suffix = cls._slugify_agent_name_part(suffix_source or "default")
		return f"tool_agent_{suffix}"

	def get_tools(self) -> list[Any]:
		return self.langchain_tools
    
	def get_agent(self) -> Any:
		return self.agent
    
	def get_hitl_approval_tools(self) -> Sequence[str]:
		return self.hitl_approval_tools

	def get_agent_name(self) -> str:
		return self.agent_name

	def get_tools_description(self) -> str:
		tools_description = "\n".join(f"## name: {tool.name}\n - description: {tool.description}\n - args_schema: {tool.args_schema}\n" for tool in self.langchain_tools)
		return tools_description

	@classmethod
	def infer_approval_tools_from_langchain_tools(cls, tools: Sequence[Any]) -> list[str]:
		inferred: list[str] = []
		seen: set[str] = set()
		for tool in tools:
			name = str(getattr(tool, "name", "") or "").strip()
			description = str(getattr(tool, "description", "") or "")
			if not name or name in seen:
				continue
			if cls._APPROVAL_METADATA_PATTERN.search(description):
				inferred.append(name)
				seen.add(name)
		return inferred

	@staticmethod
	def merge_approval_tool_names(*groups: Sequence[str] | None) -> list[str]:
		merged: list[str] = []
		seen: set[str] = set()
		for group in groups:
			for name in group or []:
				normalized = str(name).strip()
				if not normalized or normalized in seen:
					continue
				seen.add(normalized)
				merged.append(normalized)
		return merged

	async def prepare(
			self,
			runtime_config: AiChatUtilConfig,
			mcp_config: MCPServerConfig ,
			llm: BaseChatModel,
			prompts: PromptsBase,
			tool_limits: ToolLimits | None,
			agent_name: str,
			allowed_tool_names: Sequence[str] | None = None,
			explicit_user_file_paths: Sequence[str] | None = None,
			approved_tool_names: Sequence[str] | None = None,
			audit_context: AuditContext | None = None,
		):

		# Safety valves: cap tool calls and hard-timeout tool execution.
		# This enforces termination even if prompts are ignored.
		if tool_limits is not None:
			tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = tool_limits.guard_params()
		else:
			tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = (0, 0.0, 0)

		effective_tool_call_limit_int, effective_followup_tool_call_limit_int = ToolLimits.effective_call_limits(
			tool_call_limit_int,
			int(tool_limits.followup_tool_call_limit) if tool_limits is not None else 0,
			explicit_user_file_paths,
		)

		# Shared tool call counter across all tools within this workflow.
		# Using a mutable container avoids invalid `nonlocal` usage across methods.
		configured_hitl_approval_tools = [
			str(name).strip()
			for name in (runtime_config.features.hitl_approval_tools or [])
			if isinstance(name, str) and str(name).strip()
		]

		tool_call_state: dict[str, Any] = {
			"used": 0,
			"general_used": 0,
			"followup_used": 0,
			"followup_limit": effective_followup_tool_call_limit_int,
			"agent_name": agent_name,
			"audit_context": audit_context,
			"approval_tools": {
				*configured_hitl_approval_tools,
			},
			"approved_tools": {
				str(name).strip()
				for name in (approved_tool_names or [])
				if isinstance(name, str) and str(name).strip()
			},
			"auto_approve": bool(tool_limits.auto_approve) if tool_limits is not None else False,
			"explicit_user_file_paths": [
				str(path).strip()
				for path in (explicit_user_file_paths or [])
				if isinstance(path, str) and str(path).strip()
			],
		}

		agent_client = MultiServerMCPClient(mcp_config.to_langchain_config())
		# LangChainのツールリストを取得
		coding_agent_langchain_tools = await agent_client.get_tools()
		if allowed_tool_names is not None:
			allowed_names = {name for name in allowed_tool_names if isinstance(name, str) and name.strip()}
			coding_agent_langchain_tools = [
				tool for tool in coding_agent_langchain_tools if str(getattr(tool, "name", "")) in allowed_names
			]

		inferred_hitl_approval_tools = self.infer_approval_tools_from_langchain_tools(coding_agent_langchain_tools)
		hitl_approval_tools = self.merge_approval_tool_names(
			configured_hitl_approval_tools,
			inferred_hitl_approval_tools,
		)
		tool_call_state["approval_tools"] = set(hitl_approval_tools)

		ToolLimits._apply_tool_execution_guards(
			coding_agent_langchain_tools,
			tool_call_state=tool_call_state,
			tool_call_limit_int=effective_tool_call_limit_int,
			tool_timeout_seconds_f=tool_timeout_seconds_f,
			tool_timeout_retries_int=tool_timeout_retries_int,
		)
		logger.info("Creating code agent for MCP server '%s'...", ", ".join(mcp_config.servers.keys()))

		approval_tools = [t for t in (hitl_approval_tools or []) if isinstance(t, str) and t.strip()]
		approval_tools_text = ", ".join(approval_tools) if approval_tools else "(なし)"

		if tool_limits is not None and tool_limits.auto_approve:
			hitl_policy_text = prompts.auto_approve_hitl_policy_text(approval_tools_text)
		else:
			hitl_policy_text = prompts.normal_hitl_policy_text(approval_tools_text)

		tool_agent_system_prompt = prompts.tool_agent_system_prompt(
			hitl_policy_text,
			agent_name=agent_name,
			followup_poll_interval_seconds=float(getattr(runtime_config.features, "mcp_followup_poll_interval_seconds", 2.0) or 0.0),
			status_tail_lines=int(getattr(runtime_config.features, "mcp_status_tail_lines", 20) or 0),
			result_tail_lines=int(getattr(runtime_config.features, "mcp_get_result_tail_lines", 80) or 0),
		)

		tool_agent = create_agent(
			llm,
			coding_agent_langchain_tools,
			system_prompt=tool_agent_system_prompt,
			name=agent_name,
		)

		self.agent = tool_agent
		self.agent_name = agent_name
		self.langchain_tools = coding_agent_langchain_tools
		self.hitl_approval_tools = hitl_approval_tools

	@classmethod
	async def create_sub_agents(
		cls,
		runtime_config: AiChatUtilConfig,
		mcp_config: MCPServerConfig ,
		llm: BaseChatModel,
		prompts: PromptsBase,
		tool_limits: ToolLimits | None,
		include_coding_agent: bool = True,
		include_general_agent: bool = True,
		general_tool_allowlist: Sequence[str] | None = None,
		explicit_user_file_paths: Sequence[str] | None = None,
		approved_tool_names: Sequence[str] | None = None,
		audit_context: AuditContext | None = None,
	) -> list[AgentBuilder]:
		logger.info("Creating sub-agents...")

		# Safety valves: cap tool calls and hard-timeout tool execution.
		# This enforces termination even if prompts are ignored.
		if tool_limits is not None:
			tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = tool_limits.guard_params()
		else:
			tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = (0, 0.0, 0)

		# Shared tool call counter across all tools within this workflow.
		# Using a mutable container avoids invalid `nonlocal` usage across methods.
		tool_call_state: dict[str, int] = {"used": 0}

		hitl_approval_tools = runtime_config.features.hitl_approval_tools or []

		coding_agent_server_name = runtime_config.mcp.coding_agent_endpoint.mcp_server_name

		# coding_agent用のサーバー定義を取得
		code_agent_mcp_config = mcp_config.filter(include_name=coding_agent_server_name)
		normal_tools_mcp_config = mcp_config.filter(exclude_name=coding_agent_server_name)
		# allowed_tools_configにcoding_agent_nameと一致するツールがあれば、コードエージェントを作成する。
		agents = []
		should_create_coding_agent = include_coding_agent or len(normal_tools_mcp_config.servers) == 0
		if should_create_coding_agent and len(code_agent_mcp_config.servers) > 0 :
			logger.info("Creating code agent for MCP server '%s'...", ", ".join(code_agent_mcp_config.servers.keys()))
			code_agent_builder = AgentBuilder()
			await code_agent_builder.prepare(
				runtime_config=runtime_config,
				mcp_config=code_agent_mcp_config,
				llm=llm,
				prompts=prompts,
				tool_limits=tool_limits,
				agent_name=cls.build_tool_agent_name(
					group_label="coding",
					server_names=tuple(code_agent_mcp_config.servers.keys()),
				),
				explicit_user_file_paths=explicit_user_file_paths,
				approved_tool_names=approved_tool_names,
				audit_context=audit_context,
			)
			agents.append(code_agent_builder)

		should_create_general_agent = (
			include_general_agent
			or len(code_agent_mcp_config.servers) == 0
			or (general_tool_allowlist is not None and len([name for name in general_tool_allowlist if isinstance(name, str) and name.strip()]) > 0)
		)
		if should_create_general_agent and len(normal_tools_mcp_config.servers) > 0:
			logger.info("Creating normal agent for MCP server '%s'...", ", ".join(normal_tools_mcp_config.servers.keys()))
			normal_agent_builder = AgentBuilder()
			await normal_agent_builder.prepare(
				runtime_config=runtime_config,
				mcp_config=normal_tools_mcp_config,
				llm=llm,
				prompts=prompts,
				tool_limits=tool_limits,
				agent_name=cls.build_tool_agent_name(
					group_label="general",
					server_names=tuple(normal_tools_mcp_config.servers.keys()),
				),
				allowed_tool_names=general_tool_allowlist,
				explicit_user_file_paths=explicit_user_file_paths,
				approved_tool_names=approved_tool_names,
				audit_context=audit_context,
			)
			if general_tool_allowlist is None or normal_agent_builder.langchain_tools:
				agents.append(normal_agent_builder)

		# 他のサブエージェントも必要に応じてここで作成できます。
		return agents

	@classmethod
	def get_tools_description_all(cls, agents: list[AgentBuilder]) -> str:
		tools_description = "\n".join(f"## name: {tool.name}\n - description: {tool.description}\n - args_schema: {tool.args_schema}\n" for agent in agents for tool in agent.langchain_tools)
		return tools_description
