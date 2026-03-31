from abc import ABC, abstractmethod


class PromptsBase(ABC):

    @abstractmethod
    def auto_approve_hitl_policy_text(self, approval_tools_text) -> str:
        pass

    @abstractmethod
    def normal_hitl_policy_text(self, approval_tools_text) -> str:
        pass

    @abstractmethod
    def tool_agent_system_prompt(
        self,
        hitl_policy_text,
        agent_name="tool_agent",
        followup_poll_interval_seconds=2.0,
        status_tail_lines=20,
        result_tail_lines=80,
    ) -> str:
        pass

    @abstractmethod
    def tool_agent_user_prompt(self, tools_description, hitl_policy_text) -> str:
        pass

    @abstractmethod
    def supervisor_hitl_policy_text(self, approval_tools_text) -> str:
        pass

    @abstractmethod
    def supervisor_normal_hitl_policy_text(self, approval_tools_text) -> str:
        pass

    @abstractmethod
    def supervisor_system_prompt(self, tools_description, supervisor_hitl_policy_text, tool_agent_names=None, routing_guidance_text=None) -> str:
        pass

    @abstractmethod
    def routing_system_prompt(self) -> str:
        pass

    @abstractmethod
    def routing_user_prompt(self, *, user_request_text: str, available_routes_text: str, context_text: str) -> str:
        pass

    def create_tools_description(self, allowed_langchain_tools) -> str:
        return "\n".join(
            f"## name: {tool.name}\n - description: {tool.description}\n - args_schema: {tool.args_schema}\n"
            for tool in allowed_langchain_tools
        )
