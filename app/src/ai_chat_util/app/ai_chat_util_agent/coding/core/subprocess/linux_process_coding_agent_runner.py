from __future__ import annotations

from .subprocess_coding_agent_runner import SubprocessCodingAgentRunner


class LinuxProcessCodingAgentRunner(SubprocessCodingAgentRunner):
    """Linux/Unix-specific local process runner.

    Currently identical to the common implementation; split into its own module
    to allow platform-specific behavior without cluttering the shared runner.
    """

    # No overrides for now.
    pass
