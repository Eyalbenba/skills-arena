"""Agent runner module.

Provides agent implementations for running scenarios:
- ClaudeCodeAgent: Uses the Claude Agent SDK (claude-agent-sdk)
- RawClaudeAgent: Direct Anthropic API with tool_use
- RawOpenAIAgent: Direct OpenAI API with function_calling
- MockAgent: Deterministic responses for testing

Example:
    ```python
    from skills_arena.runner import ClaudeCodeAgent, MockAgent

    # Use the Claude Code agent
    async with ClaudeCodeAgent() as agent:
        selection = await agent.select_skill(
            prompt="Search the web for Python tutorials",
            available_skills=[web_search_skill, code_search_skill],
        )

    # Or use a mock for testing
    agent = MockAgent(mode="keyword")
    selection = await agent.select_skill(prompt, skills)
    ```
"""

from .base import BaseAgent
from .mock import MockAgent, ScriptedMockAgent

# Import real agents with graceful fallback
try:
    from .claude_code import ClaudeCodeAgent, ClaudeCodeAgentWithTools
except ImportError:
    ClaudeCodeAgent = None  # type: ignore
    ClaudeCodeAgentWithTools = None  # type: ignore

try:
    from .raw_claude import RawClaudeAgent
except ImportError:
    RawClaudeAgent = None  # type: ignore

try:
    from .raw_openai import RawOpenAIAgent
except ImportError:
    RawOpenAIAgent = None  # type: ignore


def get_agent(name: str, **kwargs) -> BaseAgent:
    """Factory function to get an agent by name.

    Args:
        name: Agent name. One of:
            - "claude-code": Claude Agent SDK
            - "raw-claude": Direct Anthropic API
            - "raw-openai": Direct OpenAI API
            - "mock": Mock agent for testing
        **kwargs: Additional arguments passed to the agent constructor.

    Returns:
        Initialized agent instance.

    Raises:
        ValueError: If agent name is not recognized.
        ImportError: If required SDK is not installed.

    Example:
        ```python
        agent = get_agent("claude-code", model="claude-sonnet-4-20250514")
        ```
    """
    agents = {
        "claude-code": ClaudeCodeAgent,
        "raw-claude": RawClaudeAgent,
        "raw-openai": RawOpenAIAgent,
        "mock": MockAgent,
    }

    if name not in agents:
        valid = list(agents.keys())
        raise ValueError(f"Unknown agent '{name}'. Valid agents: {valid}")

    agent_class = agents[name]

    if agent_class is None:
        raise ImportError(
            f"Agent '{name}' requires its SDK to be installed. "
            f"Check the documentation for installation instructions."
        )

    return agent_class(**kwargs)


__all__ = [
    # Base
    "BaseAgent",
    # Agents
    "ClaudeCodeAgent",
    "ClaudeCodeAgentWithTools",
    "RawClaudeAgent",
    "RawOpenAIAgent",
    "MockAgent",
    "ScriptedMockAgent",
    # Factory
    "get_agent",
]
