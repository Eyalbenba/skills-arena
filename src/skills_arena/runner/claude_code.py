"""Claude Code (Agent SDK) agent implementation.

This module implements the ClaudeCodeAgent which uses the official
Claude Agent SDK to run prompts and determine skill selection.

The Claude Agent SDK (formerly Claude Code SDK) provides access to the same
agentic capabilities that power Claude Code, including built-in tools and
skill/tool selection logic.

Usage:
    ```python
    async with ClaudeCodeAgent() as agent:
        selection = await agent.select_skill(
            prompt="Search the web for Python tutorials",
            available_skills=[web_search_skill, code_search_skill],
        )
        print(f"Selected: {selection.skill}")
    ```
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import AgentError, APIKeyError
from ..models import Skill, SkillSelection
from .base import BaseAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import the Claude Agent SDK
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
        UserMessage,
        query,
    )

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    logger.warning(
        "Claude Agent SDK not installed. Install with: pip install claude-agent-sdk"
    )


def _skill_to_tool_definition(skill: Skill) -> dict[str, Any]:
    """Convert a Skill to a tool definition for Claude.

    Args:
        skill: The skill to convert.

    Returns:
        Tool definition dict with name, description, and input_schema.
    """
    properties = {}
    required = []

    for param in skill.parameters:
        properties[param.name] = {
            "type": param.type,
            "description": param.description,
        }
        if param.required:
            required.append(param.name)

    # Ensure at least one property for valid schema
    if not properties:
        properties["query"] = {
            "type": "string",
            "description": "The input query or request for this skill",
        }

    return {
        "name": skill.name,
        "description": skill.description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


class ClaudeCodeAgent(BaseAgent):
    """Agent implementation using the Claude Agent SDK.

    This agent uses the official Claude Agent SDK (formerly Claude Code SDK)
    to test skill selection. It spawns a Claude session with skills defined
    as tools, then observes which tool the agent would invoke.

    The key insight is that we're testing the agent's SELECTION behavior,
    not actually executing the skills. We present skills as callable tools
    and see which one Claude chooses to invoke.

    Example:
        ```python
        async with ClaudeCodeAgent() as agent:
            selection = await agent.select_skill(
                prompt="Search the web for Python tutorials",
                available_skills=[web_search_skill, code_search_skill],
            )
            print(f"Selected: {selection.skill}")
        ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 3,
        timeout_seconds: int = 30,
        working_dir: str | Path | None = None,
        api_key: str | None = None,
    ):
        """Initialize the Claude Code agent.

        Args:
            model: The Claude model to use (default: claude-sonnet-4).
            max_turns: Maximum conversation turns (keep low for selection).
            timeout_seconds: Request timeout.
            working_dir: Working directory for Claude (defaults to temp dir).
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
        """
        if not CLAUDE_SDK_AVAILABLE:
            raise ImportError(
                "Claude Agent SDK is required but not installed. "
                "Install with: pip install claude-agent-sdk"
            )

        self.model = model
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        # Working directory - use temp dir by default for isolation
        if working_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="skills_arena_")
            self.working_dir = Path(self._temp_dir)
        else:
            self._temp_dir = None
            self.working_dir = Path(working_dir)

        # Client will be initialized on first use
        self._client: ClaudeSDKClient | None = None
        self._session_active: bool = False

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return "claude-code"

    async def _start_session(
        self,
        system_prompt: str,
        allowed_tools: list[str],
    ) -> ClaudeSDKClient:
        """Start a Claude SDK session with the given configuration.

        Args:
            system_prompt: The system prompt to use.
            allowed_tools: List of allowed tool names.

        Returns:
            Connected ClaudeSDKClient.

        Raises:
            APIKeyError: If no API key is available.
            AgentError: If session fails to start.
        """
        if not self._api_key:
            raise APIKeyError("Anthropic", "ANTHROPIC_API_KEY")

        try:
            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                max_turns=self.max_turns,
                cwd=str(self.working_dir),
                allowed_tools=allowed_tools,
            )

            self._client = ClaudeSDKClient(options=options)
            await self._client.connect()
            self._session_active = True

            logger.debug(f"Claude SDK session started (max_turns={self.max_turns})")
            return self._client

        except Exception as e:
            logger.error(f"Failed to start Claude SDK session: {e}")
            self._session_active = False
            raise AgentError(
                message=f"Failed to start session: {e}",
                agent_name=self.name,
            ) from e

    async def select_skill(
        self,
        prompt: str,
        available_skills: list[Skill],
    ) -> SkillSelection:
        """Run a prompt through Claude Code and determine skill selection.

        This method:
        1. Converts skills to tool definitions
        2. Creates a system prompt that asks Claude to select a tool
        3. Runs the query and observes which tool is called

        Args:
            prompt: The user prompt to process.
            available_skills: Skills available for selection.

        Returns:
            SkillSelection with the chosen skill.
        """
        if not self._api_key:
            raise APIKeyError("Anthropic", "ANTHROPIC_API_KEY")

        if not available_skills:
            return SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning="No skills available for selection",
            )

        start_time = time.time()

        # Get skill names for matching
        skill_names = [s.name for s in available_skills]

        # Build the system prompt for skill selection
        system_prompt = self._build_selection_system_prompt(available_skills)

        # Build allowed tools list - our skill tools plus minimal built-ins
        allowed_tools = [s.name for s in available_skills]

        try:
            # Use the functional query API for simplicity and stateless operation
            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                max_turns=self.max_turns,
                cwd=str(self.working_dir),
                allowed_tools=allowed_tools,
            )

            selected_skill: str | None = None
            confidence: float = 0.0
            reasoning: str = ""
            raw_response: list[Any] = []
            tool_inputs: dict[str, Any] = {}

            # Query Claude and look for skill selection
            async for message in query(prompt=prompt, options=options):
                raw_response.append(message)

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            # Check if the tool call matches one of our skills
                            if block.name in skill_names:
                                selected_skill = block.name
                                confidence = 0.95  # High confidence for explicit tool call
                                tool_inputs = block.input if hasattr(block, "input") else {}
                                reasoning = (
                                    f"Claude explicitly called {block.name} "
                                    f"with inputs: {json.dumps(tool_inputs)}"
                                )
                                logger.info(f"Tool call detected: {block.name}")

                        elif isinstance(block, TextBlock):
                            # Check for skill mentions in text (lower confidence)
                            text = block.text.lower()
                            for skill_name in skill_names:
                                if skill_name.lower() in text:
                                    if selected_skill is None:
                                        selected_skill = skill_name
                                        confidence = 0.6  # Lower confidence for text mention
                                        reasoning = f"Claude mentioned {skill_name} in response"

                elif isinstance(message, ResultMessage):
                    if message.is_error:
                        logger.warning(f"Claude returned error: {message.result}")
                        # If error, reduce confidence
                        if confidence > 0:
                            confidence *= 0.5
                            reasoning += " (with errors)"

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Skill selection completed in {latency_ms:.1f}ms")

            return SkillSelection(
                skill=selected_skill,
                confidence=confidence,
                reasoning=reasoning or "No clear skill selection detected",
                raw_response=raw_response,
            )

        except Exception as e:
            logger.error(f"Error during skill selection: {e}")
            raise AgentError(
                message=str(e),
                agent_name=self.name,
            ) from e

    def _build_selection_system_prompt(self, skills: list[Skill]) -> str:
        """Build a system prompt that frames the task as tool selection.

        Args:
            skills: Available skills to choose from.

        Returns:
            System prompt string.
        """
        skill_descriptions = []
        for skill in skills:
            desc = f"- **{skill.name}**: {skill.description}"
            if skill.when_to_use:
                desc += f"\n  Use when: {', '.join(skill.when_to_use[:3])}"
            skill_descriptions.append(desc)

        return f"""You are an AI assistant with access to specialized skills/tools.
Your task is to analyze the user's request and determine which skill would be most appropriate.

## Available Skills

{chr(10).join(skill_descriptions)}

## Instructions

1. Read the user's request carefully
2. Determine which skill best matches the request based on its description and use cases
3. If a skill matches, call it with appropriate parameters
4. If no skill is clearly relevant, respond normally without calling any skill

Think about which skill's description best matches what the user is asking for.
Only call a skill if it's clearly relevant to the request.
When calling a skill, provide relevant parameters based on the user's request."""

    async def close(self) -> None:
        """Clean up resources."""
        if self._client is not None and self._session_active:
            try:
                await self._client.disconnect()
                logger.debug("Claude SDK session closed")
            except Exception as e:
                logger.warning(f"Error disconnecting client: {e}")
            self._client = None
            self._session_active = False

        # Clean up temp directory if we created one
        if self._temp_dir is not None:
            import shutil

            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error cleaning up temp dir: {e}")
            self._temp_dir = None


class ClaudeCodeAgentWithTools(ClaudeCodeAgent):
    """Extended Claude Code agent that uses persistent sessions.

    This variant maintains a persistent ClaudeSDKClient session across
    multiple skill selection calls, which can improve performance when
    running many evaluations. The session is reused until close() is called.

    Example:
        ```python
        async with ClaudeCodeAgentWithTools() as agent:
            # All selections reuse the same session
            for scenario in scenarios:
                selection = await agent.select_skill(
                    prompt=scenario.prompt,
                    available_skills=skills,
                )
        ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 5,  # Higher default for persistent sessions
        timeout_seconds: int = 60,
        working_dir: str | Path | None = None,
        api_key: str | None = None,
    ):
        """Initialize the Claude Code agent with persistent sessions.

        Args:
            model: The Claude model to use (default: claude-sonnet-4).
            max_turns: Maximum conversation turns per query.
            timeout_seconds: Request timeout.
            working_dir: Working directory for Claude (defaults to temp dir).
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
        """
        super().__init__(
            model=model,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            working_dir=working_dir,
            api_key=api_key,
        )
        self._cached_system_prompt: str | None = None
        self._cached_skill_names: list[str] = []

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return "claude-code-persistent"

    async def select_skill(
        self,
        prompt: str,
        available_skills: list[Skill],
    ) -> SkillSelection:
        """Run skill selection with persistent session management.

        This method uses a persistent session when skills remain the same
        across calls, resetting the session only when skills change.

        Args:
            prompt: The user prompt to process.
            available_skills: Skills available for selection.

        Returns:
            SkillSelection with the chosen skill.
        """
        if not self._api_key:
            raise APIKeyError("Anthropic", "ANTHROPIC_API_KEY")

        if not available_skills:
            return SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning="No skills available for selection",
            )

        start_time = time.time()

        # Get skill names for matching
        skill_names = [s.name for s in available_skills]

        # Check if we need to rebuild the session (skills changed)
        if skill_names != self._cached_skill_names:
            if self._session_active:
                await self.close()
            self._cached_system_prompt = self._build_selection_system_prompt(available_skills)
            self._cached_skill_names = skill_names.copy()

        # Ensure session is active
        if not self._session_active:
            await self._start_session(
                system_prompt=self._cached_system_prompt or "",
                allowed_tools=skill_names,
            )

        try:
            # Send query to existing session
            await self._client.query(prompt)

            selected_skill: str | None = None
            confidence: float = 0.0
            reasoning: str = ""
            raw_response: list[Any] = []
            tool_inputs: dict[str, Any] = {}

            # Collect response
            async for message in self._client.receive_response():
                raw_response.append(message)

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            if block.name in skill_names:
                                selected_skill = block.name
                                confidence = 0.95
                                tool_inputs = block.input if hasattr(block, "input") else {}
                                reasoning = (
                                    f"Claude explicitly called {block.name} "
                                    f"with inputs: {json.dumps(tool_inputs)}"
                                )
                                logger.info(f"Tool call detected: {block.name}")

                        elif isinstance(block, TextBlock):
                            text = block.text.lower()
                            for skill_name in skill_names:
                                if skill_name.lower() in text:
                                    if selected_skill is None:
                                        selected_skill = skill_name
                                        confidence = 0.6
                                        reasoning = f"Claude mentioned {skill_name} in response"

                elif isinstance(message, ResultMessage):
                    if message.is_error:
                        logger.warning(f"Claude returned error: {message.result}")
                        if confidence > 0:
                            confidence *= 0.5
                            reasoning += " (with errors)"

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Skill selection completed in {latency_ms:.1f}ms (persistent)")

            return SkillSelection(
                skill=selected_skill,
                confidence=confidence,
                reasoning=reasoning or "No clear skill selection detected",
                raw_response=raw_response,
            )

        except Exception as e:
            logger.error(f"Error during skill selection: {e}")
            # Reset session on error
            self._session_active = False
            self._client = None
            raise AgentError(
                message=str(e),
                agent_name=self.name,
            ) from e
