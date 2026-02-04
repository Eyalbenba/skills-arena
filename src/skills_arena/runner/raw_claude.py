"""Raw Claude API agent implementation.

This module implements skill selection using the direct Anthropic API
with tool_use, without the Claude Code agent loop. This is useful for:
- Comparing raw API behavior vs Claude Code agent behavior
- Environments where Claude Code SDK isn't available
- Simpler, faster skill selection testing
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from ..exceptions import AgentError, APIKeyError
from ..models import Skill, SkillSelection
from .base import BaseAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import the Anthropic SDK
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic SDK not installed. Install with: pip install anthropic")


def _skill_to_anthropic_tool(skill: Skill) -> dict[str, Any]:
    """Convert a Skill to Anthropic tool format.

    Args:
        skill: The skill to convert.

    Returns:
        Tool definition for Anthropic API.
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

    # Ensure at least one property
    if not properties:
        properties["input"] = {
            "type": "string",
            "description": "The input for this skill",
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


class RawClaudeAgent(BaseAgent):
    """Agent using direct Anthropic API with tool_use.

    This agent uses the raw Anthropic messages API with tool definitions,
    bypassing the Claude Code agent loop. This provides a simpler, faster
    way to test skill selection but loses the agent's autonomous behavior.

    Example:
        ```python
        async with RawClaudeAgent() as agent:
            selection = await agent.select_skill(
                prompt="Search for Python tutorials",
                available_skills=[web_search, code_search],
            )
        ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        api_key: str | None = None,
    ):
        """Initialize the raw Claude agent.

        Args:
            model: The Claude model to use.
            max_tokens: Maximum tokens in response.
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic SDK is required but not installed. "
                "Install with: pip install anthropic"
            )

        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return "raw-claude"

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client.

        Returns:
            AsyncAnthropic client.

        Raises:
            APIKeyError: If no API key is available.
        """
        if not self._api_key:
            raise APIKeyError("Anthropic", "ANTHROPIC_API_KEY")

        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

        return self._client

    async def select_skill(
        self,
        prompt: str,
        available_skills: list[Skill],
    ) -> SkillSelection:
        """Select a skill using the Anthropic API with tool_use.

        Args:
            prompt: The user prompt.
            available_skills: Skills available for selection.

        Returns:
            SkillSelection based on Claude's tool choice.
        """
        if not available_skills:
            return SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning="No skills available",
            )

        client = self._get_client()
        start_time = time.time()

        # Convert skills to Anthropic tool format
        tools = [_skill_to_anthropic_tool(s) for s in available_skills]
        skill_names = {s.name for s in available_skills}

        # System prompt for tool selection
        system_prompt = (
            "You are an AI assistant that helps select the most appropriate tool "
            "for a given task. When presented with a user request, analyze it and "
            "determine which tool (if any) would be best suited to handle it. "
            "If a tool is appropriate, call it. If no tool is relevant, respond "
            "with text explaining why."
        )

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=tools,
                messages=[{"role": "user", "content": prompt}],
            )

            # Log latency for debugging (unused but useful for future metrics)
            _ = (time.time() - start_time) * 1000

            # Parse response for tool use
            selected_skill: str | None = None
            confidence: float = 0.0
            reasoning: str = ""

            for block in response.content:
                if block.type == "tool_use":
                    if block.name in skill_names:
                        selected_skill = block.name
                        confidence = 0.9  # High confidence for explicit tool call
                        reasoning = f"Claude called tool: {block.name}"
                        logger.info(f"Tool selected: {block.name}")
                        break

                elif block.type == "text":
                    # Check for skill mentions in text
                    text = block.text.lower()
                    for skill_name in skill_names:
                        if skill_name.lower() in text:
                            if selected_skill is None:
                                selected_skill = skill_name
                                confidence = 0.5  # Lower confidence for mention
                                reasoning = f"Skill mentioned in text: {skill_name}"

            if not reasoning:
                reasoning = "No clear skill selection"

            return SkillSelection(
                skill=selected_skill,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=response.model_dump(),
            )

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise AgentError(
                message=f"API error: {e}",
                agent_name=self.name,
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise AgentError(
                message=str(e),
                agent_name=self.name,
            ) from e

    async def close(self) -> None:
        """Clean up the client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
