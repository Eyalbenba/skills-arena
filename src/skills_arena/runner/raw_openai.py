"""Raw OpenAI API agent implementation.

This module implements skill selection using the direct OpenAI API
with function_calling. This is useful for:
- Testing skill selection with OpenAI models
- Comparing OpenAI vs Claude behavior
- Environments without Anthropic access
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

# Try to import the OpenAI SDK
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not installed. Install with: pip install openai")


def _skill_to_openai_function(skill: Skill) -> dict[str, Any]:
    """Convert a Skill to OpenAI function format.

    Args:
        skill: The skill to convert.

    Returns:
        Function definition for OpenAI API.
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
        "type": "function",
        "function": {
            "name": skill.name,
            "description": skill.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class RawOpenAIAgent(BaseAgent):
    """Agent using direct OpenAI API with function_calling.

    This agent uses the OpenAI chat completions API with function definitions
    to test skill selection behavior with GPT models.

    Example:
        ```python
        async with RawOpenAIAgent() as agent:
            selection = await agent.select_skill(
                prompt="Search for Python tutorials",
                available_skills=[web_search, code_search],
            )
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        api_key: str | None = None,
    ):
        """Initialize the raw OpenAI agent.

        Args:
            model: The OpenAI model to use.
            max_tokens: Maximum tokens in response.
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI SDK is required but not installed. "
                "Install with: pip install openai"
            )

        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: openai.AsyncOpenAI | None = None

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return "raw-openai"

    def _get_client(self) -> openai.AsyncOpenAI:
        """Get or create the OpenAI client.

        Returns:
            AsyncOpenAI client.

        Raises:
            APIKeyError: If no API key is available.
        """
        if not self._api_key:
            raise APIKeyError("OpenAI", "OPENAI_API_KEY")

        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self._api_key)

        return self._client

    async def select_skill(
        self,
        prompt: str,
        available_skills: list[Skill],
    ) -> SkillSelection:
        """Select a skill using the OpenAI API with function_calling.

        Args:
            prompt: The user prompt.
            available_skills: Skills available for selection.

        Returns:
            SkillSelection based on OpenAI's function choice.
        """
        if not available_skills:
            return SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning="No skills available",
            )

        client = self._get_client()
        start_time = time.time()

        # Convert skills to OpenAI function format
        tools = [_skill_to_openai_function(s) for s in available_skills]
        skill_names = {s.name for s in available_skills}

        # System prompt for tool selection
        system_prompt = (
            "You are an AI assistant that helps select the most appropriate tool "
            "for a given task. When presented with a user request, analyze it and "
            "determine which tool (if any) would be best suited to handle it. "
            "If a tool is appropriate, call it. If no tool is relevant, explain why."
        )

        try:
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                tool_choice="auto",
            )

            # Log latency for debugging (unused but useful for future metrics)
            _ = (time.time() - start_time) * 1000

            # Parse response for function calls
            selected_skill: str | None = None
            confidence: float = 0.0
            reasoning: str = ""

            message = response.choices[0].message

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    if func_name in skill_names:
                        selected_skill = func_name
                        confidence = 0.9  # High confidence for explicit function call
                        reasoning = f"OpenAI called function: {func_name}"
                        logger.info(f"Function selected: {func_name}")
                        break

            elif message.content:
                # Check for skill mentions in text
                text = message.content.lower()
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

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
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
