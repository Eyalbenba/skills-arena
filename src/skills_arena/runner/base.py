"""Base agent interface for running scenarios.

This module defines the abstract base class that all agent implementations
must inherit from. Agents are responsible for running prompts through AI
frameworks and determining which skill would be selected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Skill, SkillSelection


class BaseAgent(ABC):
    """Abstract base class for agent implementations.

    Each agent represents an AI framework that can select skills based on prompts.
    Implementations must handle converting skills to the framework's format and
    extracting which skill was selected from the response.

    Supported agents:
    - ClaudeCodeAgent: Uses the Claude Agent SDK (claude-agent-sdk)
    - MockAgent: Deterministic responses for testing
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's name identifier."""
        ...

    @abstractmethod
    async def select_skill(
        self,
        prompt: str,
        available_skills: list[Skill],
    ) -> SkillSelection:
        """Given a prompt and available skills, determine which skill to use.

        Args:
            prompt: The user message/prompt to process.
            available_skills: List of skills that could be selected.

        Returns:
            SkillSelection with the chosen skill (or None if no skill selected),
            confidence score, and reasoning.

        Raises:
            AgentError: If the agent fails to process the request.
            TimeoutError: If the request times out.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up any resources used by the agent.

        Called when the agent is no longer needed. Implementations should
        close connections, end sessions, etc.
        """
        ...

    async def __aenter__(self) -> BaseAgent:
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup resources."""
        await self.close()
