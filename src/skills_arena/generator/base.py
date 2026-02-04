"""Base generator interface for scenario generation.

This module defines the abstract base class that all scenario generators must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Scenario, Skill, Task


class BaseGenerator(ABC):
    """Abstract base class for scenario generators.

    Scenario generators create test prompts from task descriptions
    that should trigger skill selection by agents.
    """

    @abstractmethod
    async def generate(
        self,
        task: Task,
        skills: list[Skill],
        count: int = 50,
        include_adversarial: bool = True,
    ) -> list[Scenario]:
        """Generate test scenarios for the given task and skills.

        Args:
            task: The task description specifying what skills should do.
            skills: List of skills to generate scenarios for.
            count: Number of scenarios to generate.
            include_adversarial: Whether to include adversarial/edge case scenarios.

        Returns:
            List of generated Scenario objects.

        Raises:
            GeneratorError: If scenario generation fails.
        """
        pass

    def generate_sync(
        self,
        task: Task,
        skills: list[Skill],
        count: int = 50,
        include_adversarial: bool = True,
    ) -> list[Scenario]:
        """Synchronous wrapper for generate().

        For convenience when async is not needed.
        """
        import asyncio

        return asyncio.run(self.generate(task, skills, count, include_adversarial))
