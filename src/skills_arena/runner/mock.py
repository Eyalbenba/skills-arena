"""Mock agent implementation for testing.

This module provides a deterministic mock agent that can be used
for testing the Skills Arena without making actual API calls.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from ..models import Skill, SkillSelection
from .base import BaseAgent

if TYPE_CHECKING:
    pass


class MockAgent(BaseAgent):
    """Mock agent for testing purposes.

    This agent provides deterministic skill selection based on simple
    keyword matching. It's useful for:
    - Unit testing without API calls
    - Development and debugging
    - CI/CD pipelines

    Example:
        ```python
        agent = MockAgent(mode="keyword")
        selection = await agent.select_skill(
            prompt="Search the web for Python docs",
            available_skills=[web_search, code_search],
        )
        # Returns web_search if "search" or "web" in skill description
        ```
    """

    def __init__(
        self,
        mode: str = "keyword",
        default_confidence: float = 0.8,
        seed: int | None = None,
    ):
        """Initialize the mock agent.

        Args:
            mode: Selection mode:
                - "keyword": Match based on keyword overlap
                - "first": Always select the first skill
                - "random": Randomly select a skill
                - "none": Never select a skill
            default_confidence: Default confidence score for selections.
            seed: Random seed for reproducibility (only used in "random" mode).
        """
        self.mode = mode
        self.default_confidence = default_confidence
        self._random = random.Random(seed) if seed else random.Random()

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return "mock"

    async def select_skill(
        self,
        prompt: str,
        available_skills: list[Skill],
    ) -> SkillSelection:
        """Select a skill based on the configured mode.

        Args:
            prompt: The user prompt.
            available_skills: Skills available for selection.

        Returns:
            SkillSelection based on the configured mode.
        """
        if not available_skills:
            return SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning="No skills available",
            )

        if self.mode == "none":
            return SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning="Mock mode: none",
            )

        if self.mode == "first":
            return SkillSelection(
                skill=available_skills[0].name,
                confidence=self.default_confidence,
                reasoning="Mock mode: first skill selected",
            )

        if self.mode == "random":
            selected = self._random.choice(available_skills)
            return SkillSelection(
                skill=selected.name,
                confidence=self._random.uniform(0.5, 1.0),
                reasoning=f"Mock mode: randomly selected {selected.name}",
            )

        # Keyword matching mode (default)
        return self._keyword_match(prompt, available_skills)

    def _keyword_match(
        self,
        prompt: str,
        skills: list[Skill],
    ) -> SkillSelection:
        """Match skills based on keyword overlap with the prompt.

        Args:
            prompt: The user prompt.
            skills: Available skills.

        Returns:
            SkillSelection based on keyword matching.
        """
        prompt_lower = prompt.lower()
        prompt_words = set(prompt_lower.split())

        best_skill: Skill | None = None
        best_score = 0
        best_matches: list[str] = []

        for skill in skills:
            # Build keyword set from skill
            skill_text = f"{skill.name} {skill.description}"
            skill_text += " " + " ".join(skill.when_to_use)
            skill_words = set(skill_text.lower().split())

            # Calculate overlap
            matches = prompt_words & skill_words
            score = len(matches)

            # Check for phrase matches in description
            if skill.name.lower() in prompt_lower:
                score += 5  # Boost for name match

            for phrase in skill.when_to_use:
                if phrase.lower() in prompt_lower:
                    score += 3  # Boost for use-case match

            if score > best_score:
                best_score = score
                best_skill = skill
                best_matches = list(matches)

        if best_skill and best_score > 0:
            confidence = min(0.95, 0.3 + (best_score * 0.1))
            return SkillSelection(
                skill=best_skill.name,
                confidence=confidence,
                reasoning=f"Keyword matches: {', '.join(best_matches[:5])}",
            )

        # No good match - return None or random
        return SkillSelection(
            skill=None,
            confidence=0.0,
            reasoning="No keyword matches found",
        )

    async def close(self) -> None:
        """No cleanup needed for mock agent."""
        pass


class ScriptedMockAgent(MockAgent):
    """Mock agent with scripted responses for testing specific scenarios.

    This agent returns predefined responses based on prompt patterns,
    allowing precise control over test outcomes.

    Example:
        ```python
        agent = ScriptedMockAgent(
            responses={
                "search.*web": ("web_search", 0.9, "Matched 'search web' pattern"),
                "analyze.*data": ("data_analysis", 0.85, "Matched 'analyze data' pattern"),
            }
        )
        ```
    """

    def __init__(
        self,
        responses: dict[str, tuple[str | None, float, str]] | None = None,
        fallback_mode: str = "keyword",
    ):
        """Initialize with scripted responses.

        Args:
            responses: Dict mapping regex patterns to (skill_name, confidence, reasoning).
            fallback_mode: Mode to use if no pattern matches.
        """
        super().__init__(mode=fallback_mode)
        self._responses = responses or {}
        self._compiled_patterns: list[tuple[Any, tuple[str | None, float, str]]] = []

        # Compile regex patterns
        import re

        for pattern, response in self._responses.items():
            self._compiled_patterns.append(
                (re.compile(pattern, re.IGNORECASE), response)
            )

    async def select_skill(
        self,
        prompt: str,
        available_skills: list[Skill],
    ) -> SkillSelection:
        """Select skill based on scripted patterns.

        Args:
            prompt: The user prompt.
            available_skills: Skills available for selection.

        Returns:
            Scripted SkillSelection or fallback.
        """
        # Check scripted patterns first
        for pattern, (skill_name, confidence, reasoning) in self._compiled_patterns:
            if pattern.search(prompt):
                # Verify the skill exists
                skill_names = {s.name for s in available_skills}
                if skill_name is None or skill_name in skill_names:
                    return SkillSelection(
                        skill=skill_name,
                        confidence=confidence,
                        reasoning=reasoning,
                    )

        # Fallback to parent behavior
        return await super().select_skill(prompt, available_skills)
