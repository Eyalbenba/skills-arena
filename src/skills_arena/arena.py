"""Main Arena class for Skills Arena.

This module provides the primary entry point for the Skills Arena SDK.
The Arena class orchestrates skill parsing, scenario generation, agent running,
and result scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import ArenaConfig, Config, ProgressCallback
from .exceptions import NoSkillsError
from .models import (
    BattleResult,
    ComparisonResult,
    EvaluationResult,
    Insight,
    Skill,
    Task,
)

if TYPE_CHECKING:
    pass


class Arena:
    """Main entry point for Skills Arena.

    The Arena class provides methods to evaluate, compare, and run battle royale
    competitions between AI agent skills.

    Example:
        ```python
        from skills_arena import Arena

        # Simple evaluation
        arena = Arena()
        results = arena.evaluate("./my-skill.md", task="web search")
        print(results.score)

        # Compare skills
        results = arena.compare(
            skills=["./my-skill.md", "./competitor.md"],
            task="web search",
        )
        print(results.winner)
        ```
    """

    def __init__(self, config: Config | None = None):
        """Initialize the Arena.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or Config()
        self._parsed_skills: dict[str, Skill] = {}

    @classmethod
    def from_config(cls, path: str | Path) -> Arena:
        """Create an Arena from a YAML configuration file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Arena instance configured from the file.

        Example:
            ```python
            arena = Arena.from_config("./arena.yaml")
            results = arena.run()
            ```
        """
        arena_config = ArenaConfig.from_yaml(path)
        return cls(config=arena_config.evaluation)

    def evaluate(
        self,
        skill: str | Skill,
        task: str | Task,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> EvaluationResult:
        """Evaluate a single skill's performance.

        This method measures how well a skill performs when presented to agents
        alongside the task description. It generates test scenarios and measures
        selection rate, accuracy, and other metrics.

        Args:
            skill: Path to skill file or Skill object.
            task: Task description (string or Task object).
            on_progress: Optional callback for progress updates.

        Returns:
            EvaluationResult with score, grade, and insights.

        Example:
            ```python
            results = arena.evaluate("./my-skill.md", task="web search")
            print(f"Score: {results.score}")
            print(f"Grade: {results.grade}")
            ```
        """
        # Parse skill if needed (underscore prefix indicates intentionally unused in stub)
        _parsed_skill = self._ensure_skill(skill)

        # Convert task to Task object if needed
        _parsed_task = task if isinstance(task, Task) else Task.from_string(task)

        # Silence unused variable warnings for stub - will be used in Phase 1
        del _parsed_skill, _parsed_task, on_progress

        # TODO: Phase 1 implementation
        # 1. Generate scenarios using Generator
        # 2. Run scenarios through agents using Runner
        # 3. Score results using Scorer
        # 4. Generate insights

        raise NotImplementedError(
            "evaluate() is not yet implemented. "
            "This will be available in Phase 1 of development."
        )

    async def evaluate_async(
        self,
        skill: str | Skill,
        task: str | Task,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> EvaluationResult:
        """Async version of evaluate().

        See evaluate() for full documentation.
        """
        raise NotImplementedError(
            "evaluate_async() is not yet implemented. "
            "This will be available in Phase 1 of development."
        )

    def compare(
        self,
        skills: list[str | Skill],
        task: str | Task,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ComparisonResult:
        """Compare multiple skills head-to-head.

        This method runs scenarios where multiple skills compete for selection.
        It measures which skill agents prefer and why.

        Args:
            skills: List of skill paths or Skill objects (minimum 2).
            task: Task description (string or Task object).
            on_progress: Optional callback for progress updates.

        Returns:
            ComparisonResult with winner, selection rates, and insights.

        Example:
            ```python
            results = arena.compare(
                skills=["./my-skill.md", "./competitor.md"],
                task="web search and content extraction",
            )
            print(f"Winner: {results.winner}")
            print(f"Selection rates: {results.selection_rates}")
            ```
        """
        if len(skills) < 2:
            raise NoSkillsError("compare (requires at least 2 skills)")

        # Parse all skills (underscore prefix indicates intentionally unused in stub)
        _parsed_skills = [self._ensure_skill(s) for s in skills]

        # Convert task to Task object if needed
        _parsed_task = task if isinstance(task, Task) else Task.from_string(task)

        # Silence unused variable warnings for stub - will be used in Phase 1
        del _parsed_skills, _parsed_task, on_progress

        # TODO: Phase 1 implementation
        # 1. Generate scenarios for all skills
        # 2. Run head-to-head comparisons
        # 3. Score and rank results
        # 4. Generate comparative insights

        raise NotImplementedError(
            "compare() is not yet implemented. "
            "This will be available in Phase 1 of development."
        )

    async def compare_async(
        self,
        skills: list[str | Skill],
        task: str | Task,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ComparisonResult:
        """Async version of compare().

        See compare() for full documentation.
        """
        raise NotImplementedError(
            "compare_async() is not yet implemented. "
            "This will be available in Phase 1 of development."
        )

    def battle_royale(
        self,
        skills: list[str | Skill],
        task: str | Task,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> BattleResult:
        """Run a battle royale competition between multiple skills.

        This method runs a full tournament where all skills compete against
        each other. Results include ELO ratings and a leaderboard.

        Args:
            skills: List of skill paths or Skill objects (minimum 2).
            task: Task description (string or Task object).
            on_progress: Optional callback for progress updates.

        Returns:
            BattleResult with leaderboard, ELO ratings, and matchups.

        Example:
            ```python
            results = arena.battle_royale(
                skills=["./skill-a.md", "./skill-b.md", "./skill-c.md"],
                task="data analysis",
            )
            print(results.leaderboard)
            ```
        """
        if len(skills) < 2:
            raise NoSkillsError("battle_royale (requires at least 2 skills)")

        # Parse all skills (underscore prefix indicates intentionally unused in stub)
        _parsed_skills = [self._ensure_skill(s) for s in skills]

        # Convert task to Task object if needed
        _parsed_task = task if isinstance(task, Task) else Task.from_string(task)

        # Silence unused variable warnings for stub - will be used in Phase 3
        del _parsed_skills, _parsed_task, on_progress

        # TODO: Phase 3 implementation
        # 1. Generate scenarios for all skills
        # 2. Run round-robin matchups
        # 3. Calculate ELO ratings
        # 4. Generate leaderboard and insights

        raise NotImplementedError(
            "battle_royale() is not yet implemented. "
            "This will be available in Phase 3 of development."
        )

    async def battle_royale_async(
        self,
        skills: list[str | Skill],
        task: str | Task,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> BattleResult:
        """Async version of battle_royale().

        See battle_royale() for full documentation.
        """
        raise NotImplementedError(
            "battle_royale_async() is not yet implemented. "
            "This will be available in Phase 3 of development."
        )

    def insights(
        self,
        skill: str | Skill,
        results: EvaluationResult | ComparisonResult | None = None,
    ) -> list[Insight]:
        """Generate insights for a skill.

        If results are provided, insights will be based on evaluation data.
        Otherwise, static analysis of the skill description is performed.

        Args:
            skill: Path to skill file or Skill object.
            results: Optional evaluation/comparison results.

        Returns:
            List of actionable insights.
        """
        # Parse skill (underscore prefix indicates intentionally unused in stub)
        _parsed_skill = self._ensure_skill(skill)

        # Silence unused variable warnings for stub - will be used in Phase 4
        del _parsed_skill, results

        # TODO: Phase 4 implementation
        # 1. Analyze skill description
        # 2. Compare against best practices
        # 3. If results provided, analyze performance patterns
        # 4. Generate optimization suggestions

        raise NotImplementedError(
            "insights() is not yet implemented. "
            "This will be available in Phase 4 of development."
        )

    def run(self) -> EvaluationResult | ComparisonResult | BattleResult:
        """Run evaluation based on loaded configuration.

        This method is used with from_config() to run the evaluation
        specified in a YAML configuration file.

        Returns:
            Appropriate result type based on configuration mode.

        Example:
            ```python
            arena = Arena.from_config("./arena.yaml")
            results = arena.run()
            ```
        """
        raise NotImplementedError(
            "run() is not yet implemented. "
            "This will be available in Phase 1 of development."
        )

    def _ensure_skill(self, skill: str | Skill) -> Skill:
        """Ensure we have a parsed Skill object.

        Args:
            skill: Path to skill file or Skill object.

        Returns:
            Parsed Skill object.
        """
        if isinstance(skill, Skill):
            return skill

        # Check cache
        if skill in self._parsed_skills:
            return self._parsed_skills[skill]

        # Parse the skill file
        # TODO: Use Parser module once implemented
        raise NotImplementedError(
            "Skill parsing is not yet implemented. "
            "Pass a Skill object directly for now."
        )
