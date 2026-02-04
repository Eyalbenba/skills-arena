"""Main Arena class for Skills Arena.

This module provides the primary entry point for the Skills Arena SDK.
The Arena class orchestrates skill parsing, scenario generation, agent running,
and result scoring.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ArenaConfig, Config, ProgressCallback
from .exceptions import NoSkillsError
from .generator import LLMGenerator, MockGenerator
from .models import (
    BattleResult,
    ComparisonResult,
    EvaluationResult,
    Insight,
    Progress,
    SelectionResult,
    Skill,
    SkillSelection,
    Task,
)
from .parser import Parser
from .runner import MockAgent, get_agent
from .scorer import Scorer

if TYPE_CHECKING:
    from .generator import BaseGenerator
    from .runner import BaseAgent


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
        self._generator: BaseGenerator | None = None
        self._agents: dict[str, BaseAgent] = {}

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

    def _get_generator(self) -> BaseGenerator:
        """Get or create the scenario generator."""
        if self._generator is None:
            if "mock" in self.config.agents:
                self._generator = MockGenerator()
            else:
                self._generator = LLMGenerator(
                    model=self.config.generator_model,
                    temperature=self.config.temperature,
                    api_key=self.config.anthropic_api_key,
                    config=self.config,
                )
        return self._generator

    def _get_agent(self, name: str) -> BaseAgent:
        """Get or create an agent by name."""
        if name not in self._agents:
            self._agents[name] = get_agent(name)
        return self._agents[name]

    async def _cleanup_agents(self) -> None:
        """Clean up all agent resources."""
        for agent in self._agents.values():
            try:
                await agent.close()
            except Exception:
                pass
        self._agents.clear()

    async def _run_scenario(
        self,
        agent: BaseAgent,
        prompt: str,
        skills: list[Skill],
        expected_skill: str,
        scenario_id: str,
    ) -> SelectionResult:
        """Run a single scenario through an agent.

        Args:
            agent: The agent to use.
            prompt: The scenario prompt.
            skills: Available skills for selection.
            expected_skill: The expected skill name.
            scenario_id: Unique scenario identifier.

        Returns:
            SelectionResult with the agent's selection.
        """
        from .models import Scenario, Difficulty

        start_time = time.time()

        try:
            selection = await asyncio.wait_for(
                agent.select_skill(prompt, skills),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            selection = SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning="Request timed out",
            )
        except Exception as e:
            selection = SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
            )

        latency_ms = (time.time() - start_time) * 1000

        # Create a minimal scenario object for the result
        scenario = Scenario(
            id=scenario_id,
            prompt=prompt,
            expected_skill=expected_skill,
            difficulty=Difficulty.MEDIUM,
        )

        return SelectionResult(
            scenario=scenario,
            selection=selection,
            agent_name=agent.name,
            is_correct=selection.skill == expected_skill,
            latency_ms=latency_ms,
        )

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
        return asyncio.run(self.evaluate_async(skill, task, on_progress=on_progress))

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
        # Parse skill if needed
        parsed_skill = self._ensure_skill(skill)

        # Convert task to Task object if needed
        parsed_task = task if isinstance(task, Task) else Task.from_string(task)

        # Report progress: parsing complete
        if on_progress:
            on_progress(Progress(stage="parsing", percent=10, message="Skill parsed"))

        # Generate scenarios
        generator = self._get_generator()
        scenarios = await generator.generate(
            task=parsed_task,
            skills=[parsed_skill],
            count=self.config.scenarios,
            include_adversarial=self.config.include_adversarial,
        )

        if on_progress:
            on_progress(Progress(
                stage="generation",
                percent=30,
                message=f"Generated {len(scenarios)} scenarios",
            ))

        # Run scenarios through agents
        results: list[SelectionResult] = []
        total_runs = len(scenarios) * len(self.config.agents)
        completed = 0

        try:
            for agent_name in self.config.agents:
                agent = self._get_agent(agent_name)

                for scenario in scenarios:
                    result = await self._run_scenario(
                        agent=agent,
                        prompt=scenario.prompt,
                        skills=[parsed_skill],
                        expected_skill=scenario.expected_skill,
                        scenario_id=scenario.id,
                    )
                    # Update with full scenario info
                    result.scenario = scenario
                    results.append(result)

                    completed += 1
                    if on_progress:
                        percent = 30 + (completed / total_runs) * 60
                        on_progress(Progress(
                            stage="running",
                            percent=percent,
                            message=f"Processed {completed}/{total_runs} scenarios",
                        ))
        finally:
            await self._cleanup_agents()

        # Score results
        evaluation_result = Scorer.score_evaluation(parsed_skill, results)

        if on_progress:
            on_progress(Progress(stage="complete", percent=100, message="Evaluation complete"))

        return evaluation_result

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

        return asyncio.run(self.compare_async(skills, task, on_progress=on_progress))

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
        if len(skills) < 2:
            raise NoSkillsError("compare (requires at least 2 skills)")

        # Parse all skills
        parsed_skills = [self._ensure_skill(s) for s in skills]

        # Convert task to Task object if needed
        parsed_task = task if isinstance(task, Task) else Task.from_string(task)

        # Report progress: parsing complete
        if on_progress:
            on_progress(Progress(
                stage="parsing",
                percent=10,
                message=f"Parsed {len(parsed_skills)} skills",
            ))

        # Generate scenarios for all skills
        generator = self._get_generator()
        scenarios = await generator.generate(
            task=parsed_task,
            skills=parsed_skills,
            count=self.config.scenarios,
            include_adversarial=self.config.include_adversarial,
        )

        if on_progress:
            on_progress(Progress(
                stage="generation",
                percent=30,
                message=f"Generated {len(scenarios)} scenarios",
            ))

        # Run scenarios through agents with ALL skills available
        results: list[SelectionResult] = []
        total_runs = len(scenarios) * len(self.config.agents)
        completed = 0

        try:
            for agent_name in self.config.agents:
                agent = self._get_agent(agent_name)

                for scenario in scenarios:
                    result = await self._run_scenario(
                        agent=agent,
                        prompt=scenario.prompt,
                        skills=parsed_skills,  # All skills available
                        expected_skill=scenario.expected_skill,
                        scenario_id=scenario.id,
                    )
                    # Update with full scenario info
                    result.scenario = scenario
                    results.append(result)

                    completed += 1
                    if on_progress:
                        percent = 30 + (completed / total_runs) * 60
                        on_progress(Progress(
                            stage="running",
                            percent=percent,
                            message=f"Processed {completed}/{total_runs} scenarios",
                        ))
        finally:
            await self._cleanup_agents()

        # Score comparison
        comparison_result = Scorer.score_comparison(parsed_skills, results)

        if on_progress:
            on_progress(Progress(stage="complete", percent=100, message="Comparison complete"))

        return comparison_result

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

        return asyncio.run(self.battle_royale_async(skills, task, on_progress=on_progress))

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
        if len(skills) < 2:
            raise NoSkillsError("battle_royale (requires at least 2 skills)")

        # Parse all skills
        parsed_skills = [self._ensure_skill(s) for s in skills]

        # Convert task to Task object if needed
        parsed_task = task if isinstance(task, Task) else Task.from_string(task)

        if on_progress:
            on_progress(Progress(
                stage="parsing",
                percent=10,
                message=f"Parsed {len(parsed_skills)} skills",
            ))

        # Generate scenarios for all skills
        generator = self._get_generator()
        scenarios = await generator.generate(
            task=parsed_task,
            skills=parsed_skills,
            count=self.config.scenarios,
            include_adversarial=self.config.include_adversarial,
        )

        if on_progress:
            on_progress(Progress(
                stage="generation",
                percent=30,
                message=f"Generated {len(scenarios)} scenarios",
            ))

        # Run scenarios through agents
        results: list[SelectionResult] = []
        total_runs = len(scenarios) * len(self.config.agents)
        completed = 0

        try:
            for agent_name in self.config.agents:
                agent = self._get_agent(agent_name)

                for scenario in scenarios:
                    result = await self._run_scenario(
                        agent=agent,
                        prompt=scenario.prompt,
                        skills=parsed_skills,
                        expected_skill=scenario.expected_skill,
                        scenario_id=scenario.id,
                    )
                    result.scenario = scenario
                    results.append(result)

                    completed += 1
                    if on_progress:
                        percent = 30 + (completed / total_runs) * 60
                        on_progress(Progress(
                            stage="running",
                            percent=percent,
                            message=f"Processed {completed}/{total_runs} scenarios",
                        ))
        finally:
            await self._cleanup_agents()

        # Score battle
        battle_result = Scorer.score_battle(
            parsed_skills,
            results,
            k_factor=self.config.elo_k_factor,
        )

        if on_progress:
            on_progress(Progress(stage="complete", percent=100, message="Battle complete"))

        return battle_result

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
        parsed_skill = self._ensure_skill(skill)
        insights: list[Insight] = []

        # Static analysis insights
        if len(parsed_skill.description) < 50:
            insights.append(Insight(
                type="description",
                message="Skill description is very short",
                severity="warning",
                suggestion="Add more detail to help agents understand when to use this skill",
            ))

        if not parsed_skill.when_to_use:
            insights.append(Insight(
                type="examples",
                message="No usage examples provided",
                severity="info",
                suggestion="Add 'when to use' examples to improve selection accuracy",
            ))

        if not parsed_skill.parameters:
            insights.append(Insight(
                type="parameters",
                message="No parameters defined",
                severity="info",
                suggestion="Define parameters to help agents invoke the skill correctly",
            ))

        # Results-based insights
        if results:
            if isinstance(results, EvaluationResult):
                if results.selection_rate < 0.3:
                    insights.append(Insight(
                        type="selection",
                        message=f"Low selection rate ({results.selection_rate:.1%})",
                        severity="warning",
                        suggestion="Consider improving the skill description clarity",
                    ))
                if results.false_positive_rate > 0.2:
                    insights.append(Insight(
                        type="accuracy",
                        message=f"High false positive rate ({results.false_positive_rate:.1%})",
                        severity="warning",
                        suggestion="Add more specific triggers to reduce false selections",
                    ))
            elif isinstance(results, ComparisonResult):
                skill_rate = results.selection_rates.get(parsed_skill.name, 0)
                if skill_rate < max(results.selection_rates.values()) * 0.5:
                    insights.append(Insight(
                        type="competition",
                        message="Skill is underperforming compared to competitors",
                        severity="warning",
                        suggestion="Review competing skills to identify improvements",
                    ))

        return insights

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
        parsed = Parser.parse(skill)
        self._parsed_skills[skill] = parsed
        return parsed
