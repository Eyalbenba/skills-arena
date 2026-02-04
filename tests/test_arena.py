"""Tests for the Arena class.

Tests the main entry point that orchestrates skill parsing, scenario generation,
agent running, and result scoring.
"""

import os
import tempfile

import pytest

from skills_arena import Arena, Config, NoSkillsError, Skill, Task
from skills_arena.models import (
    BattleResult,
    ComparisonResult,
    EvaluationResult,
    Grade,
    Parameter,
    SkillFormat,
)


# Test fixtures
@pytest.fixture
def sample_skill():
    """Create a sample skill for testing."""
    return Skill(
        name="web-search",
        description="Search the web for information using various search engines.",
        parameters=[
            Parameter(name="query", description="Search query", type="string", required=True),
            Parameter(name="max_results", description="Maximum results", type="integer"),
        ],
        when_to_use=[
            "User asks to search for something",
            "User wants information from the web",
        ],
        source_format=SkillFormat.GENERIC,
    )


@pytest.fixture
def sample_skill_b():
    """Create a second sample skill for comparison tests."""
    return Skill(
        name="extract-content",
        description="Extract content from a web page URL.",
        parameters=[
            Parameter(name="url", description="URL to extract from", type="string", required=True),
        ],
        when_to_use=[
            "User provides a URL to analyze",
            "User wants content from a specific page",
        ],
        source_format=SkillFormat.GENERIC,
    )


@pytest.fixture
def mock_config():
    """Create a config that uses mock agent."""
    return Config(
        scenarios=10,
        agents=["mock"],
        timeout_seconds=5,
    )


@pytest.fixture
def arena_with_mock(mock_config):
    """Create an Arena with mock agent configured."""
    return Arena(config=mock_config)


class TestArenaInit:
    """Tests for Arena initialization."""

    def test_create_default(self):
        """Arena initializes with default config."""
        arena = Arena()
        assert arena.config is not None
        assert arena.config.scenarios == 50
        assert "claude-code" in arena.config.agents

    def test_create_with_config(self, mock_config):
        """Arena accepts custom config."""
        arena = Arena(config=mock_config)
        assert arena.config.scenarios == 10
        assert "mock" in arena.config.agents

    def test_from_config_yaml(self):
        """Test creating Arena from YAML config."""
        yaml_content = """
task: "web search"
skills:
  - ./skill-a.md

evaluation:
  scenarios: 75
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                arena = Arena.from_config(f.name)
                assert arena.config.scenarios == 75
            finally:
                os.unlink(f.name)


class TestEnsureSkill:
    """Tests for the _ensure_skill method."""

    def test_skill_object_passthrough(self, arena_with_mock, sample_skill):
        """Skill objects are returned as-is."""
        result = arena_with_mock._ensure_skill(sample_skill)
        assert result == sample_skill

    def test_skill_caching(self, arena_with_mock, sample_skill):
        """Parsed skills are cached."""
        # First call caches the skill
        arena_with_mock._parsed_skills["./test.md"] = sample_skill

        # Second call returns cached version
        result = arena_with_mock._ensure_skill("./test.md")
        assert result == sample_skill


class TestEvaluate:
    """Tests for the evaluate() method."""

    @pytest.mark.asyncio
    async def test_evaluate_async_with_skill_object(self, arena_with_mock, sample_skill):
        """evaluate_async works with a Skill object."""
        result = await arena_with_mock.evaluate_async(
            skill=sample_skill,
            task="web search",
        )

        assert isinstance(result, EvaluationResult)
        assert result.skill == "web-search"
        assert result.scenarios_run > 0
        assert 0 <= result.score <= 100
        assert isinstance(result.grade, Grade)

    @pytest.mark.asyncio
    async def test_evaluate_async_with_task_object(self, arena_with_mock, sample_skill):
        """evaluate_async works with a Task object."""
        task = Task(
            description="web search",
            domains=["developer", "enterprise"],
        )

        result = await arena_with_mock.evaluate_async(
            skill=sample_skill,
            task=task,
        )

        assert isinstance(result, EvaluationResult)
        assert result.scenarios_run > 0

    @pytest.mark.asyncio
    async def test_evaluate_async_progress_callback(self, arena_with_mock, sample_skill):
        """evaluate_async calls progress callback."""
        progress_stages = []

        def on_progress(progress):
            progress_stages.append(progress.stage)

        result = await arena_with_mock.evaluate_async(
            skill=sample_skill,
            task="web search",
            on_progress=on_progress,
        )

        assert "parsing" in progress_stages
        assert "generation" in progress_stages
        assert "complete" in progress_stages

    def test_evaluate_sync(self, arena_with_mock, sample_skill):
        """evaluate() synchronous method works."""
        result = arena_with_mock.evaluate(
            skill=sample_skill,
            task="web search",
        )

        assert isinstance(result, EvaluationResult)
        assert result.scenarios_run > 0


class TestCompare:
    """Tests for the compare() method."""

    def test_compare_requires_two_skills(self, arena_with_mock, sample_skill):
        """compare() raises error with fewer than 2 skills."""
        with pytest.raises(NoSkillsError, match="at least 2 skills"):
            arena_with_mock.compare(
                skills=[sample_skill],
                task="web search",
            )

    @pytest.mark.asyncio
    async def test_compare_async_two_skills(self, arena_with_mock, sample_skill, sample_skill_b):
        """compare_async works with two skills."""
        result = await arena_with_mock.compare_async(
            skills=[sample_skill, sample_skill_b],
            task="web search and extraction",
        )

        assert isinstance(result, ComparisonResult)
        assert result.winner in ["web-search", "extract-content"]
        assert "web-search" in result.selection_rates
        assert "extract-content" in result.selection_rates
        assert result.scenarios_run > 0

    @pytest.mark.asyncio
    async def test_compare_async_progress_callback(self, arena_with_mock, sample_skill, sample_skill_b):
        """compare_async calls progress callback."""
        progress_stages = []

        def on_progress(progress):
            progress_stages.append(progress.stage)

        result = await arena_with_mock.compare_async(
            skills=[sample_skill, sample_skill_b],
            task="web search",
            on_progress=on_progress,
        )

        assert "parsing" in progress_stages
        assert "generation" in progress_stages
        assert "complete" in progress_stages

    def test_compare_sync(self, arena_with_mock, sample_skill, sample_skill_b):
        """compare() synchronous method works."""
        result = arena_with_mock.compare(
            skills=[sample_skill, sample_skill_b],
            task="web search",
        )

        assert isinstance(result, ComparisonResult)
        assert result.scenarios_run > 0


class TestBattleRoyale:
    """Tests for the battle_royale() method."""

    def test_battle_requires_two_skills(self, arena_with_mock, sample_skill):
        """battle_royale() raises error with fewer than 2 skills."""
        with pytest.raises(NoSkillsError, match="at least 2 skills"):
            arena_with_mock.battle_royale(
                skills=[sample_skill],
                task="web search",
            )

    @pytest.mark.asyncio
    async def test_battle_royale_async(self, arena_with_mock, sample_skill, sample_skill_b):
        """battle_royale_async works with multiple skills."""
        result = await arena_with_mock.battle_royale_async(
            skills=[sample_skill, sample_skill_b],
            task="web search and extraction",
        )

        assert isinstance(result, BattleResult)
        assert len(result.leaderboard) == 2
        assert result.leaderboard[0].rank == 1
        assert result.leaderboard[1].rank == 2

    def test_battle_royale_sync(self, arena_with_mock, sample_skill, sample_skill_b):
        """battle_royale() synchronous method works."""
        result = arena_with_mock.battle_royale(
            skills=[sample_skill, sample_skill_b],
            task="web search",
        )

        assert isinstance(result, BattleResult)
        assert result.scenarios_run > 0


class TestInsights:
    """Tests for the insights() method."""

    def test_insights_short_description(self, arena_with_mock):
        """insights() identifies short descriptions."""
        skill = Skill(
            name="test",
            description="Short desc",
            source_format=SkillFormat.GENERIC,
        )

        insights = arena_with_mock.insights(skill)

        types = [i.type for i in insights]
        assert "description" in types

    def test_insights_no_examples(self, arena_with_mock):
        """insights() identifies missing examples."""
        skill = Skill(
            name="test",
            description="A skill that does something useful for users.",
            when_to_use=[],
            source_format=SkillFormat.GENERIC,
        )

        insights = arena_with_mock.insights(skill)

        types = [i.type for i in insights]
        assert "examples" in types

    def test_insights_with_evaluation_result(self, arena_with_mock, sample_skill):
        """insights() analyzes evaluation results."""
        eval_result = EvaluationResult(
            skill="web-search",
            score=50.0,
            grade=Grade.D,
            selection_rate=0.2,  # Low rate
            false_positive_rate=0.3,  # High rate
            invocation_accuracy=0.7,
            per_agent={},
            insights=[],
            scenarios_run=100,
        )

        insights = arena_with_mock.insights(sample_skill, results=eval_result)

        types = [i.type for i in insights]
        assert "selection" in types  # Low selection rate
        assert "accuracy" in types  # High false positive rate


class TestRunScenario:
    """Tests for the _run_scenario method."""

    @pytest.mark.asyncio
    async def test_run_scenario_success(self, arena_with_mock, sample_skill):
        """_run_scenario returns a SelectionResult."""
        from skills_arena.runner import MockAgent
        from skills_arena.models import SelectionResult

        agent = MockAgent(mode="keyword")

        result = await arena_with_mock._run_scenario(
            agent=agent,
            prompt="Search for Python tutorials",
            skills=[sample_skill],
            expected_skill="web-search",
            scenario_id="test-001",
        )

        assert isinstance(result, SelectionResult)
        assert result.scenario.id == "test-001"
        assert result.agent_name == "mock"
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_run_scenario_timeout(self, arena_with_mock, sample_skill):
        """_run_scenario handles timeout gracefully."""
        from skills_arena.models import SelectionResult

        class SlowAgent:
            name = "slow"

            async def select_skill(self, prompt, skills):
                import asyncio
                await asyncio.sleep(100)  # Very slow

            async def close(self):
                pass

        # Use very short timeout
        arena_with_mock.config.timeout_seconds = 0.01

        result = await arena_with_mock._run_scenario(
            agent=SlowAgent(),
            prompt="Test prompt",
            skills=[sample_skill],
            expected_skill="web-search",
            scenario_id="test-002",
        )

        assert isinstance(result, SelectionResult)
        assert result.selection.skill is None
        assert "timed out" in result.selection.reasoning


class TestGetGenerator:
    """Tests for the _get_generator method."""

    def test_mock_generator_for_mock_agent(self, arena_with_mock):
        """Uses MockGenerator when mock agent is configured."""
        from skills_arena.generator import MockGenerator

        generator = arena_with_mock._get_generator()
        assert isinstance(generator, MockGenerator)

    def test_llm_generator_for_other_agents(self):
        """Uses LLMGenerator for non-mock agents."""
        from skills_arena.generator import LLMGenerator

        config = Config(agents=["claude-code"])
        arena = Arena(config=config)
        generator = arena._get_generator()
        assert isinstance(generator, LLMGenerator)


class TestCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_agents(self, arena_with_mock, sample_skill):
        """Agents are cleaned up after evaluation."""
        # Run an evaluation
        await arena_with_mock.evaluate_async(sample_skill, "test")

        # Agents should be cleaned up
        assert len(arena_with_mock._agents) == 0


class TestRunNotImplemented:
    """Tests for run() which is not yet implemented."""

    def test_run_not_implemented(self):
        """Test that run raises NotImplementedError."""
        arena = Arena()

        with pytest.raises(NotImplementedError, match="Phase 1"):
            arena.run()
