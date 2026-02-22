"""Tests for the optimizer module.

Tests the SkillOptimizer, optimization models, Arena.optimize() integration,
reporter formatting, and edge cases like regression guard.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skills_arena import (
    Arena,
    ComparisonResult,
    Config,
    Grade,
    NoSkillsError,
    OptimizerError,
    Skill,
    SkillFormat,
    TextReporter,
    print_results,
)
from skills_arena.models import (
    OptimizationIteration,
    OptimizationResult,
    Parameter,
    ScenarioDetail,
)
from skills_arena.optimizer import SkillOptimizer


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def skill_a():
    """Skill to optimize."""
    return Skill(
        name="Tavily Search",
        description="Search the web for information.",
        parameters=[
            Parameter(name="query", description="Search query", type="string", required=True),
        ],
        when_to_use=["User asks to search for something"],
        source_format=SkillFormat.GENERIC,
    )


@pytest.fixture
def skill_b():
    """Competitor skill."""
    return Skill(
        name="Firecrawl CLI",
        description="Scrape and extract content from web pages.",
        parameters=[
            Parameter(name="url", description="URL to scrape", type="string", required=True),
        ],
        when_to_use=["User wants to extract page content"],
        source_format=SkillFormat.GENERIC,
    )


@pytest.fixture
def comparison_result():
    """A ComparisonResult with steal data for testing."""
    return ComparisonResult(
        winner="Firecrawl CLI",
        selection_rates={"Tavily Search": 0.4, "Firecrawl CLI": 0.6},
        head_to_head={},
        scenarios_run=10,
        scenario_details=[
            ScenarioDetail(
                scenario_id="s1",
                prompt="Find the latest AI research papers",
                expected_skill="Tavily Search",
                selected_skill="Tavily Search",
                reasoning="Clear search request",
                was_stolen=False,
            ),
            ScenarioDetail(
                scenario_id="s2",
                prompt="Get pricing info from stripe.com",
                expected_skill="Tavily Search",
                selected_skill="Firecrawl CLI",
                reasoning="URL-specific extraction suits Firecrawl",
                was_stolen=True,
            ),
            ScenarioDetail(
                scenario_id="s3",
                prompt="What are trending topics on HN?",
                expected_skill="Tavily Search",
                selected_skill="Tavily Search",
                reasoning="General search request",
                was_stolen=False,
            ),
            ScenarioDetail(
                scenario_id="s4",
                prompt="Scrape product details from amazon.com/dp/123",
                expected_skill="Firecrawl CLI",
                selected_skill="Firecrawl CLI",
                reasoning="Direct URL scraping",
                was_stolen=False,
            ),
            ScenarioDetail(
                scenario_id="s5",
                prompt="Extract the main article from this blog post",
                expected_skill="Firecrawl CLI",
                selected_skill="Firecrawl CLI",
                reasoning="Content extraction task",
                was_stolen=False,
            ),
        ],
        steals={"Tavily Search": ["s2"]},
    )


@pytest.fixture
def mock_config():
    """Config with mock agent."""
    return Config(scenarios=5, agents=["mock"], timeout_seconds=5)


@pytest.fixture
def arena_mock(mock_config):
    """Arena with mock agent."""
    return Arena(config=mock_config)


@pytest.fixture
def optimization_result(skill_a):
    """A sample OptimizationResult for reporter tests."""
    improved = Skill(
        name="Tavily Search",
        description="Improved description with examples.",
        parameters=skill_a.parameters,
        when_to_use=["Search the web", "Find current information", "Look up facts"],
        source_format=SkillFormat.GENERIC,
    )
    return OptimizationResult(
        original_skill=skill_a,
        optimized_skill=improved,
        iterations=[
            OptimizationIteration(
                iteration=1,
                skill_before=skill_a,
                skill_after=improved,
                comparison_before=ComparisonResult(
                    winner="Firecrawl CLI",
                    selection_rates={"Tavily Search": 0.34, "Firecrawl CLI": 0.66},
                    scenarios_run=50,
                ),
                comparison_after=ComparisonResult(
                    winner="Tavily Search",
                    selection_rates={"Tavily Search": 0.62, "Firecrawl CLI": 0.38},
                    scenarios_run=50,
                ),
                selection_rate_before=0.34,
                selection_rate_after=0.62,
                improvement=0.28,
                reasoning="Added usage examples and 'Use this when...' clauses",
            ),
        ],
        total_improvement=0.28,
        selection_rate_before=0.34,
        selection_rate_after=0.62,
        grade_before=Grade.F,
        grade_after=Grade.D,
        scenarios_used=50,
        competitors=["Firecrawl CLI"],
    )


# ─── Model Tests ─────────────────────────────────────────────────────────


class TestOptimizationIteration:
    """Tests for OptimizationIteration model."""

    def test_create_iteration(self, skill_a, comparison_result):
        """Test creating an optimization iteration."""
        improved = skill_a.model_copy(update={"description": "Better desc"})
        iteration = OptimizationIteration(
            iteration=1,
            skill_before=skill_a,
            skill_after=improved,
            comparison_before=comparison_result,
            comparison_after=comparison_result,
            selection_rate_before=0.4,
            selection_rate_after=0.6,
            improvement=0.2,
            reasoning="Added examples",
        )
        assert iteration.iteration == 1
        assert iteration.improvement == 0.2
        assert iteration.reasoning == "Added examples"

    def test_negative_improvement(self, skill_a, comparison_result):
        """Test iteration with regression (negative improvement)."""
        iteration = OptimizationIteration(
            iteration=2,
            skill_before=skill_a,
            skill_after=skill_a,
            comparison_before=comparison_result,
            comparison_after=comparison_result,
            selection_rate_before=0.6,
            selection_rate_after=0.45,
            improvement=-0.15,
        )
        assert iteration.improvement < 0


class TestOptimizationResult:
    """Tests for OptimizationResult model."""

    def test_create_result(self, skill_a):
        """Test creating an optimization result with defaults."""
        result = OptimizationResult(
            original_skill=skill_a,
            optimized_skill=skill_a,
        )
        assert result.total_improvement == 0.0
        assert result.grade_before == Grade.F
        assert result.iterations == []
        assert result.competitors == []

    def test_create_full_result(self, optimization_result):
        """Test a fully populated optimization result."""
        assert optimization_result.total_improvement == 0.28
        assert optimization_result.selection_rate_before == 0.34
        assert optimization_result.selection_rate_after == 0.62
        assert len(optimization_result.iterations) == 1
        assert optimization_result.competitors == ["Firecrawl CLI"]
        assert optimization_result.scenarios_used == 50


# ─── Exception Tests ─────────────────────────────────────────────────────


class TestOptimizerError:
    """Tests for OptimizerError exception."""

    def test_basic_message(self):
        """Test basic error message."""
        error = OptimizerError("LLM returned empty response")
        assert "Optimization failed" in str(error)
        assert "LLM returned empty response" in str(error)

    def test_with_skill_name(self):
        """Test error with skill name context."""
        error = OptimizerError("JSON parse failed", skill_name="Tavily Search")
        assert "Tavily Search" in str(error)
        assert error.skill_name == "Tavily Search"

    def test_inherits_from_base(self):
        """Test that OptimizerError is a SkillsArenaError."""
        from skills_arena import SkillsArenaError

        error = OptimizerError("test")
        assert isinstance(error, SkillsArenaError)
        assert isinstance(error, Exception)


# ─── SkillOptimizer Tests ────────────────────────────────────────────────


def _make_mock_response(data: dict) -> MagicMock:
    """Build a mock Anthropic API response."""
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(data)
    response = MagicMock()
    response.content = [block]
    return response


class TestSkillOptimizerInit:
    """Tests for SkillOptimizer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        optimizer = SkillOptimizer()
        assert optimizer.model == "claude-sonnet-4-20250514"
        assert optimizer.temperature == 0.3
        assert optimizer._client is None

    def test_custom_init(self):
        """Test custom initialization."""
        optimizer = SkillOptimizer(
            model="claude-haiku-4-5-20251001",
            temperature=0.5,
            api_key="sk-test",
        )
        assert optimizer.model == "claude-haiku-4-5-20251001"
        assert optimizer.temperature == 0.5
        assert optimizer._api_key == "sk-test"

    def test_lazy_client(self):
        """Test that client is created lazily."""
        optimizer = SkillOptimizer(api_key="sk-test")
        assert optimizer._client is None
        client = optimizer._get_client()
        assert client is not None
        # Second call returns same instance
        assert optimizer._get_client() is client


class TestSkillOptimizerFormatting:
    """Tests for the optimizer's internal formatting methods."""

    def test_format_stolen_scenarios(self, comparison_result):
        """Test formatting of stolen scenarios."""
        optimizer = SkillOptimizer()
        output = optimizer._format_stolen_scenarios(comparison_result, "Tavily Search")
        assert "Get pricing info from stripe.com" in output
        assert "Firecrawl CLI" in output
        assert "URL-specific extraction" in output

    def test_format_stolen_scenarios_none_stolen(self, comparison_result):
        """Test formatting when no scenarios were stolen."""
        optimizer = SkillOptimizer()
        output = optimizer._format_stolen_scenarios(comparison_result, "Firecrawl CLI")
        assert "none" in output.lower()

    def test_format_won_scenarios(self, comparison_result):
        """Test formatting of won scenarios."""
        optimizer = SkillOptimizer()
        output = optimizer._format_won_scenarios(comparison_result, "Tavily Search")
        assert "latest AI research" in output
        assert "trending topics" in output

    def test_format_won_scenarios_limits_output(self, skill_a, skill_b):
        """Test that won scenarios are limited to 5."""
        details = [
            ScenarioDetail(
                scenario_id=f"s{i}",
                prompt=f"Prompt {i}",
                expected_skill="Tavily Search",
                selected_skill="Tavily Search",
                was_stolen=False,
            )
            for i in range(10)
        ]
        result = ComparisonResult(
            winner="Tavily Search",
            selection_rates={"Tavily Search": 1.0},
            scenarios_run=10,
            scenario_details=details,
        )
        optimizer = SkillOptimizer()
        output = optimizer._format_won_scenarios(result, "Tavily Search")
        assert "5 more" in output


class TestOptimizeDescription:
    """Tests for the optimize_description method."""

    @pytest.mark.asyncio
    async def test_successful_optimization(self, skill_a, skill_b, comparison_result):
        """Test successful skill optimization."""
        optimizer = SkillOptimizer(api_key="sk-test")

        response_data = {
            "description": "Enhanced search with examples and boundaries.",
            "when_to_use": [
                "Search for current information",
                "Find news and articles",
                "Look up facts and data",
            ],
            "parameters": [
                {"name": "query", "description": "Search query string", "type": "string", "required": True},
            ],
            "reasoning": "Added usage examples and scope boundaries",
        }

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _make_mock_response(response_data)
        optimizer._client = mock_client

        optimized, reasoning = await optimizer.optimize_description(
            skill=skill_a,
            comparison_result=comparison_result,
            competitors=[skill_b],
        )

        assert optimized.name == "Tavily Search"
        assert optimized.description == "Enhanced search with examples and boundaries."
        assert len(optimized.when_to_use) == 3
        assert optimized.parameters[0].name == "query"
        assert "examples" in reasoning.lower()

    @pytest.mark.asyncio
    async def test_preserves_metadata(self, skill_a, skill_b, comparison_result):
        """Test that source_format and source_path are preserved."""
        skill = skill_a.model_copy(update={
            "source_format": SkillFormat.CLAUDE_CODE,
            "source_path": "/path/to/skill.md",
            "token_count": 500,
        })
        optimizer = SkillOptimizer(api_key="sk-test")

        response_data = {
            "description": "Better description",
            "when_to_use": ["Use case A"],
            "reasoning": "Improved clarity",
        }

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _make_mock_response(response_data)
        optimizer._client = mock_client

        optimized, _ = await optimizer.optimize_description(
            skill=skill,
            comparison_result=comparison_result,
            competitors=[skill_b],
        )

        assert optimized.source_format == SkillFormat.CLAUDE_CODE
        assert optimized.source_path == "/path/to/skill.md"
        assert optimized.token_count == 500

    @pytest.mark.asyncio
    async def test_handles_markdown_wrapped_json(self, skill_a, skill_b, comparison_result):
        """Test parsing JSON wrapped in markdown code blocks."""
        optimizer = SkillOptimizer(api_key="sk-test")

        json_data = {
            "description": "Improved",
            "when_to_use": ["A"],
            "reasoning": "Changes",
        }
        # Wrap in markdown code blocks
        block = MagicMock()
        block.type = "text"
        block.text = f"```json\n{json.dumps(json_data)}\n```"
        response = MagicMock()
        response.content = [block]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = response
        optimizer._client = mock_client

        optimized, reasoning = await optimizer.optimize_description(
            skill=skill_a,
            comparison_result=comparison_result,
            competitors=[skill_b],
        )

        assert optimized.description == "Improved"

    @pytest.mark.asyncio
    async def test_empty_response_raises(self, skill_a, skill_b, comparison_result):
        """Test that empty LLM response raises OptimizerError."""
        optimizer = SkillOptimizer(api_key="sk-test")

        block = MagicMock()
        block.type = "text"
        block.text = ""
        response = MagicMock()
        response.content = [block]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = response
        optimizer._client = mock_client

        with pytest.raises(OptimizerError, match="Empty response"):
            await optimizer.optimize_description(
                skill=skill_a,
                comparison_result=comparison_result,
                competitors=[skill_b],
            )

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self, skill_a, skill_b, comparison_result):
        """Test that invalid JSON response raises OptimizerError."""
        optimizer = SkillOptimizer(api_key="sk-test")

        block = MagicMock()
        block.type = "text"
        block.text = "This is not valid JSON at all"
        response = MagicMock()
        response.content = [block]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = response
        optimizer._client = mock_client

        with pytest.raises(OptimizerError, match="JSON"):
            await optimizer.optimize_description(
                skill=skill_a,
                comparison_result=comparison_result,
                competitors=[skill_b],
            )

    @pytest.mark.asyncio
    async def test_non_object_json_raises(self, skill_a, skill_b, comparison_result):
        """Test that JSON array (not object) raises OptimizerError."""
        optimizer = SkillOptimizer(api_key="sk-test")

        block = MagicMock()
        block.type = "text"
        block.text = json.dumps([1, 2, 3])
        response = MagicMock()
        response.content = [block]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = response
        optimizer._client = mock_client

        with pytest.raises(OptimizerError, match="Expected JSON object"):
            await optimizer.optimize_description(
                skill=skill_a,
                comparison_result=comparison_result,
                competitors=[skill_b],
            )

    @pytest.mark.asyncio
    async def test_api_error_raises(self, skill_a, skill_b, comparison_result):
        """Test that Anthropic API errors are wrapped in OptimizerError."""
        import anthropic

        optimizer = SkillOptimizer(api_key="sk-test")

        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = anthropic.APIError(
            message="Rate limit", request=MagicMock(), body=None,
        )
        optimizer._client = mock_client

        with pytest.raises(OptimizerError, match="API error"):
            await optimizer.optimize_description(
                skill=skill_a,
                comparison_result=comparison_result,
                competitors=[skill_b],
            )

    @pytest.mark.asyncio
    async def test_keeps_existing_params_when_not_returned(
        self, skill_a, skill_b, comparison_result
    ):
        """Test that parameters are preserved when LLM omits them."""
        optimizer = SkillOptimizer(api_key="sk-test")

        response_data = {
            "description": "Better",
            "when_to_use": ["A"],
            "reasoning": "No param changes",
            # No "parameters" key
        }

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _make_mock_response(response_data)
        optimizer._client = mock_client

        optimized, _ = await optimizer.optimize_description(
            skill=skill_a,
            comparison_result=comparison_result,
            competitors=[skill_b],
        )

        # Should keep original params
        assert len(optimized.parameters) == 1
        assert optimized.parameters[0].name == "query"


# ─── Arena.optimize() Integration Tests ──────────────────────────────────


class TestArenaOptimize:
    """Tests for Arena.optimize() and optimize_async()."""

    def test_optimize_requires_competitor(self, arena_mock, skill_a):
        """optimize() raises error with no competitors."""
        with pytest.raises(NoSkillsError, match="at least 1 competitor"):
            arena_mock.optimize(
                skill=skill_a,
                competitors=[],
                task="web search",
            )

    @pytest.mark.asyncio
    async def test_optimize_async_full_flow(self, arena_mock, skill_a, skill_b):
        """Test the full optimize_async flow with mock agent."""
        # Mock the optimizer to return an improved skill
        improved_skill = skill_a.model_copy(update={
            "description": "Much better search description with examples.",
            "when_to_use": ["Search web", "Find info", "Current events"],
        })

        with patch.object(
            SkillOptimizer,
            "optimize_description",
            new_callable=AsyncMock,
            return_value=(improved_skill, "Added examples"),
        ):
            result = await arena_mock.optimize_async(
                skill=skill_a,
                competitors=[skill_b],
                task="web search and extraction",
            )

        assert isinstance(result, OptimizationResult)
        assert result.original_skill.name == "Tavily Search"
        assert result.optimized_skill.name == "Tavily Search"
        assert len(result.iterations) == 1
        assert result.scenarios_used > 0
        assert result.competitors == ["Firecrawl CLI"]

    @pytest.mark.asyncio
    async def test_optimize_async_with_baseline(
        self, arena_mock, skill_a, skill_b, comparison_result
    ):
        """Test optimize_async reuses a provided baseline."""
        improved_skill = skill_a.model_copy(update={
            "description": "Improved with baseline reuse.",
        })

        with patch.object(
            SkillOptimizer,
            "optimize_description",
            new_callable=AsyncMock,
            return_value=(improved_skill, "Reused baseline"),
        ):
            result = await arena_mock.optimize_async(
                skill=skill_a,
                competitors=[skill_b],
                task="web search",
                baseline=comparison_result,
            )

        assert isinstance(result, OptimizationResult)
        assert result.selection_rate_before == 0.4  # From baseline fixture

    @pytest.mark.asyncio
    async def test_optimize_async_frozen_scenarios(
        self, arena_mock, skill_a, skill_b, comparison_result
    ):
        """Test that verification uses frozen scenarios from baseline."""
        improved_skill = skill_a.model_copy(update={
            "description": "Better description",
        })

        compare_calls = []
        original_compare = arena_mock.compare_async

        async def tracking_compare(*args, **kwargs):
            compare_calls.append(kwargs.get("scenarios"))
            return await original_compare(*args, **kwargs)

        with patch.object(
            SkillOptimizer,
            "optimize_description",
            new_callable=AsyncMock,
            return_value=(improved_skill, "Changes"),
        ), patch.object(arena_mock, "compare_async", side_effect=tracking_compare):
            await arena_mock.optimize_async(
                skill=skill_a,
                competitors=[skill_b],
                task="web search",
                baseline=comparison_result,
            )

        # The verification compare call should use frozen scenarios (CustomScenario list)
        # compare_calls[0] is the verification run's scenarios kwarg
        assert len(compare_calls) == 1
        frozen = compare_calls[0]
        assert frozen is not None
        assert len(frozen) == len(comparison_result.scenario_details)
        # Verify prompts match
        baseline_prompts = {d.prompt for d in comparison_result.scenario_details}
        frozen_prompts = {s.prompt for s in frozen}
        assert baseline_prompts == frozen_prompts

    @pytest.mark.asyncio
    async def test_optimize_async_progress_callback(self, arena_mock, skill_a, skill_b):
        """Test that progress callback is called during optimization."""
        improved_skill = skill_a.model_copy(update={"description": "Better"})
        stages = []

        def on_progress(progress):
            stages.append(progress.stage)

        with patch.object(
            SkillOptimizer,
            "optimize_description",
            new_callable=AsyncMock,
            return_value=(improved_skill, "reason"),
        ):
            await arena_mock.optimize_async(
                skill=skill_a,
                competitors=[skill_b],
                task="web search",
                on_progress=on_progress,
            )

        assert "parsing" in stages
        assert "baseline" in stages
        assert "optimizing" in stages
        assert "complete" in stages

    @pytest.mark.asyncio
    async def test_optimize_async_multi_iteration(self, arena_mock, skill_a, skill_b):
        """Test multiple optimization iterations."""
        v1 = skill_a.model_copy(update={"description": "Version 1"})
        v2 = skill_a.model_copy(update={"description": "Version 2"})

        call_count = 0

        async def mock_optimize(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (v1, "Iteration 1 changes")
            return (v2, "Iteration 2 changes")

        with patch.object(
            SkillOptimizer,
            "optimize_description",
            side_effect=mock_optimize,
        ):
            result = await arena_mock.optimize_async(
                skill=skill_a,
                competitors=[skill_b],
                task="web search",
                max_iterations=2,
            )

        assert len(result.iterations) == 2
        assert result.iterations[0].iteration == 1
        assert result.iterations[1].iteration == 2

    @pytest.mark.asyncio
    async def test_optimize_async_regression_guard(
        self, arena_mock, skill_a, skill_b
    ):
        """Test that optimization stops when selection rate drops."""
        worse_skill = skill_a.model_copy(update={"description": "Worse description"})

        # First iteration: optimizer returns a skill
        # The mock agent will evaluate it — we patch compare_async to simulate regression
        baseline_result = ComparisonResult(
            winner="Tavily Search",
            selection_rates={"Tavily Search": 0.7, "Firecrawl CLI": 0.3},
            scenarios_run=5,
            scenario_details=[
                ScenarioDetail(
                    scenario_id="s1",
                    prompt="Search for news",
                    expected_skill="Tavily Search",
                    selected_skill="Tavily Search",
                    was_stolen=False,
                ),
            ],
        )
        worse_result = ComparisonResult(
            winner="Firecrawl CLI",
            selection_rates={"Tavily Search": 0.3, "Firecrawl CLI": 0.7},
            scenarios_run=5,
            scenario_details=[],
        )

        compare_count = 0

        async def mock_compare(*args, **kwargs):
            nonlocal compare_count
            compare_count += 1
            # Verification runs return worse result
            return worse_result

        with patch.object(
            SkillOptimizer,
            "optimize_description",
            new_callable=AsyncMock,
            return_value=(worse_skill, "Made it worse"),
        ), patch.object(
            arena_mock, "compare_async", side_effect=mock_compare,
        ):
            result = await arena_mock.optimize_async(
                skill=skill_a,
                competitors=[skill_b],
                task="web search",
                max_iterations=3,
                baseline=baseline_result,
            )

        # Should stop after first iteration due to regression
        assert len(result.iterations) == 1
        assert result.iterations[0].improvement < 0
        # Best skill should still be the original (higher rate)
        assert result.optimized_skill.name == "Tavily Search"

    def test_optimize_sync_wrapper(self, arena_mock, skill_a, skill_b):
        """Test that sync optimize() calls optimize_async()."""
        improved = skill_a.model_copy(update={"description": "Better"})

        with patch.object(
            SkillOptimizer,
            "optimize_description",
            new_callable=AsyncMock,
            return_value=(improved, "reason"),
        ):
            result = arena_mock.optimize(
                skill=skill_a,
                competitors=[skill_b],
                task="web search",
            )

        assert isinstance(result, OptimizationResult)


# ─── Reporter Tests ──────────────────────────────────────────────────────


class TestOptimizationReporter:
    """Tests for optimization result formatting."""

    def test_format_optimization(self, optimization_result):
        """Test formatting an optimization result."""
        reporter = TextReporter()
        output = reporter.format_optimization(optimization_result)

        assert "OPTIMIZATION RESULTS" in output
        assert "Tavily Search" in output
        assert "Firecrawl CLI" in output
        assert "34%" in output
        assert "62%" in output
        assert "+28%" in output
        assert "Iteration 1" in output
        assert "Added usage examples" in output

    def test_format_optimization_no_iterations(self, skill_a):
        """Test formatting with no iterations (edge case)."""
        result = OptimizationResult(
            original_skill=skill_a,
            optimized_skill=skill_a,
            selection_rate_before=0.5,
            selection_rate_after=0.5,
            grade_before=Grade.F,
            grade_after=Grade.F,
            scenarios_used=10,
            competitors=["Other"],
        )
        reporter = TextReporter()
        output = reporter.format_optimization(result)

        assert "OPTIMIZATION RESULTS" in output
        assert "Iterations: 0" in output

    def test_format_optimization_multiple_iterations(self, skill_a):
        """Test formatting with multiple iterations."""
        v1 = skill_a.model_copy(update={"description": "V1"})
        v2 = skill_a.model_copy(update={"description": "V2"})
        empty_comparison = ComparisonResult(
            winner="A", selection_rates={}, scenarios_run=5,
        )
        result = OptimizationResult(
            original_skill=skill_a,
            optimized_skill=v2,
            iterations=[
                OptimizationIteration(
                    iteration=1,
                    skill_before=skill_a,
                    skill_after=v1,
                    comparison_before=empty_comparison,
                    comparison_after=empty_comparison,
                    selection_rate_before=0.3,
                    selection_rate_after=0.5,
                    improvement=0.2,
                    reasoning="First round",
                ),
                OptimizationIteration(
                    iteration=2,
                    skill_before=v1,
                    skill_after=v2,
                    comparison_before=empty_comparison,
                    comparison_after=empty_comparison,
                    selection_rate_before=0.5,
                    selection_rate_after=0.6,
                    improvement=0.1,
                    reasoning="Second round",
                ),
            ],
            total_improvement=0.3,
            selection_rate_before=0.3,
            selection_rate_after=0.6,
            grade_before=Grade.F,
            grade_after=Grade.D,
            scenarios_used=20,
            competitors=["B"],
        )
        reporter = TextReporter()
        output = reporter.format_optimization(result)

        assert "Iteration 1" in output
        assert "Iteration 2" in output
        assert "First round" in output
        assert "Second round" in output
        assert "improved" in output

    def test_print_results_optimization(self, optimization_result, capsys):
        """Test print_results dispatches to format_optimization."""
        print_results(optimization_result)
        captured = capsys.readouterr()

        assert "OPTIMIZATION RESULTS" in captured.out
        assert "Tavily Search" in captured.out

    def test_format_optimization_grade_display(self, optimization_result):
        """Test that grades are shown correctly."""
        reporter = TextReporter()
        output = reporter.format_optimization(optimization_result)

        assert "F" in output
        assert "D" in output

    def test_format_optimization_bars(self, optimization_result):
        """Test that selection rate bars are rendered."""
        reporter = TextReporter()
        output = reporter.format_optimization(optimization_result)

        # Should contain bar characters
        assert "\u2588" in output or "\u2591" in output

    def test_format_optimization_regression(self, skill_a):
        """Test formatting shows 'regressed' for negative improvement."""
        empty = ComparisonResult(winner="A", selection_rates={}, scenarios_run=5)
        result = OptimizationResult(
            original_skill=skill_a,
            optimized_skill=skill_a,
            iterations=[
                OptimizationIteration(
                    iteration=1,
                    skill_before=skill_a,
                    skill_after=skill_a,
                    comparison_before=empty,
                    comparison_after=empty,
                    selection_rate_before=0.6,
                    selection_rate_after=0.4,
                    improvement=-0.2,
                    reasoning="Made it worse",
                ),
            ],
            total_improvement=-0.2,
            selection_rate_before=0.6,
            selection_rate_after=0.4,
            scenarios_used=10,
            competitors=["B"],
        )
        reporter = TextReporter()
        output = reporter.format_optimization(result)

        assert "regressed" in output
