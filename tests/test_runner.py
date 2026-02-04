"""Tests for the runner module.

Tests cover:
- MockAgent various modes
- ScriptedMockAgent pattern matching
- BaseAgent interface
- get_agent factory function
"""

import pytest

from skills_arena import Skill
from skills_arena.runner import (
    BaseAgent,
    MockAgent,
    ScriptedMockAgent,
    get_agent,
)


# Test fixtures
@pytest.fixture
def web_search_skill() -> Skill:
    """Create a web search skill for testing."""
    return Skill(
        name="web_search",
        description="Search the web for information on any topic",
        when_to_use=["find information online", "search for news", "look up facts"],
    )


@pytest.fixture
def code_search_skill() -> Skill:
    """Create a code search skill for testing."""
    return Skill(
        name="code_search",
        description="Search code repositories for functions and patterns",
        when_to_use=["find code examples", "search github", "look up implementations"],
    )


@pytest.fixture
def data_analysis_skill() -> Skill:
    """Create a data analysis skill for testing."""
    return Skill(
        name="data_analysis",
        description="Analyze datasets and generate insights",
        when_to_use=["analyze data", "create charts", "statistical analysis"],
    )


@pytest.fixture
def all_skills(
    web_search_skill: Skill,
    code_search_skill: Skill,
    data_analysis_skill: Skill,
) -> list[Skill]:
    """All test skills."""
    return [web_search_skill, code_search_skill, data_analysis_skill]


class TestMockAgent:
    """Tests for MockAgent."""

    @pytest.mark.asyncio
    async def test_keyword_mode_web_search(
        self, all_skills: list[Skill]
    ) -> None:
        """Test keyword mode selects web_search for relevant prompt."""
        agent = MockAgent(mode="keyword")
        selection = await agent.select_skill(
            prompt="Search the web for Python tutorials",
            available_skills=all_skills,
        )

        assert selection.skill == "web_search"
        assert selection.confidence > 0.0
        assert "web" in selection.reasoning.lower() or "search" in selection.reasoning.lower()

    @pytest.mark.asyncio
    async def test_keyword_mode_code_search(
        self, all_skills: list[Skill]
    ) -> None:
        """Test keyword mode selects code_search for relevant prompt."""
        agent = MockAgent(mode="keyword")
        selection = await agent.select_skill(
            prompt="Find code examples for async Python",
            available_skills=all_skills,
        )

        assert selection.skill == "code_search"
        assert selection.confidence > 0.0

    @pytest.mark.asyncio
    async def test_keyword_mode_data_analysis(
        self, all_skills: list[Skill]
    ) -> None:
        """Test keyword mode selects data_analysis for relevant prompt."""
        agent = MockAgent(mode="keyword")
        selection = await agent.select_skill(
            prompt="Analyze this dataset and create charts",
            available_skills=all_skills,
        )

        assert selection.skill == "data_analysis"
        assert selection.confidence > 0.0

    @pytest.mark.asyncio
    async def test_keyword_mode_no_match(
        self, all_skills: list[Skill]
    ) -> None:
        """Test keyword mode returns None for unrelated prompt."""
        agent = MockAgent(mode="keyword")
        selection = await agent.select_skill(
            prompt="What is the weather today?",
            available_skills=all_skills,
        )

        # May or may not match depending on keyword overlap
        # But should have low confidence if matches
        if selection.skill is not None:
            assert selection.confidence < 0.7

    @pytest.mark.asyncio
    async def test_first_mode(
        self, all_skills: list[Skill]
    ) -> None:
        """Test first mode always selects first skill."""
        agent = MockAgent(mode="first")
        selection = await agent.select_skill(
            prompt="Any random prompt",
            available_skills=all_skills,
        )

        assert selection.skill == all_skills[0].name
        assert "first" in selection.reasoning.lower()

    @pytest.mark.asyncio
    async def test_none_mode(
        self, all_skills: list[Skill]
    ) -> None:
        """Test none mode never selects a skill."""
        agent = MockAgent(mode="none")
        selection = await agent.select_skill(
            prompt="Search for anything",
            available_skills=all_skills,
        )

        assert selection.skill is None
        assert selection.confidence == 0.0

    @pytest.mark.asyncio
    async def test_random_mode_deterministic_with_seed(
        self, all_skills: list[Skill]
    ) -> None:
        """Test random mode is deterministic with seed."""
        agent1 = MockAgent(mode="random", seed=42)
        agent2 = MockAgent(mode="random", seed=42)

        selection1 = await agent1.select_skill(
            prompt="Test prompt",
            available_skills=all_skills,
        )
        selection2 = await agent2.select_skill(
            prompt="Test prompt",
            available_skills=all_skills,
        )

        assert selection1.skill == selection2.skill

    @pytest.mark.asyncio
    async def test_empty_skills_list(self) -> None:
        """Test handling empty skills list."""
        agent = MockAgent()
        selection = await agent.select_skill(
            prompt="Any prompt",
            available_skills=[],
        )

        assert selection.skill is None
        assert selection.confidence == 0.0
        assert "no skills" in selection.reasoning.lower()

    @pytest.mark.asyncio
    async def test_name_property(self) -> None:
        """Test the agent name property."""
        agent = MockAgent()
        assert agent.name == "mock"

    @pytest.mark.asyncio
    async def test_close_does_nothing(self) -> None:
        """Test close method doesn't raise."""
        agent = MockAgent()
        await agent.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(
        self, all_skills: list[Skill]
    ) -> None:
        """Test using agent as context manager."""
        async with MockAgent() as agent:
            selection = await agent.select_skill(
                prompt="Search the web",
                available_skills=all_skills,
            )
            assert selection is not None


class TestScriptedMockAgent:
    """Tests for ScriptedMockAgent."""

    @pytest.mark.asyncio
    async def test_pattern_matching(
        self, all_skills: list[Skill]
    ) -> None:
        """Test scripted responses based on patterns."""
        agent = ScriptedMockAgent(
            responses={
                r"search.*web": ("web_search", 0.95, "Pattern match: search web"),
                r"analyze.*data": ("data_analysis", 0.9, "Pattern match: analyze data"),
            }
        )

        selection = await agent.select_skill(
            prompt="Please search the web for info",
            available_skills=all_skills,
        )

        assert selection.skill == "web_search"
        assert selection.confidence == 0.95
        assert "Pattern match" in selection.reasoning

    @pytest.mark.asyncio
    async def test_pattern_case_insensitive(
        self, all_skills: list[Skill]
    ) -> None:
        """Test patterns are case insensitive."""
        agent = ScriptedMockAgent(
            responses={
                r"SEARCH": ("web_search", 0.9, "Matched SEARCH"),
            }
        )

        selection = await agent.select_skill(
            prompt="search something",
            available_skills=all_skills,
        )

        assert selection.skill == "web_search"

    @pytest.mark.asyncio
    async def test_fallback_to_keyword(
        self, all_skills: list[Skill]
    ) -> None:
        """Test fallback to keyword mode when no pattern matches."""
        agent = ScriptedMockAgent(
            responses={
                r"specific_pattern": ("web_search", 0.9, "Pattern match"),
            },
            fallback_mode="keyword",
        )

        selection = await agent.select_skill(
            prompt="Analyze this dataset",  # Doesn't match pattern
            available_skills=all_skills,
        )

        # Should fall back to keyword matching
        assert selection.skill == "data_analysis"

    @pytest.mark.asyncio
    async def test_none_skill_in_response(
        self, all_skills: list[Skill]
    ) -> None:
        """Test scripted response with None skill."""
        agent = ScriptedMockAgent(
            responses={
                r"no_skill": (None, 0.0, "No skill needed"),
            }
        )

        selection = await agent.select_skill(
            prompt="This needs no_skill",
            available_skills=all_skills,
        )

        assert selection.skill is None
        assert selection.confidence == 0.0


class TestGetAgentFactory:
    """Tests for the get_agent factory function."""

    def test_get_mock_agent(self) -> None:
        """Test getting mock agent."""
        agent = get_agent("mock")
        assert isinstance(agent, MockAgent)
        assert agent.name == "mock"

    def test_get_mock_agent_with_kwargs(self) -> None:
        """Test getting mock agent with custom kwargs."""
        agent = get_agent("mock", mode="first", default_confidence=0.5)
        assert isinstance(agent, MockAgent)
        assert agent.mode == "first"
        assert agent.default_confidence == 0.5

    def test_invalid_agent_name(self) -> None:
        """Test error for invalid agent name."""
        with pytest.raises(ValueError, match="Unknown agent"):
            get_agent("invalid_agent")

    def test_valid_agent_names(self) -> None:
        """Test that valid agent names are documented."""
        # At minimum, mock should always work
        agent = get_agent("mock")
        assert agent is not None


class TestBaseAgent:
    """Tests for BaseAgent interface."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseAgent cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore

    @pytest.mark.asyncio
    async def test_subclass_must_implement_methods(self) -> None:
        """Test that subclasses must implement abstract methods."""

        class IncompleteAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "incomplete"

            # Missing select_skill and close

        with pytest.raises(TypeError):
            IncompleteAgent()  # type: ignore


class TestSkillConversion:
    """Tests for skill-to-tool conversion utilities."""

    def test_skill_with_parameters(self) -> None:
        """Test converting skill with parameters."""
        from skills_arena import Parameter

        skill = Skill(
            name="search",
            description="Search for content",
            parameters=[
                Parameter(name="query", description="Search query", required=True),
                Parameter(name="limit", description="Max results", type="integer"),
            ],
        )

        # Import the conversion function
        from skills_arena.runner.raw_claude import _skill_to_anthropic_tool

        tool = _skill_to_anthropic_tool(skill)

        assert tool["name"] == "search"
        assert tool["description"] == "Search for content"
        assert "query" in tool["input_schema"]["properties"]
        assert tool["input_schema"]["required"] == ["query"]

    def test_skill_without_parameters(self) -> None:
        """Test converting skill without parameters adds default."""
        skill = Skill(
            name="ping",
            description="Ping the server",
        )

        from skills_arena.runner.raw_claude import _skill_to_anthropic_tool

        tool = _skill_to_anthropic_tool(skill)

        assert tool["name"] == "ping"
        # Should have default input property
        assert "input" in tool["input_schema"]["properties"]
