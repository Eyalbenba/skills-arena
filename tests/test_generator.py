"""Tests for Skills Arena scenario generators."""

import pytest

from skills_arena import (
    BaseGenerator,
    Difficulty,
    LLMGenerator,
    MockGenerator,
    Scenario,
    Skill,
    Task,
)
from skills_arena.exceptions import GeneratorError


class TestMockGenerator:
    """Tests for MockGenerator."""

    @pytest.fixture
    def skills(self) -> list[Skill]:
        """Create test skills."""
        return [
            Skill(name="web-search", description="Search the web for information"),
            Skill(name="file-reader", description="Read contents of files"),
        ]

    @pytest.fixture
    def task(self) -> Task:
        """Create test task."""
        return Task(description="search and file operations")

    @pytest.fixture
    def generator(self) -> MockGenerator:
        """Create mock generator."""
        return MockGenerator()

    @pytest.mark.asyncio
    async def test_generate_returns_scenarios(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test that generate returns a list of scenarios."""
        scenarios = await generator.generate(task, skills, count=10)
        assert isinstance(scenarios, list)
        assert len(scenarios) == 10
        assert all(isinstance(s, Scenario) for s in scenarios)

    @pytest.mark.asyncio
    async def test_generate_respects_count(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test that generate returns the requested number of scenarios."""
        for count in [5, 10, 20, 50]:
            scenarios = await generator.generate(task, skills, count=count)
            assert len(scenarios) == count

    @pytest.mark.asyncio
    async def test_generate_distributes_across_skills(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test that scenarios are distributed across skills."""
        scenarios = await generator.generate(task, skills, count=20)

        # Count scenarios per skill
        skill_counts = {}
        for s in scenarios:
            skill_counts[s.expected_skill] = skill_counts.get(s.expected_skill, 0) + 1

        # All skills should have scenarios
        for skill in skills:
            assert skill.name in skill_counts

    @pytest.mark.asyncio
    async def test_generate_has_unique_ids(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test that all scenario IDs are unique."""
        scenarios = await generator.generate(task, skills, count=50)
        ids = [s.id for s in scenarios]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_generate_includes_difficulties(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test that scenarios include various difficulties."""
        scenarios = await generator.generate(task, skills, count=30)
        difficulties = {s.difficulty for s in scenarios}
        assert len(difficulties) >= 2  # At least 2 difficulty levels

    @pytest.mark.asyncio
    async def test_generate_with_adversarial(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test that adversarial scenarios are included when requested."""
        scenarios = await generator.generate(
            task, skills, count=30, include_adversarial=True
        )
        adversarial = [s for s in scenarios if s.is_adversarial]
        assert len(adversarial) > 0

    @pytest.mark.asyncio
    async def test_generate_without_adversarial(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test that no adversarial scenarios when disabled."""
        scenarios = await generator.generate(
            task, skills, count=30, include_adversarial=False
        )
        adversarial = [s for s in scenarios if s.is_adversarial]
        assert len(adversarial) == 0

    @pytest.mark.asyncio
    async def test_generate_with_empty_skills(
        self, generator: MockGenerator, task: Task
    ) -> None:
        """Test behavior with no skills."""
        scenarios = await generator.generate(task, [], count=10)
        assert len(scenarios) == 0

    @pytest.mark.asyncio
    async def test_generate_with_single_skill(
        self, generator: MockGenerator, task: Task
    ) -> None:
        """Test generation with a single skill."""
        skills = [Skill(name="only-skill", description="The only skill")]
        scenarios = await generator.generate(task, skills, count=10)
        assert len(scenarios) == 10
        assert all(s.expected_skill == "only-skill" for s in scenarios)

    def test_generate_sync(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test synchronous wrapper."""
        scenarios = generator.generate_sync(task, skills, count=10)
        assert len(scenarios) == 10

    @pytest.mark.asyncio
    async def test_scenarios_have_prompts(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test that all scenarios have non-empty prompts."""
        scenarios = await generator.generate(task, skills, count=10)
        for scenario in scenarios:
            assert scenario.prompt
            assert len(scenario.prompt) > 0

    @pytest.mark.asyncio
    async def test_scenarios_have_tags(
        self, generator: MockGenerator, task: Task, skills: list[Skill]
    ) -> None:
        """Test that scenarios have tags."""
        scenarios = await generator.generate(task, skills, count=10)
        for scenario in scenarios:
            assert isinstance(scenario.tags, list)


class TestLLMGeneratorUnit:
    """Unit tests for LLMGenerator (no API calls)."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        gen = LLMGenerator()
        assert gen.model == "claude-sonnet-4-20250514"
        assert gen.temperature == 0.7

    def test_init_custom_model(self) -> None:
        """Test custom model initialization."""
        gen = LLMGenerator(model="claude-3-5-sonnet-20241022")
        assert gen.model == "claude-3-5-sonnet-20241022"

    def test_init_custom_temperature(self) -> None:
        """Test custom temperature initialization."""
        gen = LLMGenerator(temperature=0.5)
        assert gen.temperature == 0.5

    def test_init_with_api_key(self) -> None:
        """Test initialization with API key."""
        gen = LLMGenerator(api_key="test-key")
        assert gen._api_key == "test-key"

    def test_format_skills_description(self) -> None:
        """Test skills description formatting."""
        gen = LLMGenerator()
        skills = [
            Skill(
                name="search",
                description="Search the web",
                when_to_use=["Find information", "Look up data"],
            ),
        ]
        formatted = gen._format_skills_description(skills)
        assert "search" in formatted
        assert "Search the web" in formatted
        assert "Find information" in formatted

    def test_build_prompt(self) -> None:
        """Test prompt building."""
        gen = LLMGenerator()
        task = Task(
            description="web search",
            domains=["enterprise", "developer"],
            edge_cases=["rate limiting"],
        )
        skills = [Skill(name="search", description="Search the web")]
        prompt = gen._build_prompt(task, skills, count=10, include_adversarial=True)

        assert "web search" in prompt
        assert "enterprise" in prompt
        assert "rate limiting" in prompt
        assert "10" in prompt
        assert "adversarial" in prompt.lower()

    def test_build_prompt_without_adversarial(self) -> None:
        """Test prompt building without adversarial instructions."""
        gen = LLMGenerator()
        task = Task(description="simple task")
        skills = [Skill(name="skill", description="A skill")]
        prompt = gen._build_prompt(task, skills, count=5, include_adversarial=False)

        # Should not include the adversarial instructions block
        assert "Include adversarial/edge case scenarios" not in prompt

    def test_parse_response_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        gen = LLMGenerator()
        skills = [Skill(name="search", description="Search")]

        response = '''[
            {
                "prompt": "Find news about AI",
                "expected_skill": "search",
                "difficulty": "easy",
                "tags": ["news"],
                "is_adversarial": false,
                "reasoning": "Clear search request"
            }
        ]'''

        scenarios = gen._parse_response(response, skills)
        assert len(scenarios) == 1
        assert scenarios[0]["prompt"] == "Find news about AI"

    def test_parse_response_markdown_code_block(self) -> None:
        """Test parsing response wrapped in markdown code block."""
        gen = LLMGenerator()
        skills = [Skill(name="search", description="Search")]

        response = '''```json
[
    {
        "prompt": "Search for something",
        "expected_skill": "search",
        "difficulty": "medium",
        "tags": [],
        "is_adversarial": false,
        "reasoning": "Test"
    }
]
```'''

        scenarios = gen._parse_response(response, skills)
        assert len(scenarios) == 1

    def test_parse_response_invalid_json(self) -> None:
        """Test parsing invalid JSON raises error."""
        gen = LLMGenerator()
        skills = [Skill(name="search", description="Search")]

        with pytest.raises(GeneratorError):
            gen._parse_response("not valid json", skills)

    def test_parse_response_filters_invalid_skills(self) -> None:
        """Test that scenarios with invalid skills are filtered."""
        gen = LLMGenerator()
        skills = [Skill(name="search", description="Search")]

        response = '''[
            {"prompt": "Test", "expected_skill": "search"},
            {"prompt": "Test2", "expected_skill": "nonexistent"}
        ]'''

        scenarios = gen._parse_response(response, skills)
        assert len(scenarios) == 1
        assert scenarios[0]["expected_skill"] == "search"

    def test_parse_response_fuzzy_skill_match(self) -> None:
        """Test fuzzy matching of skill names."""
        gen = LLMGenerator()
        skills = [Skill(name="web-search", description="Search")]

        response = '''[
            {"prompt": "Test", "expected_skill": "Web-Search"}
        ]'''

        scenarios = gen._parse_response(response, skills)
        assert len(scenarios) == 1
        assert scenarios[0]["expected_skill"] == "web-search"

    def test_dict_to_scenario(self) -> None:
        """Test converting dict to Scenario."""
        gen = LLMGenerator()

        data = {
            "prompt": "Find something",
            "expected_skill": "search",
            "difficulty": "hard",
            "tags": ["test", "search"],
            "is_adversarial": True,
        }

        scenario = gen._dict_to_scenario(data, 0)

        assert scenario.prompt == "Find something"
        assert scenario.expected_skill == "search"
        assert scenario.difficulty == Difficulty.HARD
        assert "test" in scenario.tags
        assert scenario.is_adversarial is True
        assert scenario.id.startswith("scenario-")

    def test_dict_to_scenario_default_difficulty(self) -> None:
        """Test default difficulty when not specified."""
        gen = LLMGenerator()

        data = {"prompt": "Test", "expected_skill": "skill"}
        scenario = gen._dict_to_scenario(data, 0)

        assert scenario.difficulty == Difficulty.MEDIUM

    def test_dict_to_scenario_invalid_difficulty(self) -> None:
        """Test handling of invalid difficulty."""
        gen = LLMGenerator()

        data = {"prompt": "Test", "expected_skill": "skill", "difficulty": "invalid"}
        scenario = gen._dict_to_scenario(data, 0)

        assert scenario.difficulty == Difficulty.MEDIUM


class TestBaseGenerator:
    """Tests for BaseGenerator abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseGenerator cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseGenerator()

    def test_subclass_must_implement_generate(self) -> None:
        """Test that subclasses must implement generate."""

        class IncompleteGenerator(BaseGenerator):
            pass

        with pytest.raises(TypeError):
            IncompleteGenerator()


class TestGeneratorIntegration:
    """Integration tests that can be run with mocks."""

    @pytest.fixture
    def multi_skills(self) -> list[Skill]:
        """Create multiple test skills."""
        return [
            Skill(
                name="web-search",
                description="Search the web for information",
                when_to_use=["Finding current news", "Looking up facts"],
            ),
            Skill(
                name="file-reader",
                description="Read and analyze files",
                when_to_use=["Reading documents", "Analyzing code"],
            ),
            Skill(
                name="calculator",
                description="Perform mathematical calculations",
                when_to_use=["Math operations", "Unit conversions"],
            ),
        ]

    @pytest.fixture
    def detailed_task(self) -> Task:
        """Create a detailed task."""
        return Task(
            description="general assistant capabilities",
            domains=["developer", "student", "professional"],
            edge_cases=["ambiguous requests", "multi-step tasks"],
        )

    @pytest.mark.asyncio
    async def test_mock_generator_end_to_end(
        self, multi_skills: list[Skill], detailed_task: Task
    ) -> None:
        """Test complete mock generation flow."""
        generator = MockGenerator()
        scenarios = await generator.generate(
            detailed_task,
            multi_skills,
            count=30,
            include_adversarial=True,
        )

        # Verify basic structure
        assert len(scenarios) == 30

        # Verify all skills represented
        skill_names = {s.expected_skill for s in scenarios}
        for skill in multi_skills:
            assert skill.name in skill_names

        # Verify variety
        difficulties = {s.difficulty for s in scenarios}
        assert len(difficulties) >= 2

        # Verify IDs are unique
        ids = [s.id for s in scenarios]
        assert len(ids) == len(set(ids))
