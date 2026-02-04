"""Tests for Skills Arena models."""


from skills_arena import (
    BattleResult,
    ComparisonResult,
    Difficulty,
    EvaluationResult,
    Grade,
    Insight,
    Parameter,
    Progress,
    RankedSkill,
    Scenario,
    SelectionResult,
    Skill,
    SkillFormat,
    SkillSelection,
    Task,
)


class TestGrade:
    """Tests for Grade enum."""

    def test_from_score_a_plus(self) -> None:
        """Test A+ grade for scores >= 97."""
        assert Grade.from_score(100) == Grade.A_PLUS
        assert Grade.from_score(97) == Grade.A_PLUS

    def test_from_score_a(self) -> None:
        """Test A grade for scores 93-96."""
        assert Grade.from_score(96) == Grade.A
        assert Grade.from_score(93) == Grade.A

    def test_from_score_a_minus(self) -> None:
        """Test A- grade for scores 90-92."""
        assert Grade.from_score(92) == Grade.A_MINUS
        assert Grade.from_score(90) == Grade.A_MINUS

    def test_from_score_b_plus(self) -> None:
        """Test B+ grade for scores 87-89."""
        assert Grade.from_score(89) == Grade.B_PLUS
        assert Grade.from_score(87) == Grade.B_PLUS

    def test_from_score_b(self) -> None:
        """Test B grade for scores 83-86."""
        assert Grade.from_score(86) == Grade.B
        assert Grade.from_score(83) == Grade.B

    def test_from_score_b_minus(self) -> None:
        """Test B- grade for scores 80-82."""
        assert Grade.from_score(82) == Grade.B_MINUS
        assert Grade.from_score(80) == Grade.B_MINUS

    def test_from_score_c_plus(self) -> None:
        """Test C+ grade for scores 77-79."""
        assert Grade.from_score(79) == Grade.C_PLUS
        assert Grade.from_score(77) == Grade.C_PLUS

    def test_from_score_c(self) -> None:
        """Test C grade for scores 73-76."""
        assert Grade.from_score(76) == Grade.C
        assert Grade.from_score(73) == Grade.C

    def test_from_score_c_minus(self) -> None:
        """Test C- grade for scores 70-72."""
        assert Grade.from_score(72) == Grade.C_MINUS
        assert Grade.from_score(70) == Grade.C_MINUS

    def test_from_score_d(self) -> None:
        """Test D grade for scores 60-69."""
        assert Grade.from_score(69) == Grade.D
        assert Grade.from_score(60) == Grade.D

    def test_from_score_f(self) -> None:
        """Test F grade for scores < 60."""
        assert Grade.from_score(59) == Grade.F
        assert Grade.from_score(0) == Grade.F


class TestParameter:
    """Tests for Parameter model."""

    def test_create_simple(self) -> None:
        """Test creating a simple parameter."""
        param = Parameter(name="query", description="Search query")
        assert param.name == "query"
        assert param.description == "Search query"
        assert param.type == "string"
        assert param.required is False
        assert param.default is None

    def test_create_with_all_fields(self) -> None:
        """Test creating a parameter with all fields."""
        param = Parameter(
            name="limit",
            description="Maximum results",
            type="integer",
            required=True,
            default=10,
        )
        assert param.name == "limit"
        assert param.type == "integer"
        assert param.required is True
        assert param.default == 10


class TestSkill:
    """Tests for Skill model."""

    def test_create_minimal(self) -> None:
        """Test creating a skill with minimal fields."""
        skill = Skill(name="web-search", description="Search the web")
        assert skill.name == "web-search"
        assert skill.description == "Search the web"
        assert skill.parameters == []
        assert skill.when_to_use == []
        assert skill.source_format == SkillFormat.GENERIC
        assert skill.token_count == 0

    def test_create_with_parameters(self) -> None:
        """Test creating a skill with parameters."""
        params = [
            Parameter(name="query", description="Search query", required=True),
            Parameter(name="limit", description="Max results", type="integer"),
        ]
        skill = Skill(
            name="web-search",
            description="Search the web",
            parameters=params,
            source_format=SkillFormat.CLAUDE_CODE,
        )
        assert len(skill.parameters) == 2
        assert skill.parameters[0].name == "query"
        assert skill.source_format == SkillFormat.CLAUDE_CODE


class TestTask:
    """Tests for Task model."""

    def test_from_string(self) -> None:
        """Test creating a Task from a string."""
        task = Task.from_string("web search and content extraction")
        assert task.description == "web search and content extraction"
        assert task.domains == []
        assert task.edge_cases == []

    def test_create_with_domains(self) -> None:
        """Test creating a Task with domains."""
        task = Task(
            description="web search",
            domains=["enterprise", "developer"],
            edge_cases=["rate limiting"],
        )
        assert len(task.domains) == 2
        assert "enterprise" in task.domains
        assert len(task.edge_cases) == 1


class TestScenario:
    """Tests for Scenario model."""

    def test_create_scenario(self) -> None:
        """Test creating a scenario."""
        scenario = Scenario(
            id="scenario-001",
            prompt="Find the latest news about AI",
            expected_skill="web-search",
            difficulty=Difficulty.MEDIUM,
            tags=["news", "search"],
        )
        assert scenario.id == "scenario-001"
        assert scenario.expected_skill == "web-search"
        assert scenario.difficulty == Difficulty.MEDIUM
        assert len(scenario.tags) == 2
        assert scenario.is_adversarial is False

    def test_adversarial_scenario(self) -> None:
        """Test creating an adversarial scenario."""
        scenario = Scenario(
            id="adversarial-001",
            prompt="Find AI news but don't use the internet",
            expected_skill="web-search",
            is_adversarial=True,
        )
        assert scenario.is_adversarial is True


class TestSkillSelection:
    """Tests for SkillSelection model."""

    def test_no_selection(self) -> None:
        """Test when no skill is selected."""
        selection = SkillSelection()
        assert selection.skill is None
        assert selection.confidence == 0.0

    def test_with_selection(self) -> None:
        """Test when a skill is selected."""
        selection = SkillSelection(
            skill="web-search",
            confidence=0.95,
            reasoning="User wants to search the web",
        )
        assert selection.skill == "web-search"
        assert selection.confidence == 0.95


class TestSelectionResult:
    """Tests for SelectionResult model."""

    def test_correct_selection(self) -> None:
        """Test a correct selection result."""
        scenario = Scenario(
            id="test-001",
            prompt="Search for news",
            expected_skill="web-search",
        )
        selection = SkillSelection(skill="web-search", confidence=0.9)
        result = SelectionResult(
            scenario=scenario,
            selection=selection,
            agent_name="claude-code",
            is_correct=True,
            latency_ms=150.0,
        )
        assert result.is_correct is True
        assert result.agent_name == "claude-code"
        assert result.latency_ms == 150.0


class TestInsight:
    """Tests for Insight model."""

    def test_create_insight(self) -> None:
        """Test creating an insight."""
        insight = Insight(
            type="optimization",
            message="Your skill description is too verbose",
            severity="warning",
            suggestion="Reduce description by 30%",
        )
        assert insight.type == "optimization"
        assert insight.severity == "warning"
        assert insight.suggestion is not None


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_create_result(self) -> None:
        """Test creating an evaluation result."""
        result = EvaluationResult(
            skill="web-search",
            score=85.5,
            grade=Grade.B,
            selection_rate=0.85,
            scenarios_run=50,
        )
        assert result.skill == "web-search"
        assert result.score == 85.5
        assert result.grade == Grade.B
        assert result.selection_rate == 0.85


class TestComparisonResult:
    """Tests for ComparisonResult model."""

    def test_create_result(self) -> None:
        """Test creating a comparison result."""
        result = ComparisonResult(
            winner="skill-a",
            selection_rates={"skill-a": 0.6, "skill-b": 0.4},
            scenarios_run=100,
        )
        assert result.winner == "skill-a"
        assert result.selection_rates["skill-a"] == 0.6


class TestBattleResult:
    """Tests for BattleResult model."""

    def test_create_result(self) -> None:
        """Test creating a battle result."""
        leaderboard = [
            RankedSkill(rank=1, name="skill-a", elo=1550),
            RankedSkill(rank=2, name="skill-b", elo=1480),
        ]
        result = BattleResult(
            leaderboard=leaderboard,
            elo_ratings={"skill-a": 1550, "skill-b": 1480},
            scenarios_run=200,
        )
        assert len(result.leaderboard) == 2
        assert result.leaderboard[0].name == "skill-a"
        assert result.elo_ratings["skill-a"] == 1550


class TestProgress:
    """Tests for Progress model."""

    def test_create_progress(self) -> None:
        """Test creating progress update."""
        progress = Progress(
            stage="Generating scenarios",
            percent=45.0,
            message="Generated 45 of 100 scenarios",
        )
        assert progress.stage == "Generating scenarios"
        assert progress.percent == 45.0
