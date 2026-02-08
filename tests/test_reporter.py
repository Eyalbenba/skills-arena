"""Tests for the reporter module."""

from skills_arena import (
    BattleResult,
    ComparisonResult,
    EvaluationResult,
    Grade,
    RankedSkill,
    ScenarioDetail,
    TextReporter,
    print_results,
)


class TestTextReporterEvaluation:
    """Tests for TextReporter.format_evaluation()."""

    def test_basic_evaluation(self) -> None:
        """Test formatting a basic evaluation result."""
        result = EvaluationResult(
            skill="Tavily Search",
            score=85.0,
            grade=Grade.B,
            selection_rate=0.8,
            false_positive_rate=0.1,
            invocation_accuracy=0.9,
            scenarios_run=50,
        )
        reporter = TextReporter()
        output = reporter.format_evaluation(result)

        assert "Tavily Search" in output
        assert "85.0" in output
        assert "B" in output
        assert "80%" in output
        assert "50" in output

    def test_zero_score_evaluation(self) -> None:
        """Test formatting an evaluation with zero scores."""
        result = EvaluationResult(
            skill="Bad Skill",
            score=0.0,
            grade=Grade.F,
            selection_rate=0.0,
            false_positive_rate=0.0,
            invocation_accuracy=0.0,
            scenarios_run=10,
        )
        reporter = TextReporter()
        output = reporter.format_evaluation(result)

        assert "Bad Skill" in output
        assert "F" in output


class TestTextReporterComparison:
    """Tests for TextReporter.format_comparison()."""

    def test_basic_comparison(self) -> None:
        """Test formatting a comparison result."""
        result = ComparisonResult(
            winner="Tavily Search",
            selection_rates={"Tavily Search": 0.6, "Firecrawl": 0.4},
            head_to_head={},
            scenarios_run=20,
            scenario_details=[],
            steals={},
        )
        reporter = TextReporter()
        output = reporter.format_comparison(result)

        assert "Tavily Search" in output
        assert "winner" in output.lower()
        assert "60%" in output
        assert "40%" in output

    def test_comparison_with_steals(self) -> None:
        """Test formatting a comparison with stolen scenarios."""
        result = ComparisonResult(
            winner="Tavily Search",
            selection_rates={"Tavily Search": 0.7, "Firecrawl": 0.3},
            head_to_head={},
            scenarios_run=10,
            scenario_details=[
                ScenarioDetail(
                    scenario_id="custom-abc123",
                    prompt="Scrape pricing from stripe.com",
                    expected_skill="Firecrawl",
                    selected_skill="Tavily Search",
                    was_stolen=True,
                ),
            ],
            steals={"Firecrawl": ["custom-abc123"]},
        )
        reporter = TextReporter()
        output = reporter.format_comparison(result)

        assert "Steal Detection" in output
        assert "Firecrawl" in output
        assert "1 scenario" in output


class TestTextReporterBattle:
    """Tests for TextReporter.format_battle()."""

    def test_basic_battle(self) -> None:
        """Test formatting a battle result."""
        result = BattleResult(
            leaderboard=[
                RankedSkill(rank=1, name="Tavily Search", elo=1550, wins=8, losses=2, selection_rate=0.6),
                RankedSkill(rank=2, name="Firecrawl", elo=1450, wins=2, losses=8, selection_rate=0.4),
            ],
            elo_ratings={"Tavily Search": 1550, "Firecrawl": 1450},
            matchups=[],
            scenarios_run=20,
        )
        reporter = TextReporter()
        output = reporter.format_battle(result)

        assert "Tavily Search" in output
        assert "1550" in output
        assert "Leaderboard" in output
        assert "8W" in output


class TestPrintResults:
    """Tests for the print_results convenience function."""

    def test_print_evaluation(self, capsys) -> None:
        """Test print_results with EvaluationResult."""
        result = EvaluationResult(
            skill="Test",
            score=50.0,
            grade=Grade.F,
            selection_rate=0.5,
            scenarios_run=10,
        )
        print_results(result)
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "50.0" in captured.out

    def test_print_comparison(self, capsys) -> None:
        """Test print_results with ComparisonResult."""
        result = ComparisonResult(
            winner="A",
            selection_rates={"A": 0.6, "B": 0.4},
            head_to_head={},
            scenarios_run=10,
            scenario_details=[],
            steals={},
        )
        print_results(result)
        captured = capsys.readouterr()
        assert "A" in captured.out
        assert "winner" in captured.out.lower()

    def test_print_battle(self, capsys) -> None:
        """Test print_results with BattleResult."""
        result = BattleResult(
            leaderboard=[
                RankedSkill(rank=1, name="X", elo=1500, wins=1, losses=0, selection_rate=1.0),
            ],
            elo_ratings={"X": 1500},
            matchups=[],
            scenarios_run=5,
        )
        print_results(result)
        captured = capsys.readouterr()
        assert "X" in captured.out
        assert "1500" in captured.out

    def test_unsupported_type(self) -> None:
        """Test print_results raises TypeError for unsupported input."""
        import pytest

        with pytest.raises(TypeError, match="Unsupported result type"):
            print_results("not a result")  # type: ignore
