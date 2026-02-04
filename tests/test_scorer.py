"""Tests for the scorer module."""

import pytest

from skills_arena import (
    ELO,
    Difficulty,
    Grade,
    RatingTracker,
    Scenario,
    Scorer,
    SelectionResult,
    Skill,
    SkillFormat,
    SkillSelection,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_skill_a() -> Skill:
    """Create a sample skill A for testing."""
    return Skill(
        name="skill_a",
        description="A web search skill",
        source_format=SkillFormat.GENERIC,
    )


@pytest.fixture
def sample_skill_b() -> Skill:
    """Create a sample skill B for testing."""
    return Skill(
        name="skill_b",
        description="A content extraction skill",
        source_format=SkillFormat.GENERIC,
    )


@pytest.fixture
def sample_skill_c() -> Skill:
    """Create a sample skill C for testing."""
    return Skill(
        name="skill_c",
        description="A data analysis skill",
        source_format=SkillFormat.GENERIC,
    )


@pytest.fixture
def sample_scenario_a() -> Scenario:
    """Create a scenario expecting skill_a."""
    return Scenario(
        id="scenario_1",
        prompt="Search the web for information about Python",
        expected_skill="skill_a",
        difficulty=Difficulty.EASY,
    )


@pytest.fixture
def sample_scenario_b() -> Scenario:
    """Create a scenario expecting skill_b."""
    return Scenario(
        id="scenario_2",
        prompt="Extract content from this webpage",
        expected_skill="skill_b",
        difficulty=Difficulty.MEDIUM,
    )


def create_selection_result(
    scenario: Scenario,
    selected_skill: str | None,
    agent_name: str = "mock_agent",
    latency_ms: float = 100.0,
) -> SelectionResult:
    """Helper to create SelectionResult objects."""
    is_correct = selected_skill == scenario.expected_skill
    return SelectionResult(
        scenario=scenario,
        selection=SkillSelection(skill=selected_skill, confidence=0.9),
        agent_name=agent_name,
        is_correct=is_correct,
        latency_ms=latency_ms,
    )


# ============================================================================
# ELO Tests
# ============================================================================


class TestELO:
    """Tests for the ELO class."""

    def test_expected_score_equal_ratings(self):
        """Equal ratings should give 0.5 expected score."""
        expected = ELO.expected_score(1500, 1500)
        assert expected == pytest.approx(0.5, abs=0.01)

    def test_expected_score_higher_rated_favored(self):
        """Higher rated player should have higher expected score."""
        expected = ELO.expected_score(1600, 1400)
        assert expected > 0.5
        assert expected < 1.0

    def test_expected_score_lower_rated_underdog(self):
        """Lower rated player should have lower expected score."""
        expected = ELO.expected_score(1400, 1600)
        assert expected < 0.5
        assert expected > 0.0

    def test_expected_score_symmetry(self):
        """Expected scores should sum to 1."""
        expected_a = ELO.expected_score(1600, 1400)
        expected_b = ELO.expected_score(1400, 1600)
        assert expected_a + expected_b == pytest.approx(1.0, abs=0.001)

    def test_update_winner_gains_points(self):
        """Winner should gain points."""
        new_winner, new_loser = ELO.update(1500, 1500)
        assert new_winner > 1500
        assert new_loser < 1500

    def test_update_upset_gives_more_points(self):
        """Upset victory should result in larger rating change."""
        # Favorite wins
        favorite_win_winner, favorite_win_loser = ELO.update(1600, 1400)
        favorite_gain = favorite_win_winner - 1600

        # Underdog wins
        upset_win_winner, upset_win_loser = ELO.update(1400, 1600)
        upset_gain = upset_win_winner - 1400

        assert upset_gain > favorite_gain

    def test_update_preserves_total(self):
        """Total ELO in system should be preserved."""
        winner_elo, loser_elo = 1550, 1450
        new_winner, new_loser = ELO.update(winner_elo, loser_elo)
        assert new_winner + new_loser == winner_elo + loser_elo

    def test_update_k_factor_affects_change(self):
        """Higher k-factor should result in larger changes."""
        new_winner_low_k, new_loser_low_k = ELO.update(1500, 1500, k=16)
        new_winner_high_k, new_loser_high_k = ELO.update(1500, 1500, k=64)

        change_low_k = abs(new_winner_low_k - 1500)
        change_high_k = abs(new_winner_high_k - 1500)

        assert change_high_k > change_low_k

    def test_update_draw_underdog_gains(self):
        """In a draw, underdog should gain points."""
        new_underdog, new_favorite = ELO.update_draw(1400, 1600)
        assert new_underdog > 1400
        assert new_favorite < 1600

    def test_update_draw_equal_ratings(self):
        """Draw between equal ratings should not change much."""
        new_a, new_b = ELO.update_draw(1500, 1500)
        # Should be close to original
        assert new_a == 1500
        assert new_b == 1500

    def test_create_tracker(self):
        """Should create a RatingTracker with correct initial ratings."""
        tracker = ELO.create_tracker(["a", "b", "c"])
        assert tracker.get_rating("a") == 1500
        assert tracker.get_rating("b") == 1500
        assert tracker.get_rating("c") == 1500

    def test_create_tracker_custom_rating(self):
        """Should support custom initial rating."""
        tracker = ELO.create_tracker(["a", "b"], initial_rating=1200)
        assert tracker.get_rating("a") == 1200
        assert tracker.get_rating("b") == 1200


# ============================================================================
# RatingTracker Tests
# ============================================================================


class TestRatingTracker:
    """Tests for the RatingTracker class."""

    def test_record_match_updates_ratings(self):
        """Recording a match should update ratings."""
        tracker = RatingTracker(["a", "b"])
        initial_a = tracker.get_rating("a")
        initial_b = tracker.get_rating("b")

        tracker.record_match("a", "b")

        assert tracker.get_rating("a") > initial_a
        assert tracker.get_rating("b") < initial_b

    def test_record_match_updates_stats(self):
        """Recording a match should update win/loss stats."""
        tracker = RatingTracker(["a", "b"])
        tracker.record_match("a", "b")

        stats_a = tracker.get_stats("a")
        stats_b = tracker.get_stats("b")

        assert stats_a["wins"] == 1
        assert stats_a["losses"] == 0
        assert stats_b["wins"] == 0
        assert stats_b["losses"] == 1

    def test_record_draw(self):
        """Recording a draw should update both skills."""
        tracker = RatingTracker(["a", "b"])
        tracker.record_draw("a", "b")

        stats_a = tracker.get_stats("a")
        stats_b = tracker.get_stats("b")

        assert stats_a["draws"] == 1
        assert stats_b["draws"] == 1

    def test_get_rankings(self):
        """Should return skills sorted by rating."""
        tracker = RatingTracker(["a", "b", "c"])

        # a beats b, b beats c
        tracker.record_match("a", "b")
        tracker.record_match("b", "c")
        tracker.record_match("a", "c")

        rankings = tracker.get_rankings()

        assert rankings[0][0] == "a"
        assert rankings[2][0] == "c"

    def test_get_rating_unknown_skill(self):
        """Should raise KeyError for unknown skill."""
        tracker = RatingTracker(["a", "b"])

        with pytest.raises(KeyError):
            tracker.get_rating("unknown")

    def test_record_match_unknown_skill(self):
        """Should raise KeyError for unknown skill in match."""
        tracker = RatingTracker(["a", "b"])

        with pytest.raises(KeyError):
            tracker.record_match("a", "unknown")

    def test_add_skill(self):
        """Should allow adding new skills."""
        tracker = RatingTracker(["a"])
        tracker.add_skill("b")

        assert tracker.get_rating("b") == 1500

    def test_add_skill_custom_rating(self):
        """Should support custom rating for new skill."""
        tracker = RatingTracker(["a"])
        tracker.add_skill("b", initial_rating=1400)

        assert tracker.get_rating("b") == 1400

    def test_multiple_matches(self):
        """Should handle multiple matches correctly."""
        tracker = RatingTracker(["a", "b"])

        for _ in range(10):
            tracker.record_match("a", "b")

        # After 10 wins, a should be much higher than b
        assert tracker.get_rating("a") > tracker.get_rating("b")
        assert tracker.get_stats("a")["wins"] == 10
        assert tracker.get_stats("b")["losses"] == 10


# ============================================================================
# Scorer Tests
# ============================================================================


class TestScorer:
    """Tests for the Scorer class."""

    def test_score_evaluation_empty_results(self, sample_skill_a):
        """Empty results should return zero scores."""
        result = Scorer.score_evaluation(sample_skill_a, [])

        assert result.skill == "skill_a"
        assert result.score == 0.0
        assert result.grade == Grade.F
        assert result.scenarios_run == 0

    def test_score_evaluation_perfect_score(self, sample_skill_a, sample_scenario_a):
        """Perfect selection should give high score."""
        results = [
            create_selection_result(sample_scenario_a, "skill_a")
            for _ in range(10)
        ]

        result = Scorer.score_evaluation(sample_skill_a, results)

        assert result.selection_rate == 1.0
        assert result.invocation_accuracy == 1.0
        assert result.score > 80

    def test_score_evaluation_selection_rate(self, sample_skill_a, sample_scenario_a):
        """Should calculate correct selection rate."""
        # 5 selections out of 10 scenarios
        results = []
        for i in range(10):
            selected = "skill_a" if i < 5 else "skill_b"
            results.append(create_selection_result(sample_scenario_a, selected))

        result = Scorer.score_evaluation(sample_skill_a, results)

        assert result.selection_rate == 0.5

    def test_score_evaluation_per_agent(self, sample_skill_a, sample_scenario_a):
        """Should track per-agent metrics."""
        results = [
            create_selection_result(sample_scenario_a, "skill_a", agent_name="agent_1"),
            create_selection_result(sample_scenario_a, "skill_a", agent_name="agent_1"),
            create_selection_result(sample_scenario_a, "skill_b", agent_name="agent_2"),
        ]

        result = Scorer.score_evaluation(sample_skill_a, results)

        assert "agent_1" in result.per_agent
        assert "agent_2" in result.per_agent
        assert result.per_agent["agent_1"].selection_rate == 1.0
        assert result.per_agent["agent_2"].selection_rate == 0.0

    def test_score_comparison_empty(self, sample_skill_a, sample_skill_b):
        """Empty comparison should return empty result."""
        result = Scorer.score_comparison([sample_skill_a, sample_skill_b], [])

        assert result.winner == ""
        assert result.scenarios_run == 0

    def test_score_comparison_winner(
        self, sample_skill_a, sample_skill_b, sample_scenario_a, sample_scenario_b
    ):
        """Should determine winner based on selection count."""
        results = [
            # skill_a selected more often
            create_selection_result(sample_scenario_a, "skill_a"),
            create_selection_result(sample_scenario_a, "skill_a"),
            create_selection_result(sample_scenario_a, "skill_a"),
            create_selection_result(sample_scenario_b, "skill_b"),
        ]

        result = Scorer.score_comparison(
            [sample_skill_a, sample_skill_b], results
        )

        assert result.winner == "skill_a"

    def test_score_comparison_selection_rates(
        self, sample_skill_a, sample_skill_b, sample_scenario_a
    ):
        """Should calculate selection rates for each skill."""
        results = [
            create_selection_result(sample_scenario_a, "skill_a"),
            create_selection_result(sample_scenario_a, "skill_a"),
            create_selection_result(sample_scenario_a, "skill_b"),
            create_selection_result(sample_scenario_a, "skill_b"),
        ]

        result = Scorer.score_comparison(
            [sample_skill_a, sample_skill_b], results
        )

        assert result.selection_rates["skill_a"] == 0.5
        assert result.selection_rates["skill_b"] == 0.5

    def test_score_battle_empty(self, sample_skill_a, sample_skill_b):
        """Empty battle should return empty result."""
        result = Scorer.score_battle([sample_skill_a, sample_skill_b], [])

        assert result.leaderboard == []
        assert result.elo_ratings == {}

    def test_score_battle_leaderboard(
        self, sample_skill_a, sample_skill_b, sample_skill_c, sample_scenario_a
    ):
        """Should produce ranked leaderboard."""
        # Create scenarios that prefer skill_a > skill_b > skill_c
        results = [
            create_selection_result(sample_scenario_a, "skill_a"),
            create_selection_result(
                Scenario(id="s2", prompt="test", expected_skill="skill_b"),
                "skill_a",
            ),
            create_selection_result(
                Scenario(id="s3", prompt="test", expected_skill="skill_c"),
                "skill_b",
            ),
        ]

        result = Scorer.score_battle(
            [sample_skill_a, sample_skill_b, sample_skill_c], results
        )

        assert len(result.leaderboard) == 3
        assert result.leaderboard[0].rank == 1

    def test_score_battle_elo_ratings(
        self, sample_skill_a, sample_skill_b, sample_scenario_a
    ):
        """Should update ELO ratings based on matchups."""
        results = [
            create_selection_result(sample_scenario_a, "skill_a"),
            create_selection_result(sample_scenario_a, "skill_a"),
        ]

        result = Scorer.score_battle(
            [sample_skill_a, sample_skill_b], results
        )

        # skill_a won more, should have higher ELO
        assert result.elo_ratings["skill_a"] > result.elo_ratings["skill_b"]


class TestScorerMetrics:
    """Tests for additional Scorer metric calculations."""

    def test_calculate_precision_recall_empty(self):
        """Empty results should return zero metrics."""
        metrics = Scorer.calculate_precision_recall("skill_a", [])

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_calculate_precision_recall_perfect(self, sample_scenario_a):
        """Perfect selection should give 1.0 metrics."""
        results = [
            create_selection_result(sample_scenario_a, "skill_a")
            for _ in range(5)
        ]

        metrics = Scorer.calculate_precision_recall("skill_a", results)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_calculate_precision_recall_false_positives(
        self, sample_scenario_a, sample_scenario_b
    ):
        """Should correctly handle false positives."""
        results = [
            # True positive: expected skill_a, selected skill_a
            create_selection_result(sample_scenario_a, "skill_a"),
            # False positive: expected skill_b, selected skill_a
            create_selection_result(sample_scenario_b, "skill_a"),
        ]

        metrics = Scorer.calculate_precision_recall("skill_a", results)

        # 1 TP, 1 FP -> precision = 0.5
        assert metrics["precision"] == 0.5
        # 1 TP, 0 FN -> recall = 1.0
        assert metrics["recall"] == 1.0

    def test_calculate_confusion_matrix(self, sample_scenario_a, sample_scenario_b):
        """Should build correct confusion matrix."""
        results = [
            create_selection_result(sample_scenario_a, "skill_a"),  # Correct
            create_selection_result(sample_scenario_a, "skill_b"),  # Wrong
            create_selection_result(sample_scenario_b, "skill_b"),  # Correct
        ]

        matrix = Scorer.calculate_confusion_matrix(
            ["skill_a", "skill_b"], results
        )

        # skill_a expected: 1 correct (skill_a), 1 wrong (skill_b)
        assert matrix["skill_a"]["skill_a"] == 1
        assert matrix["skill_a"]["skill_b"] == 1

        # skill_b expected: 1 correct (skill_b)
        assert matrix["skill_b"]["skill_b"] == 1
