"""Core metrics calculation for skill evaluation.

This module provides the Scorer class which calculates various metrics
from evaluation results including selection rate, accuracy, and generates
result objects for single evaluations, comparisons, and battles.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ..models import (
    AgentResult,
    BattleResult,
    ComparisonResult,
    EvaluationResult,
    Grade,
    Matchup,
    RankedSkill,
    ScenarioDetail,
    SelectionResult,
    Skill,
)
from .elo import ELO

if TYPE_CHECKING:
    pass


class Scorer:
    """Calculates metrics from evaluation results.

    The Scorer processes SelectionResult lists and produces the various
    result types (EvaluationResult, ComparisonResult, BattleResult).

    Example:
        ```python
        # Score a single skill evaluation
        result = Scorer.score_evaluation(skill, selection_results)
        print(f"Score: {result.score}, Grade: {result.grade}")

        # Score a comparison between skills
        result = Scorer.score_comparison(skills, selection_results)
        print(f"Winner: {result.winner}")

        # Score a battle royale
        result = Scorer.score_battle(skills, selection_results)
        print(result.leaderboard)
        ```
    """

    @staticmethod
    def score_evaluation(
        skill: Skill,
        results: list[SelectionResult],
    ) -> EvaluationResult:
        """Calculate metrics for a single skill evaluation.

        This method analyzes how well a skill performs across all scenarios,
        calculating selection rate, accuracy, and other metrics.

        Args:
            skill: The skill being evaluated.
            results: List of selection results from running scenarios.

        Returns:
            EvaluationResult with comprehensive metrics.

        Example:
            ```python
            result = Scorer.score_evaluation(my_skill, results)
            print(f"Selection rate: {result.selection_rate:.1%}")
            print(f"Accuracy: {result.invocation_accuracy:.1%}")
            ```
        """
        if not results:
            return EvaluationResult(
                skill=skill.name,
                score=0.0,
                grade=Grade.F,
                selection_rate=0.0,
                false_positive_rate=0.0,
                invocation_accuracy=0.0,
                per_agent={},
                insights=[],
                scenarios_run=0,
            )

        # Group results by agent
        by_agent: dict[str, list[SelectionResult]] = defaultdict(list)
        for result in results:
            by_agent[result.agent_name].append(result)

        # Calculate per-agent metrics
        per_agent_results: dict[str, AgentResult] = {}
        for agent_name, agent_results in by_agent.items():
            per_agent_results[agent_name] = Scorer._calculate_agent_result(
                agent_name, skill.name, agent_results
            )

        # Calculate overall metrics (weighted average across agents)
        total_scenarios = len(results)

        # Selection rate: how often was this skill selected?
        times_selected = sum(
            1 for r in results if r.selection.skill == skill.name
        )
        selection_rate = times_selected / total_scenarios if total_scenarios > 0 else 0.0

        # Correct selections: selected when it should have been
        correct_selections = sum(
            1
            for r in results
            if r.selection.skill == skill.name and r.scenario.expected_skill == skill.name
        )

        # False positives: selected when shouldn't have been
        false_positives = sum(
            1
            for r in results
            if r.selection.skill == skill.name and r.scenario.expected_skill != skill.name
        )

        # Scenarios where this skill should have been selected
        should_select_scenarios = sum(
            1 for r in results if r.scenario.expected_skill == skill.name
        )

        # False positive rate
        other_scenarios = total_scenarios - should_select_scenarios
        false_positive_rate = (
            false_positives / other_scenarios if other_scenarios > 0 else 0.0
        )

        # Invocation accuracy: when selected, was it correct?
        invocation_accuracy = (
            correct_selections / times_selected if times_selected > 0 else 0.0
        )

        # Calculate overall score
        # Score formula: weighted combination of metrics
        # - Selection rate when appropriate: 50%
        # - Low false positive rate: 30%
        # - Invocation accuracy: 20%
        recall = (
            correct_selections / should_select_scenarios
            if should_select_scenarios > 0
            else 0.0
        )
        score = (
            recall * 50  # Being selected when should be
            + (1 - false_positive_rate) * 30  # Not being selected when shouldn't
            + invocation_accuracy * 20  # Accuracy when selected
        )

        return EvaluationResult(
            skill=skill.name,
            score=score,
            grade=Grade.from_score(score),
            selection_rate=selection_rate,
            false_positive_rate=false_positive_rate,
            invocation_accuracy=invocation_accuracy,
            per_agent=per_agent_results,
            insights=[],  # Insights are generated separately
            scenarios_run=total_scenarios,
        )

    @staticmethod
    def score_comparison(
        skills: list[Skill],
        results: list[SelectionResult],
    ) -> ComparisonResult:
        """Calculate head-to-head comparison metrics.

        This method analyzes how skills compete against each other,
        determining a winner and calculating relative selection rates.

        Args:
            skills: List of skills being compared.
            results: List of selection results from head-to-head scenarios.

        Returns:
            ComparisonResult with winner, selection rates, and head-to-head stats.

        Example:
            ```python
            result = Scorer.score_comparison([skill_a, skill_b], results)
            print(f"Winner: {result.winner}")
            print(f"Selection rates: {result.selection_rates}")
            ```
        """
        if not results or not skills:
            return ComparisonResult(
                winner="",
                selection_rates={},
                head_to_head={},
                insights=[],
                per_agent={},
                scenarios_run=0,
                scenario_details=[],
                steals={},
            )

        skill_names = [s.name for s in skills]
        total_scenarios = len(results)

        # Calculate selection rates for each skill
        selection_counts: dict[str, int] = defaultdict(int)
        for result in results:
            if result.selection.skill:
                selection_counts[result.selection.skill] += 1

        selection_rates = {
            name: selection_counts[name] / total_scenarios if total_scenarios > 0 else 0.0
            for name in skill_names
        }

        # Calculate head-to-head matchups
        # For each scenario, record which skill "won"
        head_to_head: dict[str, dict[str, int]] = {
            name: {other: 0 for other in skill_names if other != name}
            for name in skill_names
        }

        # Build scenario details and track steals
        scenario_details: list[ScenarioDetail] = []
        steals: dict[str, list[str]] = {name: [] for name in skill_names}

        # Group results by scenario to compare skills
        by_scenario: dict[str, list[SelectionResult]] = defaultdict(list)
        for result in results:
            by_scenario[result.scenario.id].append(result)

        for scenario_id, scenario_results in by_scenario.items():
            # Find the selected skill (if any)
            selected_skill = None
            expected_skill = None
            reasoning = ""

            for r in scenario_results:
                expected_skill = r.scenario.expected_skill
                # Get Claude's reasoning text from the selection
                if r.selection.reasoning:
                    reasoning = r.selection.reasoning
                if r.selection.skill in skill_names:
                    selected_skill = r.selection.skill
                    break

            # Record head-to-head wins
            if selected_skill:
                for other_skill in skill_names:
                    if other_skill != selected_skill:
                        head_to_head[selected_skill][other_skill] += 1

            # Build scenario detail
            prompt = scenario_results[0].scenario.prompt if scenario_results else ""
            # Only mark as stolen if expected_skill was specified AND a different skill was selected
            was_stolen = (
                selected_skill is not None
                and expected_skill is not None
                and expected_skill != ""  # Blind scenarios can't be stolen
                and selected_skill != expected_skill
            )

            detail = ScenarioDetail(
                scenario_id=scenario_id,
                prompt=prompt,
                expected_skill=expected_skill or "",
                selected_skill=selected_skill,
                reasoning=reasoning,
                was_stolen=was_stolen,
            )
            scenario_details.append(detail)

            # Track steals (only for non-blind scenarios)
            if was_stolen and expected_skill and expected_skill in steals:
                steals[expected_skill].append(scenario_id)

        # Determine winner based on total selections
        if selection_counts:
            # Filter to only skills being compared
            valid_counts = {
                name: count
                for name, count in selection_counts.items()
                if name in skill_names
            }
            if valid_counts:
                winner = max(valid_counts.items(), key=lambda x: x[1])[0]
            else:
                winner = skill_names[0] if skill_names else ""
        else:
            winner = skill_names[0] if skill_names else ""

        # Calculate per-agent comparisons
        by_agent: dict[str, list[SelectionResult]] = defaultdict(list)
        for result in results:
            by_agent[result.agent_name].append(result)

        # Note: per_agent contains ComparisonResult which could be recursive
        # For simplicity, we'll leave it empty in the base implementation
        # The Arena class can populate this if needed

        return ComparisonResult(
            winner=winner,
            selection_rates=selection_rates,
            head_to_head=head_to_head,
            insights=[],  # Insights are generated separately
            per_agent={},
            scenarios_run=total_scenarios,
            scenario_details=scenario_details,
            steals=steals,
        )

    @staticmethod
    def score_battle(
        skills: list[Skill],
        results: list[SelectionResult],
        k_factor: int = ELO.DEFAULT_K,
    ) -> BattleResult:
        """Calculate ELO rankings and leaderboard for battle royale.

        This method processes all matchup results to produce final ELO ratings
        and a ranked leaderboard.

        Args:
            skills: List of all skills in the battle.
            results: List of selection results from all matchups.
            k_factor: ELO K-factor for rating volatility (default 32).

        Returns:
            BattleResult with leaderboard, ELO ratings, and matchup history.

        Example:
            ```python
            result = Scorer.score_battle(skills, results)
            for rank in result.leaderboard:
                print(f"{rank.rank}. {rank.name} (ELO: {rank.elo})")
            ```
        """
        if not results or not skills:
            return BattleResult(
                leaderboard=[],
                elo_ratings={},
                matchups=[],
                insights=[],
                scenarios_run=0,
            )

        skill_names = [s.name for s in skills]

        # Create ELO tracker
        tracker = ELO.create_tracker(skill_names, k=k_factor)

        # Process each scenario as a matchup
        matchups: list[Matchup] = []
        by_scenario: dict[str, list[SelectionResult]] = defaultdict(list)

        for result in results:
            by_scenario[result.scenario.id].append(result)

        for scenario_id, scenario_results in by_scenario.items():
            # Find which skill was selected
            selected_skill = None
            expected_skill = None

            for r in scenario_results:
                expected_skill = r.scenario.expected_skill
                if r.selection.skill in skill_names:
                    selected_skill = r.selection.skill
                    break

            if not expected_skill:
                continue

            # Create matchup record
            # In a battle scenario, we compare the selected skill vs expected
            # The "winner" is whoever got selected (even if wrong)
            for other_skill in skill_names:
                if other_skill == selected_skill:
                    continue

                matchup = Matchup(
                    skill_a=selected_skill or expected_skill,
                    skill_b=other_skill,
                    winner=selected_skill,
                    scenario_id=scenario_id,
                )
                matchups.append(matchup)

                # Update ELO if there was a selection
                if selected_skill:
                    tracker.record_match(selected_skill, other_skill)

        # Build leaderboard
        rankings = tracker.get_rankings()
        leaderboard: list[RankedSkill] = []

        for rank, (name, elo) in enumerate(rankings, start=1):
            stats = tracker.get_stats(name)

            # Calculate selection rate
            total_scenarios = len(by_scenario)
            times_selected = sum(
                1
                for result in results
                if result.selection.skill == name
            )
            selection_rate = times_selected / total_scenarios if total_scenarios > 0 else 0.0

            leaderboard.append(
                RankedSkill(
                    rank=rank,
                    name=name,
                    elo=elo,
                    wins=stats["wins"],
                    losses=stats["losses"],
                    selection_rate=selection_rate,
                )
            )

        return BattleResult(
            leaderboard=leaderboard,
            elo_ratings=dict(tracker.ratings),
            matchups=matchups,
            insights=[],  # Insights are generated separately
            scenarios_run=len(results),
        )

    @staticmethod
    def _calculate_agent_result(
        agent_name: str,
        skill_name: str,
        results: list[SelectionResult],
    ) -> AgentResult:
        """Calculate metrics for a specific agent.

        Args:
            agent_name: Name of the agent framework.
            skill_name: Name of the skill being evaluated.
            results: Results for this agent only.

        Returns:
            AgentResult with agent-specific metrics.
        """
        if not results:
            return AgentResult(
                agent_name=agent_name,
                selection_rate=0.0,
                accuracy=0.0,
                scenarios_run=0,
                avg_latency_ms=0.0,
            )

        total = len(results)

        # Selection rate
        times_selected = sum(1 for r in results if r.selection.skill == skill_name)
        selection_rate = times_selected / total if total > 0 else 0.0

        # Accuracy (correct selections / total scenarios for this skill)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0.0

        # Average latency
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return AgentResult(
            agent_name=agent_name,
            selection_rate=selection_rate,
            accuracy=accuracy,
            scenarios_run=total,
            avg_latency_ms=avg_latency,
        )

    @staticmethod
    def calculate_precision_recall(
        skill_name: str,
        results: list[SelectionResult],
    ) -> dict[str, float]:
        """Calculate precision and recall for a skill.

        Precision: Of all times selected, how many were correct?
        Recall: Of all times it should have been selected, how many were?

        Args:
            skill_name: Name of the skill.
            results: List of selection results.

        Returns:
            Dictionary with 'precision', 'recall', and 'f1' scores.
        """
        if not results:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # True positives: selected and should have been
        true_positives = sum(
            1
            for r in results
            if r.selection.skill == skill_name and r.scenario.expected_skill == skill_name
        )

        # False positives: selected but shouldn't have been
        false_positives = sum(
            1
            for r in results
            if r.selection.skill == skill_name and r.scenario.expected_skill != skill_name
        )

        # False negatives: should have been selected but wasn't
        false_negatives = sum(
            1
            for r in results
            if r.selection.skill != skill_name and r.scenario.expected_skill == skill_name
        )

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def calculate_confusion_matrix(
        skill_names: list[str],
        results: list[SelectionResult],
    ) -> dict[str, dict[str, int]]:
        """Calculate confusion matrix for skill selection.

        Shows how often each expected skill was selected as each actual skill.

        Args:
            skill_names: List of skill names.
            results: List of selection results.

        Returns:
            Nested dict: confusion[expected][actual] = count
        """
        # Initialize matrix
        confusion: dict[str, dict[str, int]] = {
            expected: {actual: 0 for actual in skill_names + ["none"]}
            for expected in skill_names
        }

        for result in results:
            expected = result.scenario.expected_skill
            actual = result.selection.skill or "none"

            if expected in confusion:
                if actual in confusion[expected]:
                    confusion[expected][actual] += 1
                else:
                    # Handle case where selected skill isn't in our list
                    confusion[expected]["none"] += 1

        return confusion
