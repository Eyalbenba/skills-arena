"""Text reporter for Skills Arena results.

Provides human-readable formatting for evaluation, comparison,
and battle royale results.
"""

from __future__ import annotations

from ..models import BattleResult, ComparisonResult, EvaluationResult


class TextReporter:
    """Formats results as human-readable text.

    Example:
        ```python
        reporter = TextReporter()
        print(reporter.format_evaluation(result))
        ```
    """

    BAR_WIDTH = 30

    @staticmethod
    def _bar(fraction: float, width: int = 30) -> str:
        """Render a simple bar chart segment."""
        filled = round(fraction * width)
        return "\u2588" * filled + "\u2591" * (width - filled)

    def format_evaluation(self, result: EvaluationResult) -> str:
        """Format an EvaluationResult as text.

        Args:
            result: The evaluation result to format.

        Returns:
            Formatted string.
        """
        lines = [
            f"Evaluation: {result.skill}",
            f"{'=' * 50}",
            f"Score: {result.score:.1f} / 100  (Grade: {result.grade.value})",
            f"Scenarios: {result.scenarios_run}",
            "",
            f"Selection Rate:      {self._bar(result.selection_rate)} {result.selection_rate:.0%}",
            f"Invocation Accuracy: {self._bar(result.invocation_accuracy)} {result.invocation_accuracy:.0%}",
            f"False Positive Rate: {self._bar(result.false_positive_rate)} {result.false_positive_rate:.0%}",
        ]

        if result.per_agent:
            lines.append("")
            lines.append("Per Agent:")
            for agent_name, agent_result in result.per_agent.items():
                lines.append(
                    f"  {agent_name}: "
                    f"sel={agent_result.selection_rate:.0%}  "
                    f"acc={agent_result.accuracy:.0%}  "
                    f"latency={agent_result.avg_latency_ms:.0f}ms"
                )

        return "\n".join(lines)

    def format_comparison(self, result: ComparisonResult) -> str:
        """Format a ComparisonResult as text.

        Args:
            result: The comparison result to format.

        Returns:
            Formatted string.
        """
        lines = [
            f"Comparison Results",
            f"{'=' * 50}",
            f"Winner: {result.winner}",
            f"Scenarios: {result.scenarios_run}",
            "",
            "Selection Rates:",
        ]

        # Sort by selection rate descending
        sorted_rates = sorted(
            result.selection_rates.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, rate in sorted_rates:
            marker = " <-- winner" if name == result.winner else ""
            lines.append(f"  {name:30s} {self._bar(rate)} {rate:.0%}{marker}")

        # Steal detection
        stolen_skills = {k: v for k, v in result.steals.items() if v}
        if stolen_skills:
            lines.append("")
            lines.append("Steal Detection:")
            for skill_name, scenario_ids in stolen_skills.items():
                lines.append(f"  {skill_name} lost {len(scenario_ids)} scenario(s) to a competitor")

        # Scenario details with steals
        stolen_details = [d for d in result.scenario_details if d.was_stolen]
        if stolen_details:
            lines.append("")
            lines.append("Stolen Scenarios:")
            for detail in stolen_details:
                lines.append(
                    f"  [{detail.scenario_id}] expected={detail.expected_skill} "
                    f"selected={detail.selected_skill}"
                )
                if detail.prompt:
                    prompt_preview = detail.prompt[:80] + ("..." if len(detail.prompt) > 80 else "")
                    lines.append(f"    prompt: {prompt_preview}")

        return "\n".join(lines)

    def format_battle(self, result: BattleResult) -> str:
        """Format a BattleResult as text.

        Args:
            result: The battle result to format.

        Returns:
            Formatted string.
        """
        lines = [
            f"Battle Royale Results",
            f"{'=' * 50}",
            f"Scenarios: {result.scenarios_run}",
            "",
            "Leaderboard:",
            f"  {'Rank':<6} {'Skill':<30} {'ELO':<8} {'W/L':<10} {'Sel Rate'}",
            f"  {'-' * 70}",
        ]

        for entry in result.leaderboard:
            lines.append(
                f"  {entry.rank:<6} {entry.name:<30} {entry.elo:<8} "
                f"{entry.wins}W/{entry.losses}L    {entry.selection_rate:.0%}"
            )

        return "\n".join(lines)


def print_results(result: EvaluationResult | ComparisonResult | BattleResult) -> None:
    """Convenience function to print formatted results.

    Automatically detects the result type and prints the appropriate format.

    Args:
        result: Any Skills Arena result object.

    Example:
        ```python
        from skills_arena import Arena, print_results

        result = Arena().compare(skills, task="web search")
        print_results(result)
        ```
    """
    reporter = TextReporter()

    if isinstance(result, EvaluationResult):
        print(reporter.format_evaluation(result))
    elif isinstance(result, ComparisonResult):
        print(reporter.format_comparison(result))
    elif isinstance(result, BattleResult):
        print(reporter.format_battle(result))
    else:
        raise TypeError(f"Unsupported result type: {type(result).__name__}")
