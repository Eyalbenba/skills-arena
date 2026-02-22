"""Text reporter for Skills Arena results.

Provides human-readable formatting for evaluation, comparison,
and battle royale results.
"""

from __future__ import annotations

from ..models import BattleResult, ComparisonResult, EvaluationResult, OptimizationResult
from ..parser.base import estimate_tokens


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

    def format_optimization(self, result: OptimizationResult) -> str:
        """Format an OptimizationResult as text.

        Args:
            result: The optimization result to format.

        Returns:
            Formatted string.
        """
        W = 70
        lines = [
            "=" * W,
            "OPTIMIZATION RESULTS",
            "=" * W,
            f"Skill:       {result.original_skill.name}",
            f"Competitors: {', '.join(result.competitors)}",
            f"Scenarios:   {result.scenarios_used}  |  Iterations: {len(result.iterations)}",
            "",
        ]

        # Token counts
        tokens_before = estimate_tokens(result.original_skill.description)
        tokens_after = estimate_tokens(result.optimized_skill.description)
        token_delta = tokens_after - tokens_before
        token_arrow = "+" if token_delta > 0 else ""
        token_warn = "  (over 200!)" if tokens_after > 200 else ""

        # Before/After summary with bars
        imp = result.total_improvement
        arrow = "+" if imp > 0 else ""
        lines.append("Before -> After:")
        lines.append(
            f"  Selection Rate:  {self._bar(result.selection_rate_before, 20)} {result.selection_rate_before:.0%}"
            f"  ->  {self._bar(result.selection_rate_after, 20)} {result.selection_rate_after:.0%}"
            f"  ({arrow}{imp:.0%})"
        )
        lines.append(
            f"  Grade:           {result.grade_before.value:>3s}"
            f"  ->  {result.grade_after.value:<3s}"
        )
        lines.append(
            f"  Tokens:          {tokens_before:>3d}"
            f"  ->  {tokens_after:<3d} ({token_arrow}{token_delta}){token_warn}"
        )

        # Per-iteration details
        for iteration in result.iterations:
            lines.append("")
            lines.append("-" * W)
            imp_i = iteration.improvement
            arrow_i = "+" if imp_i > 0 else ""
            status = "improved" if imp_i > 0 else ("no change" if imp_i == 0 else "regressed")
            lines.append(
                f"Iteration {iteration.iteration}:  "
                f"{iteration.selection_rate_before:.0%} -> {iteration.selection_rate_after:.0%}  "
                f"({arrow_i}{imp_i:.0%})  [{status}]"
            )
            if iteration.reasoning:
                lines.append("")
                # Word-wrap reasoning at ~65 chars
                reasoning = iteration.reasoning
                while len(reasoning) > 65:
                    wrap_at = reasoning.rfind(" ", 0, 65)
                    if wrap_at == -1:
                        wrap_at = 65
                    lines.append(f"  {reasoning[:wrap_at]}")
                    reasoning = reasoning[wrap_at:].lstrip()
                if reasoning:
                    lines.append(f"  {reasoning}")

            # Show scenario-level detail from the AFTER comparison
            after_details = iteration.comparison_after.scenario_details
            if after_details:
                skill_name = result.original_skill.name
                stolen = [d for d in after_details if d.was_stolen]
                won = [d for d in after_details if d.expected_skill == skill_name and not d.was_stolen]
                lines.append("")
                lines.append(f"  Scenarios:  {len(won)} won  |  {len(stolen)} stolen")
                for d in after_details:
                    tag = " STOLEN" if d.was_stolen else ""
                    prompt_preview = d.prompt[:50] + ("..." if len(d.prompt) > 50 else "")
                    lines.append(
                        f"    {d.expected_skill:>15s} -> {(d.selected_skill or 'None'):<15s}"
                        f"{tag:>7s}  {prompt_preview}"
                    )

        # Optimized description
        lines.append("")
        lines.append("=" * W)
        lines.append("OPTIMIZED DESCRIPTION")
        lines.append("-" * W)
        # Preserve original line breaks in description
        for desc_line in result.optimized_skill.description.splitlines():
            lines.append(desc_line)

        # When to use
        if result.optimized_skill.when_to_use:
            lines.append("")
            lines.append("WHEN TO USE")
            lines.append("-" * W)
            for use in result.optimized_skill.when_to_use:
                lines.append(f"  - {use}")

        lines.append("")
        lines.append("=" * W)

        return "\n".join(lines)


def print_results(result: EvaluationResult | ComparisonResult | BattleResult | OptimizationResult) -> None:
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
    elif isinstance(result, OptimizationResult):
        print(reporter.format_optimization(result))
    else:
        raise TypeError(f"Unsupported result type: {type(result).__name__}")
