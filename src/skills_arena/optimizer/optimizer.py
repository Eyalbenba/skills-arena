"""LLM-powered skill description optimizer.

Uses competition data (stolen scenarios, agent reasoning) and research-backed
best practices to rewrite skill descriptions for higher selection rates.
"""

from __future__ import annotations

import json

import anthropic

from ..exceptions import OptimizerError
from ..models import ComparisonResult, Parameter, Skill

OPTIMIZATION_PROMPT = '''You are an expert at writing AI agent skill descriptions that maximize selection rates.

## Current Skill
Name: {skill_name}
Description:
{skill_description}

When to use:
{when_to_use}

Parameters:
{parameters}

## Competition Results
This skill competed against: {competitor_names}
- Selection rate: {selection_rate:.0%} (selected {wins} out of {total} scenarios)
- Winner: {winner}

### Scenarios Where This Skill LOST (stolen by competitors)
These are scenarios that SHOULD have selected this skill but a competitor was chosen instead.
Study the agent's reasoning to understand WHY this skill lost.

{stolen_scenarios}

### Scenarios Where This Skill WON
These are working well - preserve what makes them succeed.

{won_scenarios}

## Best Practices for Skill Descriptions
Apply these research-backed techniques to improve the description:

1. **Add concrete usage examples** - Skills with 3-5 specific examples see ~25% higher selection rates
2. **Use "Use this when..." clauses** - Explicit trigger conditions help agents match intent
3. **Add "Do NOT use for..." boundaries** - Negative examples reduce false positives and clarify scope
4. **Include domain-specific keywords** - Match the vocabulary users actually use in prompts
5. **Describe output format** - Agents prefer skills that clearly describe what they return
6. **Add comparison phrases** - "Unlike X, this skill..." helps agents differentiate
7. **Front-load the key differentiator** - First sentence should capture the unique value
8. **Include parameter examples** - Show concrete parameter values, not just types
9. **Specify performance characteristics** - Speed, depth, accuracy expectations
10. **Match user intent patterns** - Use phrases like "when the user wants to...", "for tasks requiring..."

## Token Budget
CRITICAL: The description MUST be under 200 tokens (~150 words).
Every token in a skill description consumes the agent's context window. Research shows:
- A simple tool = ~96 tokens. Aim for this range.
- Longer descriptions increase confusion and incorrect tool selection.
- Be precise, not verbose. One clear sentence beats a paragraph.

The "when_to_use" array is SEPARATE from the description - do NOT embed usage examples
inside the description text. Keep them in the "when_to_use" field only.

## Instructions
Rewrite the skill's description, when_to_use, and parameters to maximize selection rate.

Rules:
- Keep the skill name unchanged
- Preserve all existing capabilities - do NOT remove functionality
- Focus on the scenarios that were LOST - make the description clearly cover those cases
- KEEP THE DESCRIPTION UNDER 200 TOKENS. Concise and precise beats long and detailed.
- Do NOT include "Use this when..." or "Do NOT use for..." sections inside the description.
  Put trigger conditions in "when_to_use" instead.
- Use natural language, not marketing speak
- Do NOT fabricate capabilities the skill doesn't have

Return a JSON object with:
- "description": The improved description text (MUST be under 200 tokens)
- "when_to_use": Array of 3-5 specific trigger examples (short phrases, not paragraphs)
- "parameters": Array of parameter objects with "name", "description", "type", "required" fields (preserve existing params, improve descriptions)
- "reasoning": Brief explanation of what you changed and why

Return ONLY the JSON object, no other text.'''


class SkillOptimizer:
    """Optimizes skill descriptions using LLM-powered rewriting.

    Uses competition data to identify weaknesses and rewrites descriptions
    to improve agent selection rates.

    Example:
        ```python
        optimizer = SkillOptimizer()
        improved = await optimizer.optimize_description(
            skill=my_skill,
            comparison_result=baseline,
            competitors=[comp_a, comp_b],
        )
        ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3,
        api_key: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            if self._api_key:
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            else:
                self._client = anthropic.AsyncAnthropic()
        return self._client

    def _format_stolen_scenarios(
        self, result: ComparisonResult, skill_name: str
    ) -> str:
        """Format scenarios where the skill lost to a competitor."""
        stolen = [
            d for d in result.scenario_details
            if d.expected_skill == skill_name and d.was_stolen
        ]
        if not stolen:
            return "(none - skill won all its scenarios)"

        lines = []
        for d in stolen:
            lines.append(f"- Prompt: \"{d.prompt}\"")
            lines.append(f"  Expected: {d.expected_skill} | Selected: {d.selected_skill}")
            if d.reasoning:
                lines.append(f"  Agent reasoning: {d.reasoning}")
            lines.append("")
        return "\n".join(lines)

    def _format_won_scenarios(
        self, result: ComparisonResult, skill_name: str
    ) -> str:
        """Format scenarios where the skill was correctly selected."""
        won = [
            d for d in result.scenario_details
            if d.expected_skill == skill_name and not d.was_stolen
        ]
        if not won:
            return "(none)"

        lines = []
        for d in won[:5]:  # Limit to 5 to keep prompt manageable
            lines.append(f"- Prompt: \"{d.prompt}\"")
            if d.reasoning:
                lines.append(f"  Agent reasoning: {d.reasoning}")
            lines.append("")
        if len(won) > 5:
            lines.append(f"  ... and {len(won) - 5} more winning scenarios")
        return "\n".join(lines)

    async def optimize_description(
        self,
        skill: Skill,
        comparison_result: ComparisonResult,
        competitors: list[Skill],
    ) -> tuple[Skill, str]:
        """Optimize a skill description based on competition data.

        Args:
            skill: The skill to optimize.
            comparison_result: Results from comparing against competitors.
            competitors: The competitor skills.

        Returns:
            Tuple of (optimized Skill, reasoning string).

        Raises:
            OptimizerError: If optimization fails.
        """
        skill_name = skill.name
        selection_rate = comparison_result.selection_rates.get(skill_name, 0.0)
        total = comparison_result.scenarios_run
        wins = round(selection_rate * total)

        # Build prompt
        when_to_use = "\n".join(f"- {u}" for u in skill.when_to_use) if skill.when_to_use else "(none defined)"
        parameters = "\n".join(
            f"- {p.name} ({p.type}{', required' if p.required else ''}): {p.description}"
            for p in skill.parameters
        ) if skill.parameters else "(none defined)"

        prompt = OPTIMIZATION_PROMPT.format(
            skill_name=skill_name,
            skill_description=skill.description,
            when_to_use=when_to_use,
            parameters=parameters,
            competitor_names=", ".join(c.name for c in competitors),
            selection_rate=selection_rate,
            wins=wins,
            total=total,
            winner=comparison_result.winner,
            stolen_scenarios=self._format_stolen_scenarios(comparison_result, skill_name),
            won_scenarios=self._format_won_scenarios(comparison_result, skill_name),
        )

        # Call LLM
        client = self._get_client()
        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as e:
            raise OptimizerError(
                f"Anthropic API error: {e}", skill_name=skill_name
            ) from e

        # Extract response text
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        if not response_text:
            raise OptimizerError(
                "Empty response from LLM", skill_name=skill_name
            )

        # Parse JSON (strip markdown code blocks if present)
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            start_idx = 1 if lines[0].strip().startswith("```") else 0
            end_idx = len(lines)
            for i in range(start_idx, len(lines)):
                if i > 0 and lines[i].strip() == "```":
                    end_idx = i
                    break
            text = "\n".join(lines[start_idx:end_idx])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise OptimizerError(
                f"Failed to parse optimizer response as JSON: {e}",
                skill_name=skill_name,
            ) from e

        if not isinstance(data, dict):
            raise OptimizerError(
                f"Expected JSON object, got {type(data).__name__}",
                skill_name=skill_name,
            )

        # Build optimized skill
        new_params = skill.parameters  # Default: keep existing
        if "parameters" in data and isinstance(data["parameters"], list):
            new_params = []
            for p in data["parameters"]:
                if isinstance(p, dict) and "name" in p:
                    new_params.append(Parameter(
                        name=p["name"],
                        description=p.get("description", ""),
                        type=p.get("type", "string"),
                        required=p.get("required", False),
                    ))

        optimized = Skill(
            name=skill.name,
            description=data.get("description", skill.description),
            parameters=new_params,
            when_to_use=data.get("when_to_use", skill.when_to_use),
            source_format=skill.source_format,
            token_count=skill.token_count,
            raw_content=skill.raw_content,
            source_path=skill.source_path,
        )

        reasoning = data.get("reasoning", "")
        return optimized, reasoning
