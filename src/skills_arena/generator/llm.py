"""LLM-powered scenario generation.

This module provides the LLMGenerator class that uses Claude or other LLMs
to generate diverse, realistic test scenarios from task descriptions.
"""

from __future__ import annotations

import json
import random
import uuid
from typing import Any

import anthropic

from ..config import Config
from ..exceptions import GeneratorError
from ..models import Difficulty, Scenario, Skill, Task
from .base import BaseGenerator

# Prompt template for scenario generation (multi-skill balanced)
GENERATION_PROMPT = '''You are generating test scenarios for evaluating AI agent skill selection.

## Task Description
{task_description}

{domain_context}
{edge_case_context}

## Available Skills
{skills_description}

## Instructions
Generate {count} diverse user prompts that would require using one of these skills.
For EACH prompt, indicate which skill SHOULD be selected.

Requirements:
1. Cover different user personas (developer, business user, casual user, expert)
2. Include simple, moderate, and complex requests
3. Vary the phrasing, vocabulary, and specificity
4. Ensure realistic, natural-sounding prompts
5. Balance scenarios across all skills roughly equally
{adversarial_instructions}

## Output Format
Return a JSON array of scenarios. Each scenario must have:
- "prompt": The user's message/request
- "expected_skill": Name of the skill that SHOULD be selected
- "difficulty": One of "easy", "medium", "hard"
- "tags": Array of relevant tags (e.g., ["search", "technical", "ambiguous"])
- "is_adversarial": Boolean, true if this is an edge case
- "reasoning": Brief explanation of why this skill should be selected

Example:
```json
[
  {{
    "prompt": "Find the latest news about electric vehicles",
    "expected_skill": "web-search",
    "difficulty": "easy",
    "tags": ["search", "news", "simple"],
    "is_adversarial": false,
    "reasoning": "Clear search request for current information"
  }}
]
```

Generate exactly {count} scenarios. Return ONLY the JSON array, no other text.'''

# Prompt template for single-skill focused scenario generation
SINGLE_SKILL_PROMPT = '''You are generating test scenarios for a specific AI agent skill.

## Target Skill
{skill_description}

## Instructions
Generate {count} diverse user prompts that would naturally require using this skill.
These are scenarios where THIS SKILL is clearly the best choice.

Requirements:
1. Cover different user personas (developer, business user, casual user, expert)
2. Include simple, moderate, and complex requests
3. Vary the phrasing - don't always mention the skill by name
4. Ensure realistic, natural-sounding prompts that real users would ask
5. Focus on the skill's core strengths and use cases

## Output Format
Return a JSON array of scenarios. Each scenario must have:
- "prompt": The user's message/request
- "difficulty": One of "easy", "medium", "hard"
- "tags": Array of relevant tags

Example:
```json
[
  {{
    "prompt": "Find the latest news about electric vehicles",
    "difficulty": "easy",
    "tags": ["search", "news", "simple"]
  }}
]
```

Generate exactly {count} scenarios. Return ONLY the JSON array, no other text.'''

ADVERSARIAL_INSTRUCTIONS = '''
6. Include adversarial/edge case scenarios:
   - Ambiguous requests where multiple skills could apply
   - Requests with conflicting requirements
   - Unusual phrasings or indirect requests
   - Scenarios testing boundary conditions
   - Requests that SEEM to match one skill but actually need another'''


class LLMGenerator(BaseGenerator):
    """LLM-powered scenario generator using Claude.

    This generator uses Claude to create diverse, realistic test scenarios
    based on the task description and available skills.

    Attributes:
        model: The Claude model to use for generation.
        temperature: Temperature for generation (higher = more diverse).
        api_key: Anthropic API key.

    Example:
        ```python
        generator = LLMGenerator(model="claude-sonnet-4-20250514")
        scenarios = await generator.generate(
            task=Task(description="web search"),
            skills=[search_skill, extract_skill],
            count=50,
        )
        ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        api_key: str | None = None,
        config: Config | None = None,
    ):
        """Initialize the LLM generator.

        Args:
            model: Claude model to use for generation.
            temperature: Temperature for generation (0.0-2.0).
            api_key: Anthropic API key. If not provided, uses config or env var.
            config: Optional Config object to get settings from.
        """
        self.model = model
        self.temperature = temperature

        # Get API key from config or parameter
        if api_key:
            self._api_key = api_key
        elif config and config.anthropic_api_key:
            self._api_key = config.anthropic_api_key
        else:
            self._api_key = None  # Will be fetched from env on first use

        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            if self._api_key:
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            else:
                # Will use ANTHROPIC_API_KEY env var
                self._client = anthropic.AsyncAnthropic()
        return self._client

    def _format_skills_description(self, skills: list[Skill]) -> str:
        """Format skills for the prompt."""
        descriptions = []
        for skill in skills:
            desc = f"### {skill.name}\n"
            desc += f"Description: {skill.description}\n"
            if skill.when_to_use:
                desc += "When to use:\n"
                for use_case in skill.when_to_use[:5]:  # Limit examples
                    desc += f"  - {use_case}\n"
            if skill.parameters:
                desc += "Parameters:\n"
                for param in skill.parameters[:5]:  # Limit parameters
                    required = " (required)" if param.required else ""
                    desc += f"  - {param.name}: {param.description}{required}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)

    async def generate_for_skill(
        self,
        skill: Skill,
        count: int = 10,
    ) -> list[Scenario]:
        """Generate scenarios specifically for one skill.

        These are scenarios where THIS skill should clearly be selected.
        Used for competitive analysis - can another skill steal these?

        Args:
            skill: The skill to generate scenarios for.
            count: Number of scenarios to generate.

        Returns:
            List of Scenario objects, all expecting this skill.
        """
        # Format skill description
        skill_desc = f"### {skill.name}\n"
        skill_desc += f"Description: {skill.description}\n"
        if skill.when_to_use:
            skill_desc += "When to use:\n"
            for use_case in skill.when_to_use[:5]:
                skill_desc += f"  - {use_case}\n"

        prompt = SINGLE_SKILL_PROMPT.format(
            skill_description=skill_desc,
            count=count,
        )

        # Call Claude
        client = self._get_client()
        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as e:
            raise GeneratorError(f"Anthropic API error: {e}") from e

        # Extract text
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        # Parse JSON
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            start_idx = 1 if lines[0].strip().startswith("```") else 0
            end_idx = len(lines)
            for i in range(start_idx, len(lines)):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            text = "\n".join(lines[start_idx:end_idx])

        try:
            scenario_dicts = json.loads(text)
        except json.JSONDecodeError as e:
            raise GeneratorError(f"Failed to parse JSON: {e}") from e

        # Convert to Scenario objects (all expect this skill)
        scenarios = []
        for i, data in enumerate(scenario_dicts):
            if not isinstance(data, dict) or "prompt" not in data:
                continue

            difficulty_str = data.get("difficulty", "medium").lower()
            try:
                difficulty = Difficulty(difficulty_str)
            except ValueError:
                difficulty = Difficulty.MEDIUM

            scenario = Scenario(
                id=f"skill-{skill.name}-{uuid.uuid4().hex[:8]}",
                prompt=data["prompt"],
                expected_skill=skill.name,  # Always this skill
                difficulty=difficulty,
                tags=data.get("tags", []) + [f"source:{skill.name}"],
                is_adversarial=False,
            )
            scenarios.append(scenario)

        return scenarios

    def _build_prompt(
        self,
        task: Task,
        skills: list[Skill],
        count: int,
        include_adversarial: bool,
    ) -> str:
        """Build the generation prompt."""
        # Build domain context
        domain_context = ""
        if task.domains:
            domain_context = f"## Target Domains\nInclude scenarios for: {', '.join(task.domains)}"

        # Build edge case context
        edge_case_context = ""
        if task.edge_cases:
            edge_case_context = (
                "## Specific Edge Cases to Include\n"
                + "\n".join(f"- {ec}" for ec in task.edge_cases)
            )

        # Build adversarial instructions
        adversarial_instructions = ADVERSARIAL_INSTRUCTIONS if include_adversarial else ""

        # Format the prompt
        return GENERATION_PROMPT.format(
            task_description=task.description,
            domain_context=domain_context,
            edge_case_context=edge_case_context,
            skills_description=self._format_skills_description(skills),
            count=count,
            adversarial_instructions=adversarial_instructions,
        )

    def _parse_response(self, response_text: str, skills: list[Skill]) -> list[dict[str, Any]]:
        """Parse the LLM response into scenario dictionaries.

        Args:
            response_text: Raw response from the LLM.
            skills: List of valid skills for validation.

        Returns:
            List of parsed scenario dictionaries.

        Raises:
            GeneratorError: If parsing fails.
        """
        # Extract JSON from response (handle markdown code blocks)
        text = response_text.strip()
        if text.startswith("```"):
            # Remove markdown code block
            lines = text.split("\n")
            # Find start and end of JSON
            start_idx = 0
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    if start_idx == 0 and i == 0:
                        start_idx = 1
                    elif i > start_idx:
                        end_idx = i
                        break
            text = "\n".join(lines[start_idx:end_idx])

        try:
            scenarios = json.loads(text)
        except json.JSONDecodeError as e:
            raise GeneratorError(
                f"Failed to parse LLM response as JSON: {e}",
                task=None,
            ) from e

        if not isinstance(scenarios, list):
            raise GeneratorError(
                f"Expected JSON array, got {type(scenarios).__name__}",
                task=None,
            )

        # Validate each scenario
        valid_skill_names = {s.name for s in skills}
        validated = []
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue

            # Required fields
            if "prompt" not in scenario or "expected_skill" not in scenario:
                continue

            # Validate expected_skill exists
            expected_skill = scenario["expected_skill"]
            if expected_skill not in valid_skill_names:
                # Try to find closest match
                closest = self._find_closest_skill(expected_skill, valid_skill_names)
                if closest:
                    scenario["expected_skill"] = closest
                else:
                    continue

            validated.append(scenario)

        return validated

    def _find_closest_skill(self, name: str, valid_names: set[str]) -> str | None:
        """Find the closest matching skill name."""
        name_lower = name.lower()
        for valid in valid_names:
            if valid.lower() == name_lower:
                return valid
            if name_lower in valid.lower() or valid.lower() in name_lower:
                return valid
        return None

    def _dict_to_scenario(self, data: dict[str, Any], index: int) -> Scenario:
        """Convert a dictionary to a Scenario object."""
        # Parse difficulty
        difficulty_str = data.get("difficulty", "medium").lower()
        try:
            difficulty = Difficulty(difficulty_str)
        except ValueError:
            difficulty = Difficulty.MEDIUM

        # Generate unique ID
        scenario_id = f"scenario-{uuid.uuid4().hex[:8]}"

        return Scenario(
            id=scenario_id,
            prompt=data["prompt"],
            expected_skill=data["expected_skill"],
            difficulty=difficulty,
            tags=data.get("tags", []),
            is_adversarial=data.get("is_adversarial", False),
        )

    async def generate(
        self,
        task: Task,
        skills: list[Skill],
        count: int = 50,
        include_adversarial: bool = True,
    ) -> list[Scenario]:
        """Generate test scenarios using Claude.

        Args:
            task: The task description.
            skills: List of skills to generate scenarios for.
            count: Number of scenarios to generate.
            include_adversarial: Whether to include adversarial scenarios.

        Returns:
            List of generated Scenario objects.

        Raises:
            GeneratorError: If generation fails.
        """
        if not skills:
            raise GeneratorError("No skills provided for scenario generation")

        if count < 1:
            raise GeneratorError("Count must be at least 1")

        # Build the prompt
        prompt = self._build_prompt(task, skills, count, include_adversarial)

        # Call Claude
        client = self._get_client()
        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
        except anthropic.AuthenticationError as e:
            raise GeneratorError(
                "Invalid Anthropic API key. Check ANTHROPIC_API_KEY environment variable.",
                task=task.description,
            ) from e
        except anthropic.RateLimitError as e:
            raise GeneratorError(
                "Anthropic API rate limit exceeded. Please try again later.",
                task=task.description,
            ) from e
        except anthropic.APIError as e:
            raise GeneratorError(
                f"Anthropic API error: {e}",
                task=task.description,
            ) from e

        # Extract text from response
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        if not response_text:
            raise GeneratorError(
                "Empty response from Claude",
                task=task.description,
            )

        # Parse the response
        try:
            scenario_dicts = self._parse_response(response_text, skills)
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(
                f"Failed to parse generated scenarios: {e}",
                task=task.description,
            ) from e

        # Convert to Scenario objects
        scenarios = [
            self._dict_to_scenario(data, i)
            for i, data in enumerate(scenario_dicts)
        ]

        # Shuffle to avoid ordering bias
        random.shuffle(scenarios)

        return scenarios


class MockGenerator(BaseGenerator):
    """Mock generator for testing purposes.

    Generates deterministic scenarios without calling any LLM.
    Useful for unit tests and development.
    """

    def __init__(self, scenarios_per_skill: int = 10):
        """Initialize the mock generator.

        Args:
            scenarios_per_skill: Number of scenarios to generate per skill.
        """
        self.scenarios_per_skill = scenarios_per_skill

    async def generate(
        self,
        task: Task,
        skills: list[Skill],
        count: int = 50,
        include_adversarial: bool = True,
    ) -> list[Scenario]:
        """Generate mock scenarios for testing.

        Args:
            task: The task description.
            skills: List of skills to generate scenarios for.
            count: Number of scenarios to generate.
            include_adversarial: Whether to include adversarial scenarios.

        Returns:
            List of generated Scenario objects.
        """
        scenarios = []
        difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]

        # Calculate scenarios per skill
        per_skill = max(1, count // len(skills)) if skills else 0
        remaining = count - (per_skill * len(skills))

        for skill in skills:
            for i in range(per_skill):
                # Determine difficulty based on index
                difficulty = difficulties[i % 3]

                # Determine if adversarial (last 20% if enabled)
                is_adversarial = include_adversarial and i >= per_skill * 0.8

                scenario = Scenario(
                    id=f"mock-{skill.name}-{i:03d}",
                    prompt=f"Test prompt #{i} for {skill.name}: {task.description}",
                    expected_skill=skill.name,
                    difficulty=difficulty,
                    tags=["mock", task.description.split()[0] if task.description else "test"],
                    is_adversarial=is_adversarial,
                )
                scenarios.append(scenario)

        # Add remaining scenarios to first skill if any
        if remaining > 0 and skills:
            first_skill = skills[0]
            for i in range(remaining):
                scenario = Scenario(
                    id=f"mock-extra-{i:03d}",
                    prompt=f"Extra test prompt #{i}: {task.description}",
                    expected_skill=first_skill.name,
                    difficulty=Difficulty.MEDIUM,
                    tags=["mock", "extra"],
                    is_adversarial=False,
                )
                scenarios.append(scenario)

        # Shuffle
        random.shuffle(scenarios)

        return scenarios
