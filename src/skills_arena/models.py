"""Core data models for Skills Arena.

This module defines the primary data structures used throughout the SDK:
- Skill: A skill definition that an agent can choose to use
- Task: A high-level description of what skills are meant to do
- Scenario: An auto-generated test prompt for skill selection
- Result types: Evaluation, Comparison, and Battle results
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SkillFormat(str, Enum):
    """Supported skill definition formats."""

    CLAUDE_CODE = "claude_code"
    OPENAI = "openai"
    MCP = "mcp"
    GENERIC = "generic"


class Difficulty(str, Enum):
    """Scenario difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Grade(str, Enum):
    """Letter grades for evaluation scores."""

    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    F = "F"

    @classmethod
    def from_score(cls, score: float) -> Grade:
        """Convert a numeric score (0-100) to a letter grade."""
        if score >= 97:
            return cls.A_PLUS
        elif score >= 93:
            return cls.A
        elif score >= 90:
            return cls.A_MINUS
        elif score >= 87:
            return cls.B_PLUS
        elif score >= 83:
            return cls.B
        elif score >= 80:
            return cls.B_MINUS
        elif score >= 77:
            return cls.C_PLUS
        elif score >= 73:
            return cls.C
        elif score >= 70:
            return cls.C_MINUS
        elif score >= 60:
            return cls.D
        else:
            return cls.F


class Parameter(BaseModel):
    """A parameter in a skill definition."""

    name: str
    description: str = ""
    type: str = "string"
    required: bool = False
    default: Any = None


class Skill(BaseModel):
    """A skill definition that an agent can choose to use.

    Parsed from various formats (Claude Code, OpenAI, MCP, generic).

    Attributes:
        name: The skill's identifier/name.
        description: The full skill description text.
        parameters: List of parameters the skill accepts.
        when_to_use: Examples of triggering prompts or use cases.
        source_format: The original format (claude_code, openai, mcp, generic).
        token_count: Approximate token count of the description.
        raw_content: Original file/definition content.
        source_path: Path to the source file (if applicable).
    """

    name: str
    description: str
    parameters: list[Parameter] = Field(default_factory=list)
    when_to_use: list[str] = Field(default_factory=list)
    source_format: SkillFormat = SkillFormat.GENERIC
    token_count: int = 0
    raw_content: str = ""
    source_path: str | None = None


class Task(BaseModel):
    """A high-level description of what skills are meant to do.

    Used to generate test scenarios.

    Attributes:
        description: The task description (e.g., "web search and content extraction").
        domains: Optional list of domains to test (e.g., ["enterprise", "developer"]).
        edge_cases: Optional specific edge cases to include.
    """

    description: str
    domains: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)

    @classmethod
    def from_string(cls, task_str: str) -> Task:
        """Create a Task from a simple string description."""
        return cls(description=task_str)


class Scenario(BaseModel):
    """An auto-generated test prompt that should trigger skill selection.

    Attributes:
        id: Unique identifier for the scenario.
        prompt: The user message/prompt.
        expected_skill: Which skill SHOULD be chosen for this scenario.
        difficulty: Difficulty level (easy, medium, hard).
        tags: Categorization tags.
        is_adversarial: Whether this is an adversarial/edge case scenario.
    """

    id: str
    prompt: str
    expected_skill: str
    difficulty: Difficulty = Difficulty.MEDIUM
    tags: list[str] = Field(default_factory=list)
    is_adversarial: bool = False


class SkillSelection(BaseModel):
    """The result of an agent selecting a skill for a scenario.

    Attributes:
        skill: The name of the selected skill (None if no skill selected).
        confidence: Agent's confidence in the selection (0.0-1.0).
        reasoning: Optional reasoning provided by the agent.
        raw_response: The raw response from the agent (for debugging).
    """

    skill: str | None = None
    confidence: float = 0.0
    reasoning: str = ""
    raw_response: Any = None


class SelectionResult(BaseModel):
    """The outcome of running a single scenario.

    Attributes:
        scenario: The scenario that was run.
        selection: The agent's skill selection.
        agent_name: Name of the agent that made the selection.
        is_correct: Whether the correct skill was selected.
        latency_ms: Time taken for the selection (milliseconds).
    """

    scenario: Scenario
    selection: SkillSelection
    agent_name: str
    is_correct: bool = False
    latency_ms: float = 0.0


class Insight(BaseModel):
    """An actionable insight about skill performance.

    Attributes:
        type: Category of insight (e.g., "optimization", "weakness", "comparison").
        message: The insight message.
        severity: Importance level ("info", "warning", "critical").
        suggestion: Optional suggested action.
    """

    type: str
    message: str
    severity: str = "info"
    suggestion: str | None = None


class AgentResult(BaseModel):
    """Results breakdown for a specific agent framework.

    Attributes:
        agent_name: Name of the agent framework.
        selection_rate: Percentage of times skill was selected (0.0-1.0).
        accuracy: Percentage of correct invocations (0.0-1.0).
        scenarios_run: Number of scenarios evaluated.
        avg_latency_ms: Average response latency in milliseconds.
    """

    agent_name: str
    selection_rate: float = 0.0
    accuracy: float = 0.0
    scenarios_run: int = 0
    avg_latency_ms: float = 0.0


class EvaluationResult(BaseModel):
    """The outcome of evaluating a single skill.

    Attributes:
        skill: Name of the evaluated skill.
        score: Overall score (0-100).
        grade: Letter grade (A+, A, B+, etc.).
        selection_rate: Percentage of times skill was selected (0.0-1.0).
        false_positive_rate: Rate of incorrect selections (0.0-1.0).
        invocation_accuracy: Accuracy of invocations when selected (0.0-1.0).
        per_agent: Breakdown by agent framework.
        insights: List of insights and suggestions.
        scenarios_run: Total number of scenarios evaluated.
    """

    skill: str
    score: float = 0.0
    grade: Grade = Grade.F
    selection_rate: float = 0.0
    false_positive_rate: float = 0.0
    invocation_accuracy: float = 0.0
    per_agent: dict[str, AgentResult] = Field(default_factory=dict)
    insights: list[Insight] = Field(default_factory=list)
    scenarios_run: int = 0


class ComparisonResult(BaseModel):
    """The outcome of comparing multiple skills.

    Attributes:
        winner: Name of the winning skill.
        selection_rates: Selection rate for each skill.
        head_to_head: Win counts (skill -> opponent -> wins).
        insights: List of comparative insights.
        per_agent: Breakdown by agent framework.
        scenarios_run: Total number of scenarios evaluated.
    """

    winner: str
    selection_rates: dict[str, float] = Field(default_factory=dict)
    head_to_head: dict[str, dict[str, int]] = Field(default_factory=dict)
    insights: list[Insight] = Field(default_factory=list)
    per_agent: dict[str, ComparisonResult] = Field(default_factory=dict)
    scenarios_run: int = 0


class RankedSkill(BaseModel):
    """A skill with its ranking information.

    Attributes:
        rank: Position in the leaderboard (1-indexed).
        name: Skill name.
        elo: ELO rating.
        wins: Total wins.
        losses: Total losses.
        selection_rate: Overall selection rate.
    """

    rank: int
    name: str
    elo: int = 1500
    wins: int = 0
    losses: int = 0
    selection_rate: float = 0.0


class Matchup(BaseModel):
    """A head-to-head matchup between two skills.

    Attributes:
        skill_a: First skill name.
        skill_b: Second skill name.
        winner: Name of the winner (or None for draw).
        scenario_id: The scenario where this matchup occurred.
    """

    skill_a: str
    skill_b: str
    winner: str | None = None
    scenario_id: str = ""


class BattleResult(BaseModel):
    """The outcome of a battle royale with multiple skills.

    Attributes:
        leaderboard: Ranked list of skills.
        elo_ratings: ELO rating for each skill.
        matchups: List of individual matchups.
        insights: List of insights about the competition.
        scenarios_run: Total number of scenarios evaluated.
    """

    leaderboard: list[RankedSkill] = Field(default_factory=list)
    elo_ratings: dict[str, int] = Field(default_factory=dict)
    matchups: list[Matchup] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    scenarios_run: int = 0


class Progress(BaseModel):
    """Progress information for long-running operations.

    Attributes:
        stage: Current stage name.
        percent: Completion percentage (0-100).
        message: Optional status message.
    """

    stage: str
    percent: float = 0.0
    message: str = ""
