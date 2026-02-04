"""Skills Arena - Benchmark and optimize AI agent skill descriptions.

The SEO platform for AI agent skills. Benchmark, optimize, and compete
your skill descriptions across agent frameworks.

Example:
    ```python
    from skills_arena import Arena

    # Simple evaluation
    results = Arena().evaluate("./my-skill.md", task="web search")
    print(results.score)

    # Compare skills
    results = arena.compare(
        skills=["./my-skill.md", "./competitor.md"],
        task="web search",
    )
    print(results.winner)
    ```
"""

from .arena import Arena
from .config import ArenaConfig, Config, ProgressCallback, SkillConfig
from .exceptions import (
    AgentError,
    APIKeyError,
    CodexBridgeError,
    ConfigError,
    GeneratorError,
    InsufficientScenariosError,
    NoSkillsError,
    SkillParseError,
    SkillsArenaError,
    TimeoutError,
    UnsupportedAgentError,
)
from .parser import Parser, parse
from .generator import BaseGenerator, LLMGenerator, MockGenerator
from .scorer import ELO, RatingTracker, Scorer
from .models import (
    AgentResult,
    BattleResult,
    ComparisonResult,
    Difficulty,
    EvaluationResult,
    Grade,
    Insight,
    Matchup,
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

__version__ = "0.1.0"

__all__ = [
    # Main entry point
    "Arena",
    # Configuration
    "Config",
    "ArenaConfig",
    "SkillConfig",
    "ProgressCallback",
    # Core models
    "Skill",
    "SkillFormat",
    "Parameter",
    "Task",
    "Scenario",
    "Difficulty",
    "Grade",
    # Selection models
    "SkillSelection",
    "SelectionResult",
    # Result models
    "EvaluationResult",
    "ComparisonResult",
    "BattleResult",
    "AgentResult",
    "RankedSkill",
    "Matchup",
    "Insight",
    "Progress",
    # Parser
    "Parser",
    "parse",
    # Generators
    "BaseGenerator",
    "LLMGenerator",
    "MockGenerator",
    # Scorer
    "Scorer",
    "ELO",
    "RatingTracker",
    # Exceptions
    "SkillsArenaError",
    "SkillParseError",
    "ConfigError",
    "APIKeyError",
    "AgentError",
    "GeneratorError",
    "TimeoutError",
    "UnsupportedAgentError",
    "CodexBridgeError",
    "NoSkillsError",
    "InsufficientScenariosError",
]
