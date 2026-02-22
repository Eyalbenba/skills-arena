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
    OptimizerError,
    SkillParseError,
    SkillsArenaError,
    TimeoutError,
    UnsupportedAgentError,
)
from .parser import Parser, parse
from .generator import BaseGenerator, LLMGenerator, MockGenerator
from .optimizer import SkillOptimizer
from .reporter import TextReporter, print_results
from .scorer import ELO, RatingTracker, Scorer
from .models import (
    AgentResult,
    BattleResult,
    ComparisonResult,
    CustomScenario,
    Difficulty,
    EvaluationResult,
    GenerateScenarios,
    Grade,
    Insight,
    Matchup,
    OptimizationIteration,
    OptimizationResult,
    Parameter,
    Progress,
    RankedSkill,
    Scenario,
    ScenarioDetail,
    SelectionResult,
    Skill,
    SkillFormat,
    SkillSelection,
    Task,
)

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"  # fallback for editable installs without build

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
    "CustomScenario",
    "GenerateScenarios",
    "ScenarioDetail",
    "Difficulty",
    "Grade",
    # Selection models
    "SkillSelection",
    "SelectionResult",
    # Result models
    "EvaluationResult",
    "ComparisonResult",
    "BattleResult",
    "OptimizationResult",
    "OptimizationIteration",
    "AgentResult",
    "RankedSkill",
    "Matchup",
    "Insight",
    "Progress",
    # Optimizer
    "SkillOptimizer",
    # Parser
    "Parser",
    "parse",
    # Generators
    "BaseGenerator",
    "LLMGenerator",
    "MockGenerator",
    # Reporter
    "TextReporter",
    "print_results",
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
    "OptimizerError",
]
