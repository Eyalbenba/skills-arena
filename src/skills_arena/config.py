"""Configuration for Skills Arena.

This module provides the Config class for customizing Arena behavior,
including scenario generation, agent selection, and execution parameters.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .models import Progress, Task


class SkillConfig(BaseModel):
    """Configuration for a single skill.

    Attributes:
        path: Path to the skill file.
        name: Optional custom name (defaults to filename).
    """

    path: str
    name: str | None = None


class Config(BaseModel):
    """Configuration for Skills Arena.

    Attributes:
        scenarios: Number of test scenarios to generate.
        generator_model: Model to use for scenario generation.
        temperature: Temperature for scenario generation (diversity).
        include_adversarial: Whether to include adversarial/edge case scenarios.
        agents: List of agent frameworks to test against.
        parallel_requests: Maximum concurrent requests.
        timeout_seconds: Timeout for each agent request.
        elo_k_factor: K-factor for ELO rating updates.
        verbose: Enable verbose output.
        anthropic_api_key: Anthropic API key (defaults to env var).
        openai_api_key: OpenAI API key (defaults to env var).
        codex_bridge_path: Path to TypeScript bridge script for Codex.
        node_executable: Path to Node.js executable.
    """

    # Scenario generation
    scenarios: int = Field(default=50, ge=1, le=1000)
    generator_model: str = "claude-sonnet-4-20250514"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    include_adversarial: bool = True

    # Scenario strategy for comparisons:
    # - "balanced": Generate scenarios for all skills together (default)
    # - "per_skill": Generate scenarios from each skill's description alone
    #                (competitive analysis - detects if competitors steal scenarios)
    scenario_strategy: str = Field(default="balanced")

    # Agent frameworks to test against
    # Options: "claude-code", "codex", "raw-claude", "raw-openai", "mock"
    agents: list[str] = Field(default_factory=lambda: ["claude-code"])

    # Execution
    parallel_requests: int = Field(default=10, ge=1, le=100)
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    # Scoring
    elo_k_factor: int = Field(default=32, ge=1, le=100)

    # Output
    verbose: bool = False

    # API Keys (defaults to env vars)
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Codex bridge settings (if using codex agent)
    codex_bridge_path: str | None = None
    node_executable: str = "node"

    @field_validator("agents")
    @classmethod
    def validate_agents(cls, v: list[str]) -> list[str]:
        """Validate agent names are supported."""
        valid_agents = {"claude-code", "codex", "raw-claude", "raw-openai", "mock"}
        for agent in v:
            if agent not in valid_agents:
                raise ValueError(
                    f"Invalid agent '{agent}'. Must be one of: {', '.join(sorted(valid_agents))}"
                )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Load API keys from environment if not provided."""
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")

    def get_anthropic_api_key(self) -> str:
        """Get Anthropic API key, raising if not available."""
        if not self.anthropic_api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass anthropic_api_key to Config."
            )
        return self.anthropic_api_key

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key, raising if not available."""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key to Config."
            )
        return self.openai_api_key


class ArenaConfig(BaseModel):
    """Full Arena configuration, typically loaded from YAML.

    Attributes:
        task: The task description (string or Task object).
        skills: List of skill paths or configurations.
        evaluation: Evaluation settings (maps to Config).
        output: Output settings.
    """

    task: str | Task
    skills: list[str | SkillConfig] = Field(default_factory=list)
    evaluation: Config = Field(default_factory=Config)
    output: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArenaConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            ArenaConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the YAML is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid config file: expected dict, got {type(data).__name__}")

        # Handle nested 'evaluation' section
        if "evaluation" in data and isinstance(data["evaluation"], dict):
            # Map YAML keys to Config attributes
            eval_data = data["evaluation"]
            data["evaluation"] = Config(**eval_data)

        # Handle skills list - can be strings or dicts
        if "skills" in data:
            skills = []
            for skill in data["skills"]:
                if isinstance(skill, str):
                    skills.append(skill)
                elif isinstance(skill, dict):
                    skills.append(SkillConfig(**skill))
                else:
                    raise ValueError(f"Invalid skill entry: {skill}")
            data["skills"] = skills

        return cls(**data)

    def get_task(self) -> Task:
        """Get the task as a Task object."""
        if isinstance(self.task, str):
            return Task.from_string(self.task)
        return self.task

    def get_skill_paths(self) -> list[tuple[str, str | None]]:
        """Get skill paths and optional names.

        Returns:
            List of (path, name) tuples. Name may be None.
        """
        result = []
        for skill in self.skills:
            if isinstance(skill, str):
                result.append((skill, None))
            else:
                result.append((skill.path, skill.name))
        return result


# Type alias for progress callback
ProgressCallback = Callable[[Progress], None]
