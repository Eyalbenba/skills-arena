"""Tests for Skills Arena configuration."""

import os
import tempfile

import pytest

from skills_arena import ArenaConfig, Config, SkillConfig, Task


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Config()
        assert config.scenarios == 50
        assert config.generator_model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.7
        assert config.include_adversarial is True
        assert config.agents == ["claude-code"]
        assert config.parallel_requests == 10
        assert config.timeout_seconds == 30
        assert config.elo_k_factor == 32
        assert config.verbose is False

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = Config(
            scenarios=100,
            temperature=0.5,
            agents=["claude-code", "mock"],
            verbose=True,
        )
        assert config.scenarios == 100
        assert config.temperature == 0.5
        assert len(config.agents) == 2
        assert config.verbose is True

    def test_invalid_agent(self) -> None:
        """Test that invalid agent names raise an error."""
        with pytest.raises(ValueError, match="Invalid agent"):
            Config(agents=["invalid-agent"])

    def test_valid_agents(self) -> None:
        """Test all valid agent names."""
        valid_agents = ["claude-code", "codex", "mock"]
        config = Config(agents=valid_agents)
        assert config.agents == valid_agents

    def test_scenario_bounds(self) -> None:
        """Test scenario count validation."""
        # Should work at boundaries
        config_min = Config(scenarios=1)
        assert config_min.scenarios == 1

        config_max = Config(scenarios=1000)
        assert config_max.scenarios == 1000

        # Should fail outside boundaries
        with pytest.raises(ValueError):
            Config(scenarios=0)

        with pytest.raises(ValueError):
            Config(scenarios=1001)

    def test_temperature_bounds(self) -> None:
        """Test temperature validation."""
        config_min = Config(temperature=0.0)
        assert config_min.temperature == 0.0

        config_max = Config(temperature=2.0)
        assert config_max.temperature == 2.0

        with pytest.raises(ValueError):
            Config(temperature=-0.1)

        with pytest.raises(ValueError):
            Config(temperature=2.1)

    def test_api_key_from_env(self) -> None:
        """Test API key loading from environment."""
        # Set env vars
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        os.environ["OPENAI_API_KEY"] = "test-openai-key"

        try:
            config = Config()
            assert config.anthropic_api_key == "test-anthropic-key"
            assert config.openai_api_key == "test-openai-key"
        finally:
            # Clean up
            del os.environ["ANTHROPIC_API_KEY"]
            del os.environ["OPENAI_API_KEY"]

    def test_api_key_explicit(self) -> None:
        """Test explicit API key takes precedence."""
        config = Config(
            anthropic_api_key="explicit-key",
        )
        assert config.anthropic_api_key == "explicit-key"

    def test_get_anthropic_api_key_missing(self) -> None:
        """Test error when Anthropic API key is missing."""
        # Ensure env var is not set
        os.environ.pop("ANTHROPIC_API_KEY", None)

        config = Config()
        with pytest.raises(ValueError, match="Anthropic API key not found"):
            config.get_anthropic_api_key()

    def test_get_openai_api_key_missing(self) -> None:
        """Test error when OpenAI API key is missing."""
        # Ensure env var is not set
        os.environ.pop("OPENAI_API_KEY", None)

        config = Config()
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            config.get_openai_api_key()


class TestSkillConfig:
    """Tests for SkillConfig class."""

    def test_create_simple(self) -> None:
        """Test creating a simple skill config."""
        config = SkillConfig(path="./my-skill.md")
        assert config.path == "./my-skill.md"
        assert config.name is None

    def test_create_with_name(self) -> None:
        """Test creating a skill config with custom name."""
        config = SkillConfig(path="./my-skill.md", name="custom-name")
        assert config.path == "./my-skill.md"
        assert config.name == "custom-name"


class TestArenaConfig:
    """Tests for ArenaConfig class."""

    def test_from_yaml_basic(self) -> None:
        """Test loading basic YAML config."""
        yaml_content = """
task: "web search"
skills:
  - ./skill-a.md
  - ./skill-b.md
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = ArenaConfig.from_yaml(f.name)
                assert config.task == "web search"
                assert len(config.skills) == 2
                assert config.skills[0] == "./skill-a.md"
            finally:
                os.unlink(f.name)

    def test_from_yaml_with_evaluation(self) -> None:
        """Test loading YAML config with evaluation settings."""
        yaml_content = """
task: "web search"
skills:
  - ./skill-a.md

evaluation:
  scenarios: 100
  agents:
    - claude-code
    - mock
  temperature: 0.5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = ArenaConfig.from_yaml(f.name)
                assert config.evaluation.scenarios == 100
                assert len(config.evaluation.agents) == 2
                assert config.evaluation.temperature == 0.5
            finally:
                os.unlink(f.name)

    def test_from_yaml_with_skill_configs(self) -> None:
        """Test loading YAML config with named skills."""
        yaml_content = """
task: "web search"
skills:
  - path: ./skill-a.md
    name: skill-a
  - path: ./skill-b.md
    name: skill-b
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = ArenaConfig.from_yaml(f.name)
                assert len(config.skills) == 2
                assert isinstance(config.skills[0], SkillConfig)
                assert config.skills[0].name == "skill-a"
            finally:
                os.unlink(f.name)

    def test_from_yaml_missing_file(self) -> None:
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ArenaConfig.from_yaml("/nonexistent/config.yaml")

    def test_get_task(self) -> None:
        """Test converting task to Task object."""
        config = ArenaConfig(task="web search", skills=["./skill.md"])
        task = config.get_task()
        assert isinstance(task, Task)
        assert task.description == "web search"

    def test_get_task_already_task(self) -> None:
        """Test get_task when task is already a Task object."""
        task = Task(description="web search", domains=["dev"])
        config = ArenaConfig(task=task, skills=["./skill.md"])
        result = config.get_task()
        assert result is task
        assert result.domains == ["dev"]

    def test_get_skill_paths(self) -> None:
        """Test getting skill paths."""
        config = ArenaConfig(
            task="test",
            skills=[
                "./skill-a.md",
                SkillConfig(path="./skill-b.md", name="custom"),
            ],
        )
        paths = config.get_skill_paths()
        assert len(paths) == 2
        assert paths[0] == ("./skill-a.md", None)
        assert paths[1] == ("./skill-b.md", "custom")
