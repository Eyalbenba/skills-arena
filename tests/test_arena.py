"""Tests for Skills Arena main class."""

import os
import tempfile

import pytest

from skills_arena import Arena, Config, NoSkillsError, Skill


class TestArena:
    """Tests for Arena class."""

    def test_create_default(self) -> None:
        """Test creating Arena with default config."""
        arena = Arena()
        assert arena.config is not None
        assert arena.config.scenarios == 50

    def test_create_with_config(self) -> None:
        """Test creating Arena with custom config."""
        config = Config(scenarios=100, verbose=True)
        arena = Arena(config=config)
        assert arena.config.scenarios == 100
        assert arena.config.verbose is True

    def test_from_config_yaml(self) -> None:
        """Test creating Arena from YAML config."""
        yaml_content = """
task: "web search"
skills:
  - ./skill-a.md

evaluation:
  scenarios: 75
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                arena = Arena.from_config(f.name)
                assert arena.config.scenarios == 75
            finally:
                os.unlink(f.name)

    def test_evaluate_not_implemented(self) -> None:
        """Test that evaluate raises NotImplementedError."""
        arena = Arena()
        skill = Skill(name="test", description="Test skill")

        with pytest.raises(NotImplementedError, match="Phase 1"):
            arena.evaluate(skill, task="test task")

    def test_compare_not_implemented(self) -> None:
        """Test that compare raises NotImplementedError."""
        arena = Arena()
        skills = [
            Skill(name="test-a", description="Test A"),
            Skill(name="test-b", description="Test B"),
        ]

        with pytest.raises(NotImplementedError, match="Phase 1"):
            arena.compare(skills, task="test task")

    def test_compare_requires_two_skills(self) -> None:
        """Test that compare requires at least 2 skills."""
        arena = Arena()
        skill = Skill(name="test", description="Test skill")

        with pytest.raises(NoSkillsError, match="at least 2 skills"):
            arena.compare([skill], task="test task")

    def test_battle_royale_not_implemented(self) -> None:
        """Test that battle_royale raises NotImplementedError."""
        arena = Arena()
        skills = [
            Skill(name="test-a", description="Test A"),
            Skill(name="test-b", description="Test B"),
        ]

        with pytest.raises(NotImplementedError, match="Phase 3"):
            arena.battle_royale(skills, task="test task")

    def test_battle_royale_requires_two_skills(self) -> None:
        """Test that battle_royale requires at least 2 skills."""
        arena = Arena()
        skill = Skill(name="test", description="Test skill")

        with pytest.raises(NoSkillsError, match="at least 2 skills"):
            arena.battle_royale([skill], task="test task")

    def test_insights_not_implemented(self) -> None:
        """Test that insights raises NotImplementedError."""
        arena = Arena()
        skill = Skill(name="test", description="Test skill")

        with pytest.raises(NotImplementedError, match="Phase 4"):
            arena.insights(skill)

    def test_run_not_implemented(self) -> None:
        """Test that run raises NotImplementedError."""
        arena = Arena()

        with pytest.raises(NotImplementedError, match="Phase 1"):
            arena.run()


class TestArenaAsync:
    """Tests for async Arena methods."""

    @pytest.mark.asyncio
    async def test_evaluate_async_not_implemented(self) -> None:
        """Test that evaluate_async raises NotImplementedError."""
        arena = Arena()
        skill = Skill(name="test", description="Test skill")

        with pytest.raises(NotImplementedError, match="Phase 1"):
            await arena.evaluate_async(skill, task="test task")

    @pytest.mark.asyncio
    async def test_compare_async_not_implemented(self) -> None:
        """Test that compare_async raises NotImplementedError."""
        arena = Arena()
        skills = [
            Skill(name="test-a", description="Test A"),
            Skill(name="test-b", description="Test B"),
        ]

        with pytest.raises(NotImplementedError, match="Phase 1"):
            await arena.compare_async(skills, task="test task")

    @pytest.mark.asyncio
    async def test_battle_royale_async_not_implemented(self) -> None:
        """Test that battle_royale_async raises NotImplementedError."""
        arena = Arena()
        skills = [
            Skill(name="test-a", description="Test A"),
            Skill(name="test-b", description="Test B"),
        ]

        with pytest.raises(NotImplementedError, match="Phase 3"):
            await arena.battle_royale_async(skills, task="test task")
