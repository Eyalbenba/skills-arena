"""Tests for Skills Arena exceptions."""


from skills_arena import (
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


class TestSkillsArenaError:
    """Tests for base exception."""

    def test_is_exception(self) -> None:
        """Test that SkillsArenaError is an Exception."""
        error = SkillsArenaError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"


class TestSkillParseError:
    """Tests for SkillParseError."""

    def test_basic_message(self) -> None:
        """Test basic error message."""
        error = SkillParseError("Invalid syntax")
        assert "Invalid syntax" in str(error)
        assert "https://skills-arena.dev/docs/formats" in str(error)

    def test_with_path(self) -> None:
        """Test error with path."""
        error = SkillParseError("Invalid syntax", path="./skill.md")
        assert "./skill.md" in str(error)
        assert error.path == "./skill.md"

    def test_with_expected_format(self) -> None:
        """Test error with expected format."""
        error = SkillParseError(
            "Invalid syntax",
            expected_format="Claude Code (.md with description section)",
        )
        assert "Expected format:" in str(error)
        assert "Claude Code" in str(error)


class TestConfigError:
    """Tests for ConfigError."""

    def test_basic_message(self) -> None:
        """Test basic error message."""
        error = ConfigError("Invalid value")
        assert "Invalid value" in str(error)

    def test_with_field(self) -> None:
        """Test error with field name."""
        error = ConfigError("must be positive", field="scenarios")
        assert "scenarios" in str(error)
        assert error.field == "scenarios"


class TestAPIKeyError:
    """Tests for APIKeyError."""

    def test_message_format(self) -> None:
        """Test error message format."""
        error = APIKeyError("Anthropic", "ANTHROPIC_API_KEY")
        assert "Anthropic" in str(error)
        assert "ANTHROPIC_API_KEY" in str(error)
        assert "export" in str(error)
        assert error.provider == "Anthropic"
        assert error.env_var == "ANTHROPIC_API_KEY"


class TestAgentError:
    """Tests for AgentError."""

    def test_basic_message(self) -> None:
        """Test basic error message."""
        error = AgentError("Connection failed", agent_name="claude-code")
        assert "claude-code" in str(error)
        assert "Connection failed" in str(error)
        assert error.agent_name == "claude-code"

    def test_with_scenario_id(self) -> None:
        """Test error with scenario ID."""
        error = AgentError(
            "Timeout", agent_name="claude-code", scenario_id="test-001"
        )
        assert "test-001" in str(error)
        assert error.scenario_id == "test-001"


class TestGeneratorError:
    """Tests for GeneratorError."""

    def test_basic_message(self) -> None:
        """Test basic error message."""
        error = GeneratorError("Failed to generate scenarios")
        assert "generation failed" in str(error).lower()

    def test_with_task(self) -> None:
        """Test error with task."""
        error = GeneratorError("Invalid task", task="web search")
        assert "web search" in str(error)
        assert error.task == "web search"


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_message_format(self) -> None:
        """Test error message format."""
        error = TimeoutError(
            "Agent did not respond",
            operation="evaluate",
            timeout_seconds=30,
        )
        assert "30 seconds" in str(error)
        assert "evaluate" in str(error)
        assert "timeout_seconds" in str(error)
        assert error.operation == "evaluate"
        assert error.timeout_seconds == 30


class TestUnsupportedAgentError:
    """Tests for UnsupportedAgentError."""

    def test_message_format(self) -> None:
        """Test error message format."""
        error = UnsupportedAgentError(
            "invalid-agent",
            supported=["claude-code", "raw-openai"],
        )
        assert "invalid-agent" in str(error)
        assert "claude-code" in str(error)
        assert error.agent_name == "invalid-agent"
        assert "raw-openai" in error.supported


class TestCodexBridgeError:
    """Tests for CodexBridgeError."""

    def test_message_includes_setup(self) -> None:
        """Test error message includes setup instructions."""
        error = CodexBridgeError("Bridge not found")
        message = str(error)
        assert "Bridge not found" in message
        assert "Node.js" in message
        assert "@openai/codex" in message
        assert "skills-arena init-codex" in message


class TestNoSkillsError:
    """Tests for NoSkillsError."""

    def test_message_format(self) -> None:
        """Test error message format."""
        error = NoSkillsError("compare")
        assert "compare" in str(error)
        assert "at least one skill" in str(error).lower()
        assert error.operation == "compare"


class TestInsufficientScenariosError:
    """Tests for InsufficientScenariosError."""

    def test_message_format(self) -> None:
        """Test error message format."""
        error = InsufficientScenariosError(
            requested=50, generated=5, minimum=10
        )
        assert "50" in str(error)
        assert "5" in str(error)
        assert "10" in str(error)
        assert error.requested == 50
        assert error.generated == 5
        assert error.minimum == 10
