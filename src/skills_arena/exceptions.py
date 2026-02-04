"""Custom exceptions for Skills Arena.

This module provides clear, actionable error messages following the SDK's
design principle of developer-friendly feedback.
"""

from __future__ import annotations


class SkillsArenaError(Exception):
    """Base exception for all Skills Arena errors."""

    pass


class SkillParseError(SkillsArenaError):
    """Error parsing a skill definition.

    Raised when a skill file cannot be parsed or is in an unrecognized format.
    """

    def __init__(
        self,
        message: str,
        path: str | None = None,
        expected_format: str | None = None,
    ):
        self.path = path
        self.expected_format = expected_format

        full_message = message
        if path:
            full_message = f"Could not parse skill at '{path}'.\n{message}"
        if expected_format:
            full_message += f"\nExpected format: {expected_format}"
        full_message += "\nSee: https://skills-arena.dev/docs/formats"

        super().__init__(full_message)


class ConfigError(SkillsArenaError):
    """Error in configuration.

    Raised when configuration is invalid or missing required fields.
    """

    def __init__(self, message: str, field: str | None = None):
        self.field = field
        full_message = message
        if field:
            full_message = f"Configuration error in '{field}': {message}"
        super().__init__(full_message)


class APIKeyError(SkillsArenaError):
    """Missing or invalid API key.

    Raised when a required API key is not available.
    """

    def __init__(self, provider: str, env_var: str):
        self.provider = provider
        self.env_var = env_var
        message = (
            f"{provider} API key not found.\n"
            f"Set the {env_var} environment variable or pass it to Config.\n"
            f"Example: export {env_var}=sk-..."
        )
        super().__init__(message)


class AgentError(SkillsArenaError):
    """Error from an agent framework.

    Raised when an agent fails to process a request.
    """

    def __init__(
        self,
        message: str,
        agent_name: str,
        scenario_id: str | None = None,
    ):
        self.agent_name = agent_name
        self.scenario_id = scenario_id

        full_message = f"Agent '{agent_name}' error: {message}"
        if scenario_id:
            full_message += f"\nScenario ID: {scenario_id}"

        super().__init__(full_message)


class GeneratorError(SkillsArenaError):
    """Error generating scenarios.

    Raised when scenario generation fails.
    """

    def __init__(self, message: str, task: str | None = None):
        self.task = task
        full_message = f"Scenario generation failed: {message}"
        if task:
            full_message += f"\nTask: {task}"
        super().__init__(full_message)


class TimeoutError(SkillsArenaError):
    """Operation timed out.

    Raised when an operation exceeds the configured timeout.
    """

    def __init__(
        self,
        message: str,
        operation: str,
        timeout_seconds: int,
    ):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        full_message = (
            f"Operation '{operation}' timed out after {timeout_seconds} seconds.\n"
            f"{message}\n"
            f"Try increasing timeout_seconds in Config."
        )
        super().__init__(full_message)


class UnsupportedAgentError(SkillsArenaError):
    """Unsupported agent framework requested.

    Raised when an unknown agent type is specified.
    """

    def __init__(self, agent_name: str, supported: list[str]):
        self.agent_name = agent_name
        self.supported = supported
        message = (
            f"Unsupported agent '{agent_name}'.\n"
            f"Supported agents: {', '.join(sorted(supported))}"
        )
        super().__init__(message)


class CodexBridgeError(SkillsArenaError):
    """Error with the Codex TypeScript bridge.

    Raised when the Codex bridge fails or is not properly configured.
    """

    def __init__(self, message: str):
        full_message = (
            f"Codex bridge error: {message}\n"
            "Codex requires Node.js and @openai/codex installed.\n"
            "Run: npm install -g @openai/codex typescript ts-node\n"
            "Then: skills-arena init-codex"
        )
        super().__init__(full_message)


class NoSkillsError(SkillsArenaError):
    """No skills provided for evaluation.

    Raised when an operation requires skills but none were provided.
    """

    def __init__(self, operation: str):
        self.operation = operation
        message = (
            f"No skills provided for '{operation}'.\n"
            f"Provide at least one skill path or definition."
        )
        super().__init__(message)


class InsufficientScenariosError(SkillsArenaError):
    """Not enough scenarios for evaluation.

    Raised when scenario generation produces too few scenarios.
    """

    def __init__(self, requested: int, generated: int, minimum: int = 1):
        self.requested = requested
        self.generated = generated
        self.minimum = minimum
        message = (
            f"Generated {generated} scenarios, but at least {minimum} required "
            f"(requested {requested}).\n"
            "Try a more specific task description or adjust generator settings."
        )
        super().__init__(message)
