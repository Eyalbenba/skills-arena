"""Base parser interface for skill definitions.

This module defines the abstract base class that all skill parsers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skills_arena.models import Skill, SkillFormat


class BaseParser(ABC):
    """Abstract base class for skill parsers.

    All skill format parsers (Claude Code, OpenAI, MCP, Generic) must inherit
    from this class and implement the required methods.
    """

    @property
    @abstractmethod
    def format(self) -> SkillFormat:
        """Return the skill format this parser handles."""
        ...

    @abstractmethod
    def parse(self, content: str, source_path: str | None = None) -> Skill:
        """Parse content into a Skill object.

        Args:
            content: The raw content to parse.
            source_path: Optional path to the source file.

        Returns:
            A Skill object representing the parsed definition.

        Raises:
            SkillParseError: If the content cannot be parsed.
        """
        ...

    @abstractmethod
    def can_parse(self, content: str) -> bool:
        """Check if this parser can handle the given content.

        Args:
            content: The content to check.

        Returns:
            True if this parser can handle the content, False otherwise.
        """
        ...

    def parse_file(self, path: str | Path) -> Skill:
        """Parse a skill from a file path.

        Args:
            path: Path to the skill file.

        Returns:
            A Skill object representing the parsed definition.

        Raises:
            SkillParseError: If the file cannot be read or parsed.
        """
        from skills_arena.exceptions import SkillParseError

        path = Path(path)
        if not path.exists():
            raise SkillParseError(
                f"File not found: {path}",
                path=str(path),
            )

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise SkillParseError(
                f"Could not read file: {e}",
                path=str(path),
            ) from e

        return self.parse(content, source_path=str(path))


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple heuristic: ~4 characters per token for English text.
    This is a rough approximation suitable for skill comparison.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    # Simple heuristic: ~4 characters per token
    # More accurate would be to use tiktoken, but this is good enough
    # for comparing skill description lengths
    return max(1, len(text) // 4)
