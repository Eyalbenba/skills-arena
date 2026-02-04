"""Main parser interface with auto-detection.

This module provides the unified Parser class that automatically detects
the format of skill definitions and parses them appropriately.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from skills_arena.exceptions import SkillParseError
from skills_arena.models import Skill, SkillFormat
from skills_arena.parser.base import BaseParser
from skills_arena.parser.claude_code import ClaudeCodeParser
from skills_arena.parser.generic import GenericParser
from skills_arena.parser.mcp import MCPParser
from skills_arena.parser.openai import OpenAIParser

if TYPE_CHECKING:
    pass


class Parser:
    """Unified skill parser with auto-detection.

    This class provides a simple interface for parsing skill definitions
    from various formats. It automatically detects the format and uses
    the appropriate parser.

    Example:
        ```python
        from skills_arena import Parser

        # Auto-detect and parse
        skill = Parser.parse("./my-skill.md")

        # Parse with explicit format
        skill = Parser.parse_claude_code("./my-skill.md")

        # Parse from dict (OpenAI/MCP)
        skill = Parser.parse_openai({"name": "...", "description": "..."})
        ```
    """

    # Registered parsers (order matters for auto-detection)
    _parsers: list[BaseParser] = [
        MCPParser(),  # Check MCP first (has inputSchema)
        OpenAIParser(),  # Then OpenAI (JSON with name/description)
        ClaudeCodeParser(),  # Then Claude Code (markdown)
        GenericParser(),  # Fallback (always works)
    ]

    @classmethod
    def parse(
        cls,
        source: str | Path | dict[str, Any],
        format: SkillFormat | str | None = None,
    ) -> Skill:
        """Parse a skill definition from various sources.

        Args:
            source: The skill source. Can be:
                - A file path (str or Path)
                - A string containing the skill definition
                - A dictionary (for JSON-based formats)
            format: Optional explicit format. If not provided, auto-detects.
                Can be a SkillFormat enum or string ("claude_code", "openai", "mcp", "generic").

        Returns:
            A parsed Skill object.

        Raises:
            SkillParseError: If the skill cannot be parsed.
        """
        # Handle dictionary input
        if isinstance(source, dict):
            return cls._parse_dict(source, format)

        # Handle file path
        source_path = Path(source) if isinstance(source, (str, Path)) else None
        if source_path and source_path.exists():
            return cls._parse_file(source_path, format)

        # Handle string content
        if isinstance(source, str):
            return cls._parse_content(source, format=format)

        raise SkillParseError(
            f"Cannot parse skill from type: {type(source).__name__}",
            expected_format="file path, string content, or dictionary",
        )

    @classmethod
    def parse_file(cls, path: str | Path, format: SkillFormat | str | None = None) -> Skill:
        """Parse a skill from a file.

        Args:
            path: Path to the skill file.
            format: Optional explicit format.

        Returns:
            A parsed Skill object.
        """
        return cls._parse_file(Path(path), format)

    @classmethod
    def parse_content(
        cls,
        content: str,
        format: SkillFormat | str | None = None,
        source_path: str | None = None,
    ) -> Skill:
        """Parse a skill from string content.

        Args:
            content: The skill definition content.
            format: Optional explicit format.
            source_path: Optional source path for error messages.

        Returns:
            A parsed Skill object.
        """
        return cls._parse_content(content, format, source_path)

    @classmethod
    def parse_claude_code(cls, source: str | Path) -> Skill:
        """Parse a Claude Code skill (.md format).

        Args:
            source: File path or markdown content.

        Returns:
            A parsed Skill object.
        """
        return cls.parse(source, format=SkillFormat.CLAUDE_CODE)

    @classmethod
    def parse_openai(cls, source: str | Path | dict[str, Any]) -> Skill:
        """Parse an OpenAI function schema.

        Args:
            source: File path, JSON string, or dictionary.

        Returns:
            A parsed Skill object.
        """
        return cls.parse(source, format=SkillFormat.OPENAI)

    @classmethod
    def parse_mcp(cls, source: str | Path | dict[str, Any]) -> Skill:
        """Parse an MCP tool definition.

        Args:
            source: File path, JSON string, or dictionary.

        Returns:
            A parsed Skill object.
        """
        return cls.parse(source, format=SkillFormat.MCP)

    @classmethod
    def parse_generic(cls, source: str | Path) -> Skill:
        """Parse a generic/plain text skill description.

        Args:
            source: File path or text content.

        Returns:
            A parsed Skill object.
        """
        return cls.parse(source, format=SkillFormat.GENERIC)

    @classmethod
    def detect_format(cls, content: str) -> SkillFormat:
        """Detect the format of skill content.

        Args:
            content: The content to analyze.

        Returns:
            The detected SkillFormat.
        """
        for parser in cls._parsers:
            if parser.can_parse(content):
                return parser.format
        return SkillFormat.GENERIC

    @classmethod
    def _parse_file(cls, path: Path, format: SkillFormat | str | None = None) -> Skill:
        """Parse a skill from a file path."""
        if not path.exists():
            raise SkillParseError(f"File not found: {path}", path=str(path))

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise SkillParseError(f"Could not read file: {e}", path=str(path)) from e

        # Hint format from file extension
        if format is None:
            format = cls._format_from_extension(path)

        return cls._parse_content(content, format, source_path=str(path))

    @classmethod
    def _parse_content(
        cls,
        content: str,
        format: SkillFormat | str | None = None,
        source_path: str | None = None,
    ) -> Skill:
        """Parse skill from string content."""
        content = content.strip()
        if not content:
            raise SkillParseError("Empty skill content", path=source_path)

        # Normalize format
        skill_format = cls._normalize_format(format) if format else None

        # If format specified, use that parser
        if skill_format:
            parser = cls._get_parser(skill_format)
            return parser.parse(content, source_path)

        # Auto-detect format
        for parser in cls._parsers:
            if parser.can_parse(content):
                return parser.parse(content, source_path)

        # Should never reach here since GenericParser always works
        raise SkillParseError("Could not parse skill content", path=source_path)

    @classmethod
    def _parse_dict(
        cls, data: dict[str, Any], format: SkillFormat | str | None = None
    ) -> Skill:
        """Parse skill from a dictionary."""
        import json

        content = json.dumps(data)

        # Infer format from dict structure if not specified
        if format is None:
            if "inputSchema" in data:
                format = SkillFormat.MCP
            elif "name" in data or (data.get("type") == "function" and "function" in data):
                format = SkillFormat.OPENAI

        return cls._parse_content(content, format)

    @classmethod
    def _normalize_format(cls, format: SkillFormat | str) -> SkillFormat:
        """Normalize format to SkillFormat enum."""
        if isinstance(format, SkillFormat):
            return format
        if isinstance(format, str):
            try:
                return SkillFormat(format.lower())
            except ValueError as e:
                # Try mapping common aliases
                aliases = {
                    "claude": SkillFormat.CLAUDE_CODE,
                    "claude-code": SkillFormat.CLAUDE_CODE,
                    "claudecode": SkillFormat.CLAUDE_CODE,
                    "md": SkillFormat.CLAUDE_CODE,
                    "markdown": SkillFormat.CLAUDE_CODE,
                    "function": SkillFormat.OPENAI,
                    "tool": SkillFormat.MCP,
                    "text": SkillFormat.GENERIC,
                    "plain": SkillFormat.GENERIC,
                }
                if format.lower() in aliases:
                    return aliases[format.lower()]
                raise SkillParseError(
                    f"Unknown format: '{format}'",
                    expected_format="claude_code, openai, mcp, or generic",
                ) from e
        raise SkillParseError(f"Invalid format type: {type(format).__name__}")

    @classmethod
    def _format_from_extension(cls, path: Path) -> SkillFormat | None:
        """Infer format from file extension."""
        ext = path.suffix.lower()
        extension_map = {
            ".md": SkillFormat.CLAUDE_CODE,
            ".markdown": SkillFormat.CLAUDE_CODE,
            ".json": None,  # Could be OpenAI or MCP, need to check content
            ".txt": SkillFormat.GENERIC,
        }
        return extension_map.get(ext)

    @classmethod
    def _get_parser(cls, format: SkillFormat) -> BaseParser:
        """Get parser for a specific format."""
        for parser in cls._parsers:
            if parser.format == format:
                return parser
        raise SkillParseError(f"No parser for format: {format}")


# Convenience function for simple usage
def parse(source: str | Path | dict[str, Any], format: SkillFormat | str | None = None) -> Skill:
    """Parse a skill definition.

    This is a convenience function that delegates to Parser.parse().

    Args:
        source: The skill source (file path, content string, or dict).
        format: Optional explicit format.

    Returns:
        A parsed Skill object.
    """
    return Parser.parse(source, format)
