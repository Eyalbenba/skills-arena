"""Skill parsing module.

Provides parsers for various skill definition formats:
- Claude Code (.md files)
- OpenAI function schemas
- MCP tool definitions
- Generic text descriptions

Example:
    ```python
    from skills_arena.parser import Parser, parse

    # Auto-detect and parse
    skill = Parser.parse("./my-skill.md")

    # Or use the convenience function
    skill = parse("./my-skill.md")

    # Parse with explicit format
    skill = Parser.parse_claude_code("./my-skill.md")
    skill = Parser.parse_openai({"name": "...", "description": "..."})
    skill = Parser.parse_mcp({"name": "...", "inputSchema": {...}})
    ```
"""

from skills_arena.parser.base import BaseParser, estimate_tokens
from skills_arena.parser.claude_code import ClaudeCodeParser
from skills_arena.parser.generic import GenericParser
from skills_arena.parser.mcp import MCPParser
from skills_arena.parser.openai import OpenAIParser
from skills_arena.parser.parser import Parser, parse

__all__ = [
    # Main interface
    "Parser",
    "parse",
    # Base class
    "BaseParser",
    # Individual parsers
    "ClaudeCodeParser",
    "OpenAIParser",
    "MCPParser",
    "GenericParser",
    # Utilities
    "estimate_tokens",
]
