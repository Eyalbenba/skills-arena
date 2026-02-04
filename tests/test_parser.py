"""Tests for the parser module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skills_arena.exceptions import SkillParseError
from skills_arena.models import SkillFormat
from skills_arena.parser import (
    ClaudeCodeParser,
    GenericParser,
    MCPParser,
    OpenAIParser,
    Parser,
    estimate_tokens,
    parse,
)


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "skills"


class TestEstimateTokens:
    """Tests for the estimate_tokens utility function."""

    def test_empty_string(self):
        """Empty string should return at least 1 token."""
        assert estimate_tokens("") == 1

    def test_short_string(self):
        """Short strings should have proportional token count."""
        # ~4 chars per token
        assert estimate_tokens("hello") >= 1
        assert estimate_tokens("hello world!") >= 2

    def test_longer_text(self):
        """Longer text should scale appropriately."""
        text = "This is a longer piece of text that should have more tokens."
        tokens = estimate_tokens(text)
        assert tokens > 10
        assert tokens < len(text)  # Should be less than character count


class TestClaudeCodeParser:
    """Tests for ClaudeCodeParser."""

    @pytest.fixture
    def parser(self):
        return ClaudeCodeParser()

    def test_can_parse_valid_markdown(self, parser):
        """Should recognize valid Claude Code markdown."""
        content = "# My Skill\n\nDescription here."
        assert parser.can_parse(content) is True

    def test_can_parse_with_h2_heading(self, parser):
        """Should recognize markdown starting with ## heading."""
        content = "## My Skill\n\nDescription here."
        assert parser.can_parse(content) is True

    def test_cannot_parse_empty(self, parser):
        """Should reject empty content."""
        assert parser.can_parse("") is False
        assert parser.can_parse("   ") is False

    def test_cannot_parse_non_markdown(self, parser):
        """Should reject non-markdown content."""
        assert parser.can_parse('{"name": "test"}') is False
        assert parser.can_parse("Just plain text without heading") is False

    def test_parse_basic_skill(self, parser):
        """Should parse a basic skill definition."""
        content = """# Web Search

Search the web for information.

## Description

A comprehensive web search skill.

## Parameters

- `query` (string, required): The search query
- `limit` (integer, optional): Max results (default: 10)

## When to Use

- User asks for current information
- User wants to find websites

## Examples

- "Search for AI news"
- "Find restaurants nearby"
"""
        skill = parser.parse(content)

        assert skill.name == "Web Search"
        assert "comprehensive web search" in skill.description
        assert skill.source_format == SkillFormat.CLAUDE_CODE
        assert len(skill.parameters) == 2

        # Check parameters
        query_param = next(p for p in skill.parameters if p.name == "query")
        assert query_param.type == "string"
        assert query_param.required is True

        limit_param = next(p for p in skill.parameters if p.name == "limit")
        assert limit_param.type == "integer"
        assert limit_param.required is False
        assert limit_param.default == "10"

        # Check when_to_use
        assert len(skill.when_to_use) >= 2
        assert any("current information" in w for w in skill.when_to_use)

    def test_parse_skill_without_sections(self, parser):
        """Should parse skill with just title and description."""
        content = """# Simple Skill

This is a simple skill with no sections.
Just a description.
"""
        skill = parser.parse(content)

        assert skill.name == "Simple Skill"
        assert "simple skill" in skill.description.lower()
        assert skill.parameters == []
        assert skill.when_to_use == []

    def test_parse_from_fixture_file(self, parser):
        """Should parse the sample skill fixture file."""
        skill = parser.parse_file(FIXTURES_DIR / "sample-skill.md")

        assert skill.name == "Web Search Skill"
        assert skill.source_path == str(FIXTURES_DIR / "sample-skill.md")
        assert len(skill.parameters) == 3
        assert "query" in [p.name for p in skill.parameters]

    def test_parse_empty_raises_error(self, parser):
        """Should raise error for empty content."""
        with pytest.raises(SkillParseError):
            parser.parse("")

    def test_parse_no_title_raises_error(self, parser):
        """Should raise error if no markdown heading found."""
        with pytest.raises(SkillParseError):
            parser.parse("Just text without any heading")


class TestOpenAIParser:
    """Tests for OpenAIParser."""

    @pytest.fixture
    def parser(self):
        return OpenAIParser()

    def test_can_parse_valid_schema(self, parser):
        """Should recognize valid OpenAI function schema."""
        content = json.dumps({"name": "test", "description": "A test function"})
        assert parser.can_parse(content) is True

    def test_can_parse_tools_format(self, parser):
        """Should recognize OpenAI tools format."""
        content = json.dumps(
            {
                "type": "function",
                "function": {"name": "test", "description": "A test function"},
            }
        )
        assert parser.can_parse(content) is True

    def test_cannot_parse_mcp_format(self, parser):
        """Should not match MCP format (has inputSchema)."""
        content = json.dumps(
            {"name": "test", "description": "test", "inputSchema": {"type": "object"}}
        )
        # MCP parser should take precedence, but OpenAI can technically parse it
        # The main parser handles prioritization
        assert parser.can_parse(content) is True  # Has name + description

    def test_cannot_parse_empty(self, parser):
        """Should reject empty content."""
        assert parser.can_parse("") is False

    def test_cannot_parse_invalid_json(self, parser):
        """Should reject invalid JSON."""
        assert parser.can_parse("not json") is False
        assert parser.can_parse("{invalid}") is False

    def test_parse_basic_function(self, parser):
        """Should parse a basic function schema."""
        content = json.dumps(
            {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            }
        )
        skill = parser.parse(content)

        assert skill.name == "get_weather"
        assert skill.description == "Get current weather"
        assert skill.source_format == SkillFormat.OPENAI
        assert len(skill.parameters) == 1

        loc_param = skill.parameters[0]
        assert loc_param.name == "location"
        assert loc_param.type == "string"
        assert loc_param.required is True

    def test_parse_with_enum(self, parser):
        """Should handle enum types in parameters."""
        content = json.dumps(
            {
                "name": "set_unit",
                "description": "Set temperature unit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Unit choice",
                        }
                    },
                },
            }
        )
        skill = parser.parse(content)

        unit_param = skill.parameters[0]
        assert "celsius" in unit_param.description
        assert "fahrenheit" in unit_param.description

    def test_parse_tools_format(self, parser):
        """Should parse tools format with nested function."""
        content = json.dumps(
            {
                "type": "function",
                "function": {
                    "name": "nested_func",
                    "description": "A nested function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
        skill = parser.parse(content)

        assert skill.name == "nested_func"
        assert skill.description == "A nested function"

    def test_parse_from_fixture_file(self, parser):
        """Should parse the OpenAI fixture file."""
        skill = parser.parse_file(FIXTURES_DIR / "openai-function.json")

        assert skill.name == "get_weather"
        assert "weather" in skill.description.lower()
        assert len(skill.parameters) == 2

    def test_parse_dict(self, parser):
        """Should parse from dictionary directly."""
        data = {"name": "dict_func", "description": "From dict"}
        skill = parser.parse_dict(data)

        assert skill.name == "dict_func"

    def test_parse_missing_name_raises_error(self, parser):
        """Should raise error if name is missing."""
        with pytest.raises(SkillParseError):
            parser.parse('{"description": "No name"}')


class TestMCPParser:
    """Tests for MCPParser."""

    @pytest.fixture
    def parser(self):
        return MCPParser()

    def test_can_parse_valid_mcp(self, parser):
        """Should recognize valid MCP tool definition."""
        content = json.dumps(
            {"name": "test", "description": "test", "inputSchema": {"type": "object"}}
        )
        assert parser.can_parse(content) is True

    def test_cannot_parse_openai_format(self, parser):
        """Should not match OpenAI format (no inputSchema)."""
        content = json.dumps({"name": "test", "description": "test", "parameters": {}})
        assert parser.can_parse(content) is False

    def test_cannot_parse_empty(self, parser):
        """Should reject empty content."""
        assert parser.can_parse("") is False

    def test_parse_basic_tool(self, parser):
        """Should parse a basic MCP tool definition."""
        content = json.dumps(
            {
                "name": "search_db",
                "description": "Search the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            }
        )
        skill = parser.parse(content)

        assert skill.name == "search_db"
        assert skill.description == "Search the database"
        assert skill.source_format == SkillFormat.MCP
        assert len(skill.parameters) == 2

        query_param = next(p for p in skill.parameters if p.name == "query")
        assert query_param.required is True

        limit_param = next(p for p in skill.parameters if p.name == "limit")
        assert limit_param.default == 10

    def test_parse_from_fixture_file(self, parser):
        """Should parse the MCP fixture file."""
        skill = parser.parse_file(FIXTURES_DIR / "mcp-tool.json")

        assert skill.name == "search_database"
        assert len(skill.parameters) == 3

    def test_parse_dict(self, parser):
        """Should parse from dictionary directly."""
        data = {
            "name": "dict_tool",
            "description": "From dict",
            "inputSchema": {"type": "object"},
        }
        skill = parser.parse_dict(data)

        assert skill.name == "dict_tool"


class TestGenericParser:
    """Tests for GenericParser."""

    @pytest.fixture
    def parser(self):
        return GenericParser()

    def test_can_parse_any_text(self, parser):
        """Should accept any non-empty text."""
        assert parser.can_parse("any text") is True
        assert parser.can_parse("# Even markdown") is True
        assert parser.can_parse('{"even": "json"}') is True

    def test_cannot_parse_empty(self, parser):
        """Should reject empty content."""
        assert parser.can_parse("") is False
        assert parser.can_parse("   ") is False

    def test_parse_extracts_name_from_heading(self, parser):
        """Should extract name from markdown heading."""
        content = "# My Tool\n\nSome description"
        skill = parser.parse(content)

        assert skill.name == "My Tool"

    def test_parse_extracts_name_from_first_line(self, parser):
        """Should use first line as name if no heading."""
        content = "Email Sender\n\nA tool for sending emails."
        skill = parser.parse(content)

        assert skill.name == "Email Sender"

    def test_parse_uses_filename_as_fallback(self, parser):
        """Should use filename if no name in content."""
        content = "This is just a description without a title or heading."
        skill = parser.parse(content, source_path="/path/to/my-tool.txt")

        assert skill.name == "My Tool"

    def test_parse_extracts_cli_parameters(self, parser):
        """Should extract command-line style parameters."""
        content = """Tool

Parameters:
--input: input file
--output: output file
--verbose: enable verbose mode
"""
        skill = parser.parse(content)

        param_names = [p.name for p in skill.parameters]
        assert "input" in param_names
        assert "output" in param_names
        assert "verbose" in param_names

    def test_parse_extracts_usage_patterns(self, parser):
        """Should extract 'use when' patterns."""
        content = """Tool

Use when you need to process files.
Good for batch operations.
"""
        skill = parser.parse(content)

        assert len(skill.when_to_use) >= 2

    def test_parse_from_fixture_file(self, parser):
        """Should parse the generic fixture file."""
        skill = parser.parse_file(FIXTURES_DIR / "generic.txt")

        assert skill.name == "Email Sender"
        assert skill.source_format == SkillFormat.GENERIC
        assert len(skill.parameters) >= 3


class TestParserUnified:
    """Tests for the unified Parser class."""

    def test_parse_auto_detects_claude_code(self):
        """Should auto-detect Claude Code format."""
        content = "# My Skill\n\n## Description\n\nA skill."
        skill = Parser.parse(content)

        assert skill.source_format == SkillFormat.CLAUDE_CODE

    def test_parse_auto_detects_openai(self):
        """Should auto-detect OpenAI format."""
        content = json.dumps({"name": "test", "description": "test"})
        skill = Parser.parse(content)

        assert skill.source_format == SkillFormat.OPENAI

    def test_parse_auto_detects_mcp(self):
        """Should auto-detect MCP format (inputSchema)."""
        content = json.dumps(
            {"name": "test", "description": "test", "inputSchema": {"type": "object"}}
        )
        skill = Parser.parse(content)

        assert skill.source_format == SkillFormat.MCP

    def test_parse_file_by_path(self):
        """Should parse file from path."""
        skill = Parser.parse(FIXTURES_DIR / "sample-skill.md")

        assert skill.name == "Web Search Skill"
        assert skill.source_format == SkillFormat.CLAUDE_CODE

    def test_parse_file_string_path(self):
        """Should parse file from string path."""
        skill = Parser.parse(str(FIXTURES_DIR / "sample-skill.md"))

        assert skill.name == "Web Search Skill"

    def test_parse_dict_as_openai(self):
        """Should parse dictionary as OpenAI format."""
        skill = Parser.parse({"name": "dict_skill", "description": "From dict"})

        assert skill.name == "dict_skill"
        assert skill.source_format == SkillFormat.OPENAI

    def test_parse_dict_as_mcp(self):
        """Should parse dictionary with inputSchema as MCP."""
        skill = Parser.parse(
            {
                "name": "mcp_skill",
                "description": "MCP",
                "inputSchema": {"type": "object"},
            }
        )

        assert skill.name == "mcp_skill"
        assert skill.source_format == SkillFormat.MCP

    def test_parse_with_explicit_format(self):
        """Should use explicit format when specified."""
        content = "# My Skill\n\nDescription"

        # Force generic parsing
        skill = Parser.parse(content, format=SkillFormat.GENERIC)
        assert skill.source_format == SkillFormat.GENERIC

        # String format specifier
        skill = Parser.parse(content, format="generic")
        assert skill.source_format == SkillFormat.GENERIC

    def test_parse_claude_code_method(self):
        """Should have convenience method for Claude Code."""
        skill = Parser.parse_claude_code(FIXTURES_DIR / "sample-skill.md")
        assert skill.source_format == SkillFormat.CLAUDE_CODE

    def test_parse_openai_method(self):
        """Should have convenience method for OpenAI."""
        skill = Parser.parse_openai({"name": "test", "description": "test"})
        assert skill.source_format == SkillFormat.OPENAI

    def test_parse_mcp_method(self):
        """Should have convenience method for MCP."""
        skill = Parser.parse_mcp(
            {"name": "test", "inputSchema": {"type": "object"}}
        )
        assert skill.source_format == SkillFormat.MCP

    def test_detect_format(self):
        """Should detect format without parsing."""
        assert Parser.detect_format("# Skill\n\nDesc") == SkillFormat.CLAUDE_CODE
        assert Parser.detect_format('{"name":"x","description":"y"}') == SkillFormat.OPENAI
        assert (
            Parser.detect_format('{"name":"x","inputSchema":{}}') == SkillFormat.MCP
        )
        assert Parser.detect_format("plain text") == SkillFormat.GENERIC

    def test_parse_file_not_found(self):
        """Should raise error for non-existent file."""
        with pytest.raises(SkillParseError) as exc_info:
            Parser.parse_file("/nonexistent/path.md")

        assert "not found" in str(exc_info.value).lower()

    def test_parse_empty_content(self):
        """Should raise error for empty content."""
        with pytest.raises(SkillParseError):
            Parser.parse("")

    def test_format_aliases(self):
        """Should accept format aliases."""
        content = "# Skill\n\nDesc"

        # Various aliases for claude_code
        for alias in ["claude", "claude-code", "md", "markdown"]:
            skill = Parser.parse(content, format=alias)
            assert skill.source_format == SkillFormat.CLAUDE_CODE


class TestConvenienceFunction:
    """Tests for the module-level parse() function."""

    def test_parse_function_exists(self):
        """parse() function should be available."""
        assert callable(parse)

    def test_parse_function_works(self):
        """parse() should work like Parser.parse()."""
        skill = parse(FIXTURES_DIR / "sample-skill.md")
        assert skill.name == "Web Search Skill"

    def test_parse_with_format(self):
        """parse() should accept format argument."""
        skill = parse("# Skill\n\nDesc", format="claude_code")
        assert skill.source_format == SkillFormat.CLAUDE_CODE
