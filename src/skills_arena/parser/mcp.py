"""Parser for MCP (Model Context Protocol) tool definitions.

MCP tools follow a specific schema format for defining tools that can be
used by AI models. The format is similar to OpenAI function schemas but
with some MCP-specific conventions.

See: https://modelcontextprotocol.io/docs/concepts/tools
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from skills_arena.exceptions import SkillParseError
from skills_arena.models import Parameter, Skill, SkillFormat
from skills_arena.parser.base import BaseParser, estimate_tokens

if TYPE_CHECKING:
    pass


class MCPParser(BaseParser):
    """Parser for MCP tool definitions.

    Expected format (JSON):
    ```json
    {
        "name": "tool_name",
        "description": "Tool description",
        "inputSchema": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param1"]
        }
    }
    ```

    MCP uses `inputSchema` instead of OpenAI's `parameters`.
    """

    @property
    def format(self) -> SkillFormat:
        """Return the skill format this parser handles."""
        return SkillFormat.MCP

    def can_parse(self, content: str) -> bool:
        """Check if content looks like an MCP tool definition.

        Looks for JSON with:
        - "name" key
        - "inputSchema" key (MCP-specific)
        """
        content = content.strip()
        if not content:
            return False

        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                return False

            # MCP-specific: look for inputSchema
            return "name" in data and "inputSchema" in data
        except json.JSONDecodeError:
            return False

    def parse(self, content: str, source_path: str | None = None) -> Skill:
        """Parse MCP tool definition content.

        Args:
            content: The JSON content to parse.
            source_path: Optional path to the source file.

        Returns:
            A Skill object.

        Raises:
            SkillParseError: If the content cannot be parsed.
        """
        content = content.strip()
        if not content:
            raise SkillParseError(
                "Empty skill content",
                path=source_path,
                expected_format="MCP tool definition (JSON)",
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise SkillParseError(
                f"Invalid JSON: {e}",
                path=source_path,
                expected_format="MCP tool definition (JSON)",
            ) from e

        if not isinstance(data, dict):
            raise SkillParseError(
                "Expected JSON object at root level",
                path=source_path,
                expected_format="MCP tool definition (JSON)",
            )

        # Extract name
        name = data.get("name")
        if not name:
            raise SkillParseError(
                "Missing 'name' field in MCP tool definition",
                path=source_path,
                expected_format="MCP tool definition (JSON)",
            )

        # Extract description
        description = data.get("description", "")

        # Extract parameters from inputSchema
        input_schema = data.get("inputSchema", {})
        parameters = self._extract_parameters(input_schema)

        # Extract additional MCP-specific fields if present
        when_to_use = []

        # Some MCP tools include hints about when to use them
        if "hints" in data:
            hints = data["hints"]
            if isinstance(hints, dict) and "whenToUse" in hints:
                when_hints = hints["whenToUse"]
                if isinstance(when_hints, list):
                    when_to_use = [str(h) for h in when_hints]
                elif isinstance(when_hints, str):
                    when_to_use = [when_hints]

        return Skill(
            name=name,
            description=description,
            parameters=parameters,
            when_to_use=when_to_use,
            source_format=SkillFormat.MCP,
            token_count=estimate_tokens(content),
            raw_content=content,
            source_path=source_path,
        )

    def _extract_parameters(self, input_schema: dict[str, Any]) -> list[Parameter]:
        """Extract parameters from an MCP inputSchema object.

        The inputSchema follows JSON Schema format similar to OpenAI parameters.
        """
        if not isinstance(input_schema, dict):
            return []

        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))

        parameters = []
        for name, schema in properties.items():
            if not isinstance(schema, dict):
                continue

            param_type = schema.get("type", "string")
            description = schema.get("description", "")
            default = schema.get("default")

            # Handle enum types
            if "enum" in schema:
                enum_values = ", ".join(str(v) for v in schema["enum"])
                if description:
                    description += f" (one of: {enum_values})"
                else:
                    description = f"One of: {enum_values}"

            # Handle anyOf for union types
            if "anyOf" in schema:
                types = [s.get("type", "unknown") for s in schema["anyOf"] if isinstance(s, dict)]
                param_type = " | ".join(filter(None, types)) or "string"

            parameters.append(
                Parameter(
                    name=name,
                    description=description,
                    type=param_type,
                    required=name in required,
                    default=default,
                )
            )

        return parameters

    def parse_dict(self, data: dict[str, Any], source_path: str | None = None) -> Skill:
        """Parse an MCP tool definition from a dictionary.

        This is a convenience method for when you already have parsed JSON.

        Args:
            data: The tool definition dictionary.
            source_path: Optional path to the source.

        Returns:
            A Skill object.
        """
        return self.parse(json.dumps(data), source_path)
