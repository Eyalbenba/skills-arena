"""Parser for OpenAI function calling schema format.

OpenAI function schemas are JSON objects that define:
- name: Function name
- description: Function description
- parameters: JSON Schema for parameters

See: https://platform.openai.com/docs/guides/function-calling
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from skills_arena.exceptions import SkillParseError
from skills_arena.models import Parameter, Skill, SkillFormat
from skills_arena.parser.base import BaseParser, estimate_tokens

if TYPE_CHECKING:
    pass


class OpenAIParser(BaseParser):
    """Parser for OpenAI function calling schemas.

    Expected format (JSON):
    ```json
    {
        "name": "function_name",
        "description": "What the function does",
        "parameters": {
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

    Can also parse the tools format:
    ```json
    {
        "type": "function",
        "function": {
            "name": "function_name",
            "description": "...",
            "parameters": {...}
        }
    }
    ```
    """

    @property
    def format(self) -> SkillFormat:
        """Return the skill format this parser handles."""
        return SkillFormat.OPENAI

    def can_parse(self, content: str) -> bool:
        """Check if content looks like an OpenAI function schema.

        Looks for JSON with either:
        - "name" and "description" keys at root level
        - "type": "function" with nested "function" object
        """
        content = content.strip()
        if not content:
            return False

        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                return False

            # Check for direct function schema
            if "name" in data and "description" in data:
                return True

            # Check for tools format
            if data.get("type") == "function" and "function" in data:
                return True

            return False
        except json.JSONDecodeError:
            return False

    def parse(self, content: str, source_path: str | None = None) -> Skill:
        """Parse OpenAI function schema content.

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
                expected_format="OpenAI function schema (JSON)",
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise SkillParseError(
                f"Invalid JSON: {e}",
                path=source_path,
                expected_format="OpenAI function schema (JSON)",
            ) from e

        # Extract the function definition (handle tools format)
        func_def = self._extract_function_definition(data, source_path)

        # Extract name
        name = func_def.get("name")
        if not name:
            raise SkillParseError(
                "Missing 'name' field in function schema",
                path=source_path,
                expected_format="OpenAI function schema (JSON)",
            )

        # Extract description
        description = func_def.get("description", "")

        # Extract parameters
        parameters = self._extract_parameters(func_def.get("parameters", {}))

        return Skill(
            name=name,
            description=description,
            parameters=parameters,
            when_to_use=[],  # OpenAI schema doesn't have this
            source_format=SkillFormat.OPENAI,
            token_count=estimate_tokens(content),
            raw_content=content,
            source_path=source_path,
        )

    def _extract_function_definition(
        self, data: dict[str, Any], source_path: str | None
    ) -> dict[str, Any]:
        """Extract the function definition from various formats.

        Handles:
        - Direct function schema: {"name": "...", "description": "..."}
        - Tools format: {"type": "function", "function": {...}}
        """
        # Check for tools format
        if data.get("type") == "function" and "function" in data:
            func_def = data["function"]
            if not isinstance(func_def, dict):
                raise SkillParseError(
                    "Invalid 'function' field - expected object",
                    path=source_path,
                    expected_format="OpenAI function schema (JSON)",
                )
            return func_def

        # Assume direct format
        return data

    def _extract_parameters(self, params_schema: dict[str, Any]) -> list[Parameter]:
        """Extract parameters from a JSON Schema parameters object.

        Handles:
        - properties: Object mapping param names to schemas
        - required: Array of required parameter names
        """
        if not isinstance(params_schema, dict):
            return []

        properties = params_schema.get("properties", {})
        required = set(params_schema.get("required", []))

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
        """Parse an OpenAI function schema from a dictionary.

        This is a convenience method for when you already have parsed JSON.

        Args:
            data: The function schema dictionary.
            source_path: Optional path to the source.

        Returns:
            A Skill object.
        """
        return self.parse(json.dumps(data), source_path)
