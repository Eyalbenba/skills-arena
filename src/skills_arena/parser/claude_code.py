"""Parser for Claude Code skill format (.md files).

Claude Code skills are defined in Markdown files with a specific structure:
- Title (# header)
- Description section
- Parameters section (optional)
- When to Use section (optional)
- Examples section (optional)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from skills_arena.exceptions import SkillParseError
from skills_arena.models import Parameter, Skill, SkillFormat
from skills_arena.parser.base import BaseParser, estimate_tokens

if TYPE_CHECKING:
    pass


class ClaudeCodeParser(BaseParser):
    """Parser for Claude Code .md skill files.

    Expected format:
    ```markdown
    # Skill Name

    Brief description...

    ## Description

    Detailed description...

    ## Parameters

    - `param_name` (type, required/optional): Description
    - `another_param` (type, optional): Description (default: value)

    ## When to Use

    Use this skill when:
    - Condition 1
    - Condition 2

    ## Examples

    - "Example prompt 1"
    - "Example prompt 2"
    ```
    """

    @property
    def format(self) -> SkillFormat:
        """Return the skill format this parser handles."""
        return SkillFormat.CLAUDE_CODE

    def can_parse(self, content: str) -> bool:
        """Check if content looks like a Claude Code skill definition.

        A valid Claude Code skill should have:
        - A markdown title (# heading)
        - Some description text
        """
        content = content.strip()
        if not content:
            return False

        # Must start with a markdown heading
        lines = content.split("\n")
        first_non_empty = next((line for line in lines if line.strip()), "")

        return first_non_empty.startswith("#")

    def parse(self, content: str, source_path: str | None = None) -> Skill:
        """Parse Claude Code skill markdown content.

        Args:
            content: The markdown content to parse.
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
                expected_format="Claude Code (.md)",
            )

        # Extract skill name from title
        name = self._extract_name(content)
        if not name:
            raise SkillParseError(
                "Could not find skill name. Expected a markdown heading (# Title).",
                path=source_path,
                expected_format="Claude Code (.md)",
            )

        # Extract sections
        description = self._extract_description(content)
        parameters = self._extract_parameters(content)
        when_to_use = self._extract_when_to_use(content)

        # Combine examples from "Examples" section and "When to Use"
        examples = self._extract_examples(content)
        all_when_to_use = when_to_use + examples

        return Skill(
            name=name,
            description=description,
            parameters=parameters,
            when_to_use=all_when_to_use,
            source_format=SkillFormat.CLAUDE_CODE,
            token_count=estimate_tokens(content),
            raw_content=content,
            source_path=source_path,
        )

    def _extract_name(self, content: str) -> str | None:
        """Extract skill name from the first markdown heading."""
        # Match # Title at the start
        match = re.search(r"^#\s+(.+?)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_description(self, content: str) -> str:
        """Extract the description from the content.

        Looks for:
        1. A dedicated ## Description section
        2. Falls back to text between title and first section
        """
        # Try to find dedicated Description section
        description_section = self._extract_section(content, "Description")
        if description_section:
            return description_section.strip()

        # Fallback: get text between title and first ## section
        lines = content.split("\n")
        description_lines = []
        in_description = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines at the start
            if not in_description and not stripped:
                continue

            # Start after the title
            if stripped.startswith("# ") and not in_description:
                in_description = True
                continue

            # Stop at the first ## section
            if stripped.startswith("## "):
                break

            if in_description:
                description_lines.append(line)

        return "\n".join(description_lines).strip()

    def _extract_section(self, content: str, section_name: str) -> str | None:
        """Extract content of a specific ## section.

        Args:
            content: The full markdown content.
            section_name: The name of the section to extract.

        Returns:
            The section content, or None if not found.
        """
        # Pattern to match the section header (case-insensitive)
        pattern = rf"^##\s+{re.escape(section_name)}\s*$"
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if not match:
            return None

        # Get content from after the header to the next ## or end
        start = match.end()
        next_section = re.search(r"^##\s+", content[start:], re.MULTILINE)

        if next_section:
            end = start + next_section.start()
        else:
            end = len(content)

        return content[start:end].strip()

    def _extract_parameters(self, content: str) -> list[Parameter]:
        """Extract parameters from the Parameters section.

        Expected format:
        - `param_name` (type, required): Description
        - `param_name` (type, optional): Description (default: value)
        - `param_name` (type): Description
        """
        params_section = self._extract_section(content, "Parameters")
        if not params_section:
            return []

        parameters = []

        # Pattern to match parameter definitions
        # Matches: - `name` (type, required/optional): description
        param_pattern = re.compile(
            r"^[-*]\s*`([^`]+)`\s*"  # - `param_name`
            r"(?:\(([^)]+)\))?"  # (type, required/optional) - optional
            r"(?:\s*:)?\s*"  # : separator - optional
            r"(.*)$",  # description
            re.MULTILINE,
        )

        for match in param_pattern.finditer(params_section):
            name = match.group(1).strip()
            type_info = match.group(2) or ""
            description = match.group(3).strip()

            # Parse type info (e.g., "string, required" or "integer, optional")
            param_type = "string"
            required = False
            default = None

            if type_info:
                type_parts = [p.strip().lower() for p in type_info.split(",")]
                for part in type_parts:
                    if part in ("string", "integer", "number", "boolean", "array", "object"):
                        param_type = part
                    elif part == "required":
                        required = True
                    elif part == "optional":
                        required = False

            # Check for default value in description
            default_match = re.search(r"\(default:\s*([^)]+)\)", description, re.IGNORECASE)
            if default_match:
                default = default_match.group(1).strip()
                # Remove default from description
                description = re.sub(r"\s*\(default:\s*[^)]+\)", "", description).strip()

            parameters.append(
                Parameter(
                    name=name,
                    description=description,
                    type=param_type,
                    required=required,
                    default=default,
                )
            )

        return parameters

    def _extract_when_to_use(self, content: str) -> list[str]:
        """Extract 'when to use' patterns from the When to Use section."""
        when_section = self._extract_section(content, "When to Use")
        if not when_section:
            return []

        patterns = []

        # Extract bullet points
        for line in when_section.split("\n"):
            line = line.strip()
            if line.startswith(("-", "*")):
                # Remove bullet point marker
                pattern = re.sub(r"^[-*]\s*", "", line).strip()
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _extract_examples(self, content: str) -> list[str]:
        """Extract example prompts from the Examples section."""
        examples_section = self._extract_section(content, "Examples")
        if not examples_section:
            return []

        examples = []

        # Extract bullet points and quoted strings
        for line in examples_section.split("\n"):
            line = line.strip()
            if line.startswith(("-", "*")):
                # Remove bullet point marker
                example = re.sub(r"^[-*]\s*", "", line).strip()

                # Remove surrounding quotes if present
                if (example.startswith('"') and example.endswith('"')) or (
                    example.startswith("'") and example.endswith("'")
                ):
                    example = example[1:-1]

                if example:
                    examples.append(example)

        return examples
