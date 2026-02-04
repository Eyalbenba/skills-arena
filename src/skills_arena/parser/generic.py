"""Parser for generic/plain text skill descriptions.

This parser handles plain text skill definitions that don't follow any
specific format. It attempts to extract structure from unformatted text.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from skills_arena.models import Parameter, Skill, SkillFormat
from skills_arena.parser.base import BaseParser, estimate_tokens

if TYPE_CHECKING:
    pass


class GenericParser(BaseParser):
    """Parser for generic/plain text skill descriptions.

    This is a fallback parser that handles skill definitions that don't
    follow a specific format. It makes best-effort attempts to extract:
    - Name: From first line or heading
    - Description: From the full text
    - Parameters: From parameter-like patterns (optional)

    Since this is a fallback, it should always succeed (no parsing errors).
    """

    @property
    def format(self) -> SkillFormat:
        """Return the skill format this parser handles."""
        return SkillFormat.GENERIC

    def can_parse(self, content: str) -> bool:
        """Generic parser can always parse any non-empty content.

        Returns:
            True if content is not empty, False otherwise.
        """
        return bool(content and content.strip())

    def parse(self, content: str, source_path: str | None = None) -> Skill:
        """Parse generic text content into a Skill.

        Makes best-effort extraction of skill components from unstructured text.

        Args:
            content: The text content to parse.
            source_path: Optional path to the source file.

        Returns:
            A Skill object with extracted information.
        """
        content = content.strip()

        # Extract name from first line
        name = self._extract_name(content, source_path)

        # Use full content as description
        description = self._extract_description(content)

        # Attempt to extract parameters
        parameters = self._extract_parameters(content)

        # Attempt to extract usage patterns
        when_to_use = self._extract_patterns(content)

        return Skill(
            name=name,
            description=description,
            parameters=parameters,
            when_to_use=when_to_use,
            source_format=SkillFormat.GENERIC,
            token_count=estimate_tokens(content),
            raw_content=content,
            source_path=source_path,
        )

    def _extract_name(self, content: str, source_path: str | None) -> str:
        """Extract skill name from content.

        Priority:
        1. Markdown heading (# Title)
        2. First non-empty line
        3. Filename from source_path
        4. Default name
        """
        lines = content.split("\n")

        # Try markdown heading
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                # Remove # markers and clean
                name = re.sub(r"^#+\s*", "", stripped)
                return name

        # Try first non-empty line (if it's short enough to be a title)
        for line in lines:
            stripped = line.strip()
            if stripped:
                # If first line is reasonably short, use as name
                if len(stripped) <= 100 and not stripped.endswith((".", "!", "?")):
                    return stripped
                break

        # Try filename
        if source_path:
            from pathlib import Path

            filename = Path(source_path).stem
            if filename:
                # Convert filename to readable name
                return filename.replace("-", " ").replace("_", " ").title()

        return "Unnamed Skill"

    def _extract_description(self, content: str) -> str:
        """Extract description from content.

        For generic text, we use the full content as description,
        optionally removing the first line if it looks like a title.
        """
        lines = content.split("\n")
        first_line = next((line.strip() for line in lines if line.strip()), "")

        # If first line looks like a title, exclude it from description
        if first_line.startswith("#") or (
            len(first_line) <= 100 and not first_line.endswith((".", "!", "?"))
        ):
            # Skip the first non-empty line
            found_first = False
            description_lines = []
            for line in lines:
                if not found_first and line.strip():
                    found_first = True
                    continue
                description_lines.append(line)
            return "\n".join(description_lines).strip()

        return content

    def _extract_parameters(self, content: str) -> list[Parameter]:
        """Attempt to extract parameters from text.

        Looks for patterns like:
        - parameter_name: description
        - --parameter-name: description
        - `parameter`: description
        """
        parameters = []
        seen_names: set[str] = set()

        # Pattern for command-line style parameters
        cli_pattern = re.compile(
            r"(?:^|\s)--?([a-z][a-z0-9_-]*)"  # --param or -param
            r"(?:\s*[=:]\s*|\s+)"  # separator
            r"([^-\n][^\n]{0,200})?",  # description (optional)
            re.IGNORECASE | re.MULTILINE,
        )

        for match in cli_pattern.finditer(content):
            name = match.group(1).strip().replace("-", "_")
            description = (match.group(2) or "").strip()

            if name and name not in seen_names:
                seen_names.add(name)
                parameters.append(
                    Parameter(
                        name=name,
                        description=description,
                        type="string",
                        required=False,
                    )
                )

        # Pattern for backtick parameters
        backtick_pattern = re.compile(
            r"`([a-z][a-z0-9_]*)`"  # `param_name`
            r"(?:\s*[:-]\s*|\s+)"  # separator
            r"([^`\n][^\n]{0,200})?",  # description (optional)
            re.IGNORECASE,
        )

        for match in backtick_pattern.finditer(content):
            name = match.group(1).strip()
            description = (match.group(2) or "").strip()

            if name and name not in seen_names:
                seen_names.add(name)
                parameters.append(
                    Parameter(
                        name=name,
                        description=description,
                        type="string",
                        required=False,
                    )
                )

        return parameters

    def _extract_patterns(self, content: str) -> list[str]:
        """Attempt to extract usage patterns from text.

        Looks for patterns like:
        - "Use when..."
        - "Good for..."
        - Example phrases in quotes
        """
        patterns = []

        # Look for "use when" patterns
        use_patterns = re.findall(
            r"(?:use|good|helpful|useful)\s+(?:for|when|if)\s+([^.\n]+)",
            content,
            re.IGNORECASE,
        )
        patterns.extend(use_patterns)

        # Look for quoted examples
        quoted = re.findall(r'"([^"]{10,100})"', content)
        patterns.extend(quoted)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_patterns = []
        for p in patterns:
            p_clean = p.strip()
            if p_clean and p_clean.lower() not in seen:
                seen.add(p_clean.lower())
                unique_patterns.append(p_clean)

        return unique_patterns[:10]  # Limit to 10 patterns
