"""Claude Code (Agent SDK) agent implementation.

This module implements the ClaudeCodeAgent which uses the official
Claude Agent SDK to run prompts and determine skill selection.

The approach mimics real Claude Code behavior:
1. Skills are written to .claude/skills/ in the working directory
2. Claude Code discovers them naturally via setting_sources=["project"]
3. A generic task is given (vanilla system prompt)
4. Claude decides which skill to invoke
5. We intercept the selection via PreToolUse hook (without executing)

This tests REAL skill discovery - not injected content.

Usage:
    ```python
    async with ClaudeCodeAgent() as agent:
        selection = await agent.select_skill(
            prompt="Research AI news and summarize findings",
            available_skills=[tavily_skill, firecrawl_skill],
        )
        print(f"Selected: {selection.skill}")
    ```
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import AgentError, APIKeyError
from ..models import Skill, SkillSelection
from .base import BaseAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import the Claude Agent SDK
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        HookMatcher,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolUseBlock,
        query,
    )

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    logger.warning(
        "Claude Agent SDK not installed. Install with: pip install claude-agent-sdk"
    )


class ClaudeCodeAgent(BaseAgent):
    """Agent implementation using the Claude Agent SDK.

    This agent mimics real Claude Code behavior by:
    1. Writing skills to .claude/skills/ in the working directory
    2. Letting Claude Code discover them via setting_sources=["project"]
    3. Using vanilla system prompt (no injected content)
    4. Intercepting the selection via PreToolUse hook (without executing)

    This tests REAL skill discovery - exactly how users experience it.

    Example:
        ```python
        async with ClaudeCodeAgent() as agent:
            selection = await agent.select_skill(
                prompt="Research AI news and summarize findings",
                available_skills=[tavily_skill, firecrawl_skill],
            )
            print(f"Selected: {selection.skill}")
        ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 10,
        timeout_seconds: int = 60,
        working_dir: str | Path | None = None,
        api_key: str | None = None,
    ):
        """Initialize the Claude Code agent.

        Args:
            model: The Claude model to use (default: claude-sonnet-4).
            max_turns: Maximum turns before selecting (allows thinking time).
            timeout_seconds: Request timeout.
            working_dir: Working directory for Claude (defaults to temp dir).
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
        """
        if not CLAUDE_SDK_AVAILABLE:
            raise ImportError(
                "Claude Agent SDK is required but not installed. "
                "Install with: pip install claude-agent-sdk"
            )

        if shutil.which("claude") is None:
            raise AgentError(
                message=(
                    "The 'claude' CLI binary was not found on PATH. "
                    "The Claude Agent SDK requires Claude Code to be installed. "
                    "Install it with: npm install -g @anthropic-ai/claude-code"
                ),
                agent_name="claude-code",
            )

        self.model = model
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        # Working directory - use temp dir by default for isolation
        if working_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="skills_arena_")
            self.working_dir = Path(self._temp_dir)
        else:
            self._temp_dir = None
            self.working_dir = Path(working_dir)

        # Track installed skills for cleanup
        self._installed_skills_dir: Path | None = None

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return "claude-code"

    def _install_skills_to_filesystem(self, skills: list[Skill]) -> None:
        """Write skills to .claude/skills/ for Claude Code to discover.

        This creates the actual skill file structure that Claude Code
        expects, allowing natural skill discovery.

        Args:
            skills: Skills to install.
        """
        skills_dir = self.working_dir / ".claude" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        self._installed_skills_dir = skills_dir

        for skill in skills:
            # Create skill directory using sanitized skill name
            skill_slug = skill.name.lower().replace(" ", "-").replace("_", "-")
            skill_dir = skills_dir / skill_slug
            skill_dir.mkdir(exist_ok=True)

            # Build SKILL.md content with frontmatter
            skill_content = self._build_skill_file_content(skill)

            # Write SKILL.md
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(skill_content)
            logger.debug(f"Installed skill: {skill.name} -> {skill_file}")

    def _build_skill_file_content(self, skill: Skill) -> str:
        """Build the SKILL.md file content for a skill.

        Args:
            skill: The skill to convert.

        Returns:
            SKILL.md content with frontmatter.
        """
        # Build frontmatter
        frontmatter_lines = [
            "---",
            f"name: {skill.name}",
        ]

        # Add description (handle multiline)
        if skill.description:
            if "\n" in skill.description:
                frontmatter_lines.append("description: |")
                for line in skill.description.split("\n")[:10]:  # Limit to 10 lines in frontmatter
                    frontmatter_lines.append(f"  {line}")
            else:
                frontmatter_lines.append(f"description: {skill.description[:200]}")

        frontmatter_lines.append("---")

        # Build body content
        body_lines = [
            "",
            f"# {skill.name}",
            "",
        ]

        # Add full description in body
        if skill.description:
            body_lines.append(skill.description)
            body_lines.append("")

        # Add when to use section
        if skill.when_to_use:
            body_lines.append("## When to Use")
            body_lines.append("")
            for example in skill.when_to_use:
                body_lines.append(f"- {example}")
            body_lines.append("")

        # NOTE: We intentionally do NOT include raw_content here.
        # Claude Code discovers skills and only shows a brief summary in
        # the system-reminder. The full skill content is only loaded when
        # the skill is actually invoked. This tests real discovery behavior.

        return "\n".join(frontmatter_lines) + "\n".join(body_lines)

    async def select_skill(
        self,
        prompt: str,
        available_skills: list[Skill],
    ) -> SkillSelection:
        """Run a prompt through Claude Code and determine skill selection.

        This method:
        1. Installs skills to .claude/skills/ in the working directory
        2. Configures Claude Code with setting_sources=["project"] for isolation
        3. Uses vanilla system prompt (no injected skill content)
        4. Intercepts skill selection via PreToolUse hook

        This tests REAL skill discovery behavior.

        Args:
            prompt: The user prompt (task) to process.
            available_skills: Skills available for selection.

        Returns:
            SkillSelection with the chosen skill.
        """
        if not self._api_key:
            raise APIKeyError("Anthropic", "ANTHROPIC_API_KEY")

        if not available_skills:
            return SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning="No skills available for selection",
            )

        start_time = time.time()

        # Install skills to filesystem for Claude Code to discover
        self._install_skills_to_filesystem(available_skills)

        # Get skill names for matching
        skill_names = [s.name for s in available_skills]

        # State to capture selection and reasoning
        selection_state = {
            "selected_skill": None,
            "tool_input": {},
            "reasoning_text": [],  # Capture Claude's text output before selection
        }

        # Build a mapping from skill name variations to canonical names
        # This helps match "firecrawl" -> "Firecrawl CLI", etc.
        skill_name_map: dict[str, str] = {}
        for skill in available_skills:
            # Map exact name
            skill_name_map[skill.name.lower()] = skill.name
            # Map simplified versions (remove "skill", "cli", etc.)
            simplified = skill.name.lower().replace(" skill", "").replace(" cli", "").strip()
            skill_name_map[simplified] = skill.name
            # Map the slug version (how it appears in filesystem)
            slug = skill.name.lower().replace(" ", "-").replace("_", "-")
            skill_name_map[slug] = skill.name

        # Build PreToolUse hook to intercept skill selection
        async def capture_skill_selection(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            """Hook to capture when Claude selects one of our test skills."""
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})

            logger.debug(f"[HOOK] PreToolUse fired: tool={tool_name}, input_keys={list(tool_input.keys())}")

            selected_skill_name: str | None = None

            # Check if Claude is using the native Skill tool
            if tool_name == "Skill":
                # Parse the skill being invoked from the input
                invoked_skill = tool_input.get("skill", "").lower()
                logger.info(f"[HOOK] Skill tool invoked with: {invoked_skill}")

                # Match against our test skills
                if invoked_skill in skill_name_map:
                    selected_skill_name = skill_name_map[invoked_skill]

            # Also check direct tool calls matching our skills
            elif tool_name in skill_names:
                selected_skill_name = tool_name
            elif tool_name.lower() in skill_name_map:
                selected_skill_name = skill_name_map[tool_name.lower()]

            # If we found a match, record it
            if selected_skill_name:
                selection_state["selected_skill"] = selected_skill_name
                selection_state["tool_input"] = tool_input
                logger.info(f"★ Skill selection captured: {selected_skill_name}")

                # Block execution - we only wanted to see the choice
                return {
                    "decision": "block",
                    "reason": f"Selection recorded: {selected_skill_name}",
                }

            # Allow built-in tools (Read, Glob, Bash, etc.) to run normally
            return {}

        try:
            # Configure with vanilla settings - no custom system prompt
            # Skills are discovered from .claude/skills/ via setting_sources
            options = ClaudeAgentOptions(
                max_turns=self.max_turns,
                cwd=str(self.working_dir),
                model=self.model,
                permission_mode="acceptEdits",  # Don't prompt for permissions
                setting_sources=["project"],  # Only load project skills (isolation!)
                hooks={
                    "PreToolUse": [
                        # Intercept Skill tool calls
                        HookMatcher(
                            matcher="Skill",
                            hooks=[capture_skill_selection],
                        ),
                    ],
                },
            )

            raw_response: list[Any] = []

            # Log the prompt being sent
            logger.info(f">>> PROMPT: {prompt}")
            logger.debug(f"Available skills: {skill_names}")

            # Suppress asyncio errors from breaking out of async generator early
            # This happens because we break the loop when skill is selected
            loop = asyncio.get_event_loop()
            original_handler = loop.get_exception_handler()

            def suppress_cancel_scope_errors(loop: asyncio.AbstractEventLoop, context: dict) -> None:
                """Suppress cancel scope errors from early generator exit."""
                exception = context.get("exception")
                if exception and "cancel scope" in str(exception):
                    return  # Suppress this specific error
                # Call original handler for other exceptions
                if original_handler:
                    original_handler(loop, context)
                else:
                    loop.default_exception_handler(context)

            loop.set_exception_handler(suppress_cancel_scope_errors)

            # Query Claude and let it naturally decide
            message_count = 0
            try:
                async for message in query(prompt=prompt, options=options):
                    message_count += 1
                    message_type = type(message).__name__
                    raw_response.append(message)

                    # Log each message type
                    logger.debug(f"[Message {message_count}] Type: {message_type}")

                    # Log SystemMessage content (shows skill discovery)
                    if isinstance(message, SystemMessage):
                        logger.info(f"  └─ SystemMessage subtype: {message.subtype}")
                        # Log the full data for debugging
                        if message.data:
                            data_str = json.dumps(message.data, indent=2, default=str)
                            # Log full content at DEBUG, summary at INFO
                            if len(data_str) > 500:
                                logger.info(f"  └─ SystemMessage data (truncated): {data_str[:500]}...")
                                logger.debug(f"  └─ SystemMessage data (full):\n{data_str}")
                            else:
                                logger.info(f"  └─ SystemMessage data: {data_str}")

                    # Parse AssistantMessage content blocks
                    elif isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                # Capture Claude's reasoning text
                                selection_state["reasoning_text"].append(block.text)
                                # Log text (truncated for readability)
                                text_preview = block.text[:150].replace('\n', ' ')
                                logger.debug(f"  └─ TextBlock: {text_preview}...")


                            elif isinstance(block, ToolUseBlock):
                                # Log tool call details
                                tool_input = block.input if hasattr(block, "input") else {}
                                tool_input_str = json.dumps(tool_input)[:100] if tool_input else "{}"
                                logger.info(f"  └─ ToolUseBlock: {block.name}")
                                logger.debug(f"      Input: {tool_input_str}")

                                # Check if Claude is using the native Skill tool
                                if block.name == "Skill" and selection_state["selected_skill"] is None:
                                    invoked_skill = tool_input.get("skill", "").lower()
                                    logger.info(f"  └─ Skill tool invoking: '{invoked_skill}'")

                                    # Match against our test skills
                                    if invoked_skill in skill_name_map:
                                        matched_skill = skill_name_map[invoked_skill]
                                        selection_state["selected_skill"] = matched_skill
                                        selection_state["tool_input"] = tool_input
                                        logger.info(f"  └─ ★ SKILL SELECTED: {matched_skill}")

                                # Also check direct tool calls matching our skills
                                elif block.name in skill_names and selection_state["selected_skill"] is None:
                                    selection_state["selected_skill"] = block.name
                                    selection_state["tool_input"] = tool_input
                                    logger.info(f"  └─ ★ SKILL SELECTED: {block.name}")

                    elif isinstance(message, ResultMessage):
                        logger.debug(f"  └─ ResultMessage: turns={message.num_turns}, error={message.is_error}")

                    # Stop once we have a selection - don't waste more tokens
                    if selection_state["selected_skill"]:
                        if not selection_state.get("_logged"):
                            logger.info(f"<<< SELECTED: {selection_state['selected_skill']}")
                            selection_state["_logged"] = True
                        break
            except (asyncio.CancelledError, RuntimeError):
                # Ignore errors during cleanup when breaking out of async generator
                pass
            finally:
                # Restore original exception handler
                loop.set_exception_handler(original_handler)

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Skill selection completed in {latency_ms:.1f}ms")

            # Log final result
            if not selection_state["selected_skill"]:
                logger.warning(f"<<< NO SELECTION made for prompt: {prompt[:50]}...")

            selected = selection_state["selected_skill"] is not None
            # Join all captured reasoning text
            reasoning = "\n".join(selection_state["reasoning_text"]) if selection_state["reasoning_text"] else ""
            return SkillSelection(
                skill=selection_state["selected_skill"],
                confidence=1.0 if selected else 0.0,
                reasoning=reasoning,
                raw_response=raw_response,
            )

        except Exception as e:
            logger.error(f"Error during skill selection: {e}")
            raise AgentError(
                message=str(e),
                agent_name=self.name,
            ) from e

    async def close(self) -> None:
        """Clean up resources."""
        # Clean up installed skills
        if self._installed_skills_dir is not None:
            try:
                shutil.rmtree(self._installed_skills_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error cleaning up skills dir: {e}")
            self._installed_skills_dir = None

        # Clean up temp directory if we created one
        if self._temp_dir is not None:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error cleaning up temp dir: {e}")
            self._temp_dir = None


class ClaudeCodeAgentWithClient(ClaudeCodeAgent):
    """Extended Claude Code agent that uses ClaudeSDKClient for persistent sessions.

    This variant maintains a ClaudeSDKClient for multi-turn conversations,
    useful when you need Claude to build context across multiple exchanges
    before making a skill selection.

    Example:
        ```python
        async with ClaudeCodeAgentWithClient() as agent:
            selection = await agent.select_skill(
                prompt="I need to research competitors and summarize findings",
                available_skills=skills,
            )
        ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 10,
        timeout_seconds: int = 60,
        working_dir: str | Path | None = None,
        api_key: str | None = None,
    ):
        """Initialize the Claude Code agent with client support.

        Args:
            model: The Claude model to use.
            max_turns: Maximum conversation turns.
            timeout_seconds: Request timeout.
            working_dir: Working directory for Claude.
            api_key: Anthropic API key.
        """
        super().__init__(
            model=model,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            working_dir=working_dir,
            api_key=api_key,
        )
        self._client: ClaudeSDKClient | None = None

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return "claude-code-client"

    async def select_skill(
        self,
        prompt: str,
        available_skills: list[Skill],
    ) -> SkillSelection:
        """Run skill selection using ClaudeSDKClient for persistent context.

        Uses ClaudeSDKClient which maintains conversation context,
        allowing multi-turn interactions before skill selection.
        Skills are discovered from .claude/skills/ (filesystem-based).

        Args:
            prompt: The user prompt to process.
            available_skills: Skills available for selection.

        Returns:
            SkillSelection with the chosen skill.
        """
        if not self._api_key:
            raise APIKeyError("Anthropic", "ANTHROPIC_API_KEY")

        if not available_skills:
            return SkillSelection(
                skill=None,
                confidence=0.0,
                reasoning="No skills available for selection",
            )

        start_time = time.time()

        # Install skills to filesystem for Claude Code to discover
        self._install_skills_to_filesystem(available_skills)

        # Build skill name mapping (same as parent class)
        skill_name_map: dict[str, str] = {}
        for skill in available_skills:
            skill_name_map[skill.name.lower()] = skill.name
            simplified = skill.name.lower().replace(" skill", "").replace(" cli", "").strip()
            skill_name_map[simplified] = skill.name
            slug = skill.name.lower().replace(" ", "-").replace("_", "-")
            skill_name_map[slug] = skill.name

        selection_state = {
            "selected_skill": None,
            "tool_input": {},
        }

        async def capture_skill_selection(
            input_data: dict[str, Any],
            tool_use_id: str | None,  # noqa: ARG001
            context: Any,  # noqa: ARG001
        ) -> dict[str, Any]:
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})

            if tool_name == "Skill":
                invoked_skill = tool_input.get("skill", "").lower()
                if invoked_skill in skill_name_map:
                    selection_state["selected_skill"] = skill_name_map[invoked_skill]
                    selection_state["tool_input"] = tool_input
                    return {"decision": "block", "reason": f"Selection: {skill_name_map[invoked_skill]}"}
            return {}

        try:
            # Configure with vanilla settings - no custom system prompt
            options = ClaudeAgentOptions(
                max_turns=self.max_turns,
                cwd=str(self.working_dir),
                model=self.model,
                permission_mode="acceptEdits",
                setting_sources=["project"],  # Only load project skills (isolation!)
                hooks={
                    "PreToolUse": [
                        HookMatcher(
                            matcher="Skill",
                            hooks=[capture_skill_selection],
                        ),
                    ],
                },
            )

            raw_response: list[Any] = []

            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)

                async for message in client.receive_response():
                    raw_response.append(message)

                    # Check for Skill tool calls in response
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                if block.name == "Skill" and selection_state["selected_skill"] is None:
                                    tool_input = block.input if hasattr(block, "input") else {}
                                    invoked_skill = tool_input.get("skill", "").lower()
                                    if invoked_skill in skill_name_map:
                                        selection_state["selected_skill"] = skill_name_map[invoked_skill]
                                        selection_state["tool_input"] = tool_input

                    if selection_state["selected_skill"]:
                        break

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Skill selection completed in {latency_ms:.1f}ms (client mode)")

            selected = selection_state["selected_skill"] is not None
            return SkillSelection(
                skill=selection_state["selected_skill"],
                confidence=1.0 if selected else 0.0,
                reasoning=f"Selected: {selection_state['selected_skill']}" if selected else "No selection",
                raw_response=raw_response,
            )

        except Exception as e:
            logger.error(f"Error during skill selection: {e}")
            raise AgentError(
                message=str(e),
                agent_name=self.name,
            ) from e

    async def close(self) -> None:
        """Clean up resources."""
        self._client = None
        await super().close()
