"""Scenario generation module.

Provides LLM-powered scenario generation from task descriptions.

Example:
    ```python
    from skills_arena.generator import LLMGenerator, MockGenerator

    # LLM-powered generation
    generator = LLMGenerator()
    scenarios = await generator.generate(
        task=Task(description="web search"),
        skills=[skill_a, skill_b],
        count=50,
    )

    # Mock generation for testing
    mock = MockGenerator()
    scenarios = await mock.generate(task, skills, count=10)
    ```
"""

from .base import BaseGenerator
from .llm import LLMGenerator, MockGenerator

__all__ = [
    "BaseGenerator",
    "LLMGenerator",
    "MockGenerator",
]
