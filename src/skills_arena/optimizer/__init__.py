"""Optimizer module for improving skill descriptions.

Uses LLM-powered rewriting based on competition data to improve
how often agents select a skill.

Example:
    ```python
    from skills_arena.optimizer import SkillOptimizer

    optimizer = SkillOptimizer()
    improved_skill = await optimizer.optimize_description(
        skill=my_skill,
        comparison_result=baseline,
        competitors=[competitor_a, competitor_b],
    )
    ```
"""

from .optimizer import SkillOptimizer

__all__ = [
    "SkillOptimizer",
]
