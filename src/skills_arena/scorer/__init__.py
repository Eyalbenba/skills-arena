"""Scoring module for Skills Arena.

This module provides metrics calculation and ELO rating system for
evaluating skill performance.

Components:
    - Scorer: Main class for calculating metrics from evaluation results
    - ELO: ELO rating system for skill rankings
    - RatingTracker: Tracks ELO ratings across multiple matchups

Example:
    ```python
    from skills_arena.scorer import Scorer, ELO

    # Score a single skill evaluation
    result = Scorer.score_evaluation(skill, selection_results)
    print(f"Score: {result.score}, Grade: {result.grade}")

    # Use ELO system directly
    new_winner, new_loser = ELO.update(1500, 1400)
    ```
"""

from .elo import ELO, RatingTracker
from .metrics import Scorer

__all__ = [
    "Scorer",
    "ELO",
    "RatingTracker",
]
