"""ELO rating system for skill rankings.

This module implements the ELO rating system commonly used in chess and other
competitive games. It's adapted for skill selection competitions where skills
compete head-to-head based on agent selection.
"""

from __future__ import annotations


class ELO:
    """ELO rating system for skill rankings.

    The ELO system calculates ratings based on expected vs actual performance.
    After each matchup, both winner and loser have their ratings adjusted.

    Example:
        ```python
        # After a matchup where skill_a (1500) beats skill_b (1400)
        new_a, new_b = ELO.update(1500, 1400, k=32)
        # new_a ≈ 1510, new_b ≈ 1390

        # With a RatingTracker for multiple skills
        tracker = ELO.create_tracker(["skill_a", "skill_b", "skill_c"])
        tracker.record_match("skill_a", "skill_b")  # skill_a wins
        print(tracker.get_rating("skill_a"))  # > 1500
        ```
    """

    DEFAULT_RATING = 1500
    DEFAULT_K = 32

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B.

        The expected score represents the probability of player A winning,
        based on the difference in ratings.

        Args:
            rating_a: ELO rating of player A.
            rating_b: ELO rating of player B.

        Returns:
            Expected score between 0 and 1.

        Example:
            ```python
            # Equal ratings = 50% chance
            ELO.expected_score(1500, 1500)  # 0.5

            # Higher rated player expected to win more
            ELO.expected_score(1600, 1400)  # ~0.76
            ```
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    @staticmethod
    def update(
        winner_elo: int | float,
        loser_elo: int | float,
        k: int = DEFAULT_K,
    ) -> tuple[int, int]:
        """Update ELO ratings after a matchup.

        Uses the standard ELO formula where the winner gains points and
        the loser loses points. The amount depends on the upset factor.

        Args:
            winner_elo: Current ELO rating of the winner.
            loser_elo: Current ELO rating of the loser.
            k: K-factor determining rating volatility (default 32).
                Higher k = more volatile ratings.

        Returns:
            Tuple of (new_winner_elo, new_loser_elo).

        Example:
            ```python
            # Favorite wins (small change)
            new_winner, new_loser = ELO.update(1600, 1400, k=32)
            # new_winner ≈ 1608, new_loser ≈ 1392

            # Upset victory (big change)
            new_winner, new_loser = ELO.update(1400, 1600, k=32)
            # new_winner ≈ 1424, new_loser ≈ 1576
            ```
        """
        expected_winner = ELO.expected_score(winner_elo, loser_elo)
        expected_loser = 1 - expected_winner

        # Winner gets 1.0, loser gets 0.0
        new_winner = round(winner_elo + k * (1.0 - expected_winner))
        new_loser = round(loser_elo + k * (0.0 - expected_loser))

        return new_winner, new_loser

    @staticmethod
    def update_draw(
        rating_a: int | float,
        rating_b: int | float,
        k: int = DEFAULT_K,
    ) -> tuple[int, int]:
        """Update ELO ratings after a draw.

        In a draw, both players get 0.5 score. The underdog gains slightly
        and the favorite loses slightly.

        Args:
            rating_a: Current ELO rating of player A.
            rating_b: Current ELO rating of player B.
            k: K-factor determining rating volatility (default 32).

        Returns:
            Tuple of (new_rating_a, new_rating_b).
        """
        expected_a = ELO.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a

        # Both get 0.5 for a draw
        new_a = round(rating_a + k * (0.5 - expected_a))
        new_b = round(rating_b + k * (0.5 - expected_b))

        return new_a, new_b

    @classmethod
    def create_tracker(
        cls,
        skills: list[str],
        initial_rating: int = DEFAULT_RATING,
        k: int = DEFAULT_K,
    ) -> RatingTracker:
        """Create a rating tracker for multiple skills.

        Args:
            skills: List of skill names to track.
            initial_rating: Starting ELO rating (default 1500).
            k: K-factor for rating updates (default 32).

        Returns:
            RatingTracker instance.
        """
        return RatingTracker(skills, initial_rating, k)


class RatingTracker:
    """Tracks ELO ratings for multiple skills over a series of matchups.

    This class maintains state across multiple matches and provides
    methods to record results and query current standings.

    Example:
        ```python
        tracker = RatingTracker(["skill_a", "skill_b", "skill_c"])

        # Record some matches
        tracker.record_match("skill_a", "skill_b")  # skill_a wins
        tracker.record_match("skill_b", "skill_c")  # skill_b wins
        tracker.record_match("skill_a", "skill_c")  # skill_a wins

        # Get rankings
        rankings = tracker.get_rankings()
        # [("skill_a", 1532), ("skill_b", 1500), ("skill_c", 1468)]

        # Get individual rating
        print(tracker.get_rating("skill_a"))  # 1532
        ```
    """

    def __init__(
        self,
        skills: list[str],
        initial_rating: int = ELO.DEFAULT_RATING,
        k: int = ELO.DEFAULT_K,
    ):
        """Initialize the rating tracker.

        Args:
            skills: List of skill names to track.
            initial_rating: Starting ELO rating (default 1500).
            k: K-factor for rating updates (default 32).
        """
        self.ratings: dict[str, int] = {skill: initial_rating for skill in skills}
        self.k = k
        self.match_history: list[dict[str, str | None]] = []
        self.wins: dict[str, int] = {skill: 0 for skill in skills}
        self.losses: dict[str, int] = {skill: 0 for skill in skills}
        self.draws: dict[str, int] = {skill: 0 for skill in skills}

    def record_match(self, winner: str, loser: str) -> tuple[int, int]:
        """Record a match result and update ratings.

        Args:
            winner: Name of the winning skill.
            loser: Name of the losing skill.

        Returns:
            Tuple of (new_winner_rating, new_loser_rating).

        Raises:
            KeyError: If either skill is not being tracked.
        """
        if winner not in self.ratings:
            raise KeyError(f"Skill '{winner}' is not being tracked")
        if loser not in self.ratings:
            raise KeyError(f"Skill '{loser}' is not being tracked")

        # Update ratings
        new_winner, new_loser = ELO.update(
            self.ratings[winner],
            self.ratings[loser],
            self.k,
        )
        self.ratings[winner] = new_winner
        self.ratings[loser] = new_loser

        # Update stats
        self.wins[winner] += 1
        self.losses[loser] += 1

        # Record history
        self.match_history.append({"winner": winner, "loser": loser})

        return new_winner, new_loser

    def record_draw(self, skill_a: str, skill_b: str) -> tuple[int, int]:
        """Record a draw and update ratings.

        Args:
            skill_a: Name of first skill.
            skill_b: Name of second skill.

        Returns:
            Tuple of (new_rating_a, new_rating_b).

        Raises:
            KeyError: If either skill is not being tracked.
        """
        if skill_a not in self.ratings:
            raise KeyError(f"Skill '{skill_a}' is not being tracked")
        if skill_b not in self.ratings:
            raise KeyError(f"Skill '{skill_b}' is not being tracked")

        # Update ratings
        new_a, new_b = ELO.update_draw(
            self.ratings[skill_a],
            self.ratings[skill_b],
            self.k,
        )
        self.ratings[skill_a] = new_a
        self.ratings[skill_b] = new_b

        # Update stats
        self.draws[skill_a] += 1
        self.draws[skill_b] += 1

        # Record history
        self.match_history.append({"skill_a": skill_a, "skill_b": skill_b, "draw": True})

        return new_a, new_b

    def get_rating(self, skill: str) -> int:
        """Get current rating for a skill.

        Args:
            skill: Skill name.

        Returns:
            Current ELO rating.

        Raises:
            KeyError: If skill is not being tracked.
        """
        if skill not in self.ratings:
            raise KeyError(f"Skill '{skill}' is not being tracked")
        return self.ratings[skill]

    def get_rankings(self) -> list[tuple[str, int]]:
        """Get all skills ranked by ELO rating.

        Returns:
            List of (skill_name, rating) tuples, sorted by rating descending.
        """
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def get_stats(self, skill: str) -> dict[str, int]:
        """Get win/loss/draw statistics for a skill.

        Args:
            skill: Skill name.

        Returns:
            Dictionary with 'wins', 'losses', 'draws', and 'rating'.

        Raises:
            KeyError: If skill is not being tracked.
        """
        if skill not in self.ratings:
            raise KeyError(f"Skill '{skill}' is not being tracked")

        return {
            "rating": self.ratings[skill],
            "wins": self.wins[skill],
            "losses": self.losses[skill],
            "draws": self.draws[skill],
        }

    def add_skill(self, skill: str, initial_rating: int | None = None) -> None:
        """Add a new skill to track.

        Args:
            skill: Skill name.
            initial_rating: Starting rating (defaults to 1500).
        """
        if skill in self.ratings:
            return  # Already tracked

        rating = initial_rating if initial_rating is not None else ELO.DEFAULT_RATING
        self.ratings[skill] = rating
        self.wins[skill] = 0
        self.losses[skill] = 0
        self.draws[skill] = 0
