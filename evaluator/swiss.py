"""Swiss-system tournament for efficient pairwise evaluation.

Instead of exhaustive C(n,2) comparisons, a Swiss tournament matches solutions
with similar standings each round, producing only R * floor(n/2) comparisons
where R = ceil(log2(n)).  For n=30 this is 75 calls vs 435 — an 83% reduction.

The Swiss system is a scheduling strategy that wraps any TournamentPredictor.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from evaluator.models import Solution, TaskContext, TournamentPrediction
from evaluator.predictors import TournamentPredictor


@dataclass
class SwissStandings:
    """Tracks cumulative standings for a Swiss tournament.

    Attributes:
        wins: Cumulative win count per solution ID.
        played: Set of frozensets tracking which pairs have already met.
        bye_ids: Set of solution IDs that have received a bye.
    """

    wins: dict[str, int] = field(default_factory=dict)
    played: set[frozenset[str]] = field(default_factory=set)
    bye_ids: set[str] = field(default_factory=set)

    def record_result(self, winner_id: str, loser_id: str) -> None:
        """Record a match result."""
        self.wins.setdefault(winner_id, 0)
        self.wins.setdefault(loser_id, 0)
        self.wins[winner_id] += 1
        self.played.add(frozenset([winner_id, loser_id]))

    def have_played(self, a_id: str, b_id: str) -> bool:
        """Check if two solutions have already played each other."""
        return frozenset([a_id, b_id]) in self.played

    def record_bye(self, solution_id: str) -> None:
        """Record a bye (free win) for a solution."""
        self.wins.setdefault(solution_id, 0)
        self.wins[solution_id] += 1
        self.bye_ids.add(solution_id)


def swiss_pair_round(
    solution_ids: list[str],
    standings: SwissStandings,
    rng: random.Random,
) -> tuple[list[tuple[str, str]], str | None]:
    """Generate pairings for one Swiss round.

    1. Sort by wins descending, break ties with seeded random shuffle.
    2. If odd count: give bye to lowest-ranked without one yet.
    3. Greedy top-down pairing: pair each with next unpaired, unplayed opponent.
    4. Fallback: if no unplayed opponent exists, force re-pair with closest.

    Args:
        solution_ids: List of solution IDs still in the tournament.
        standings: Current Swiss standings.
        rng: Seeded random instance for tie-breaking.

    Returns:
        (pairs, bye_id) where pairs is a list of (id_a, id_b) tuples
        and bye_id is the ID that got a bye (or None).
    """
    if len(solution_ids) < 2:
        return [], None

    # Sort by wins desc; break ties with a shuffled ordering for fairness
    shuffled = list(solution_ids)
    rng.shuffle(shuffled)
    ranked = sorted(shuffled, key=lambda sid: standings.wins.get(sid, 0), reverse=True)

    bye_id: str | None = None
    active = list(ranked)

    # Handle odd count: give bye to lowest-ranked who hasn't had one
    if len(active) % 2 == 1:
        # Search from bottom for someone without a bye
        for i in range(len(active) - 1, -1, -1):
            if active[i] not in standings.bye_ids:
                bye_id = active.pop(i)
                break
        else:
            # Everyone has had a bye — give it to the bottom player
            bye_id = active.pop()

    # Greedy top-down pairing
    paired: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for sid in active:
        if sid in paired:
            continue
        # Find best opponent: first unpaired, unplayed
        best = None
        for candidate in active:
            if candidate == sid or candidate in paired:
                continue
            if not standings.have_played(sid, candidate):
                best = candidate
                break
        # Fallback: closest-ranked unpaired (even if already played)
        if best is None:
            for candidate in active:
                if candidate == sid or candidate in paired:
                    continue
                best = candidate
                break
        if best is not None:
            pairs.append((sid, best))
            paired.add(sid)
            paired.add(best)

    return pairs, bye_id


def run_swiss_tournament(
    predictor: TournamentPredictor,
    solutions: list[Solution],
    task: TaskContext,
    num_rounds: int | None = None,
    seed: int | None = None,
) -> list[TournamentPrediction]:
    """Run a Swiss-system tournament using the given predictor.

    Filters to non-buggy solutions with scores, runs ceil(log2(n)) rounds of
    Swiss pairings, and returns all TournamentPrediction objects (flat list,
    compatible with tournament_accuracy/tournament_rank_correlation).

    Args:
        predictor: Any TournamentPredictor (LLM, random, etc.).
        solutions: Solutions with ground-truth scores (scores stripped before predicting).
        task: Task context.
        num_rounds: Number of Swiss rounds (default: ceil(log2(n))).
        seed: Random seed for pairing tie-breaks.

    Returns:
        Flat list of all TournamentPrediction objects across all rounds.
    """
    # Filter to scored, non-buggy solutions
    scored = [s for s in solutions if not s.is_buggy and s.score is not None]
    if len(scored) < 2:
        return []

    if num_rounds is None:
        num_rounds = math.ceil(math.log2(len(scored)))

    rng = random.Random(seed)
    standings = SwissStandings()
    solution_ids = [s.id for s in scored]

    # Initialize win counts
    for sid in solution_ids:
        standings.wins[sid] = 0

    # Build lookup for blind copies (strip scores)
    blind_lookup: dict[str, Solution] = {}
    for s in scored:
        blind_lookup[s.id] = replace(s, score=None)

    all_predictions: list[TournamentPrediction] = []

    for round_num in range(num_rounds):
        pairs, bye_id = swiss_pair_round(solution_ids, standings, rng)

        if bye_id is not None:
            standings.record_bye(bye_id)

        if not pairs:
            break

        # Build blind solution pairs for the predictor
        blind_pairs = [
            (blind_lookup[a_id], blind_lookup[b_id])
            for a_id, b_id in pairs
        ]

        # Run predictions for this round
        round_preds = predictor.predict_tournament(blind_pairs, task)
        all_predictions.extend(round_preds)

        # Update standings from results
        for pred in round_preds:
            winner = pred.winner_id
            loser = (
                pred.solution_b_id
                if winner == pred.solution_a_id
                else pred.solution_a_id
            )
            standings.record_result(winner, loser)

    return all_predictions
