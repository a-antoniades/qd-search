"""Evaluation metrics for score and tournament predictions.

All implemented in pure Python (no numpy/scipy dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from evaluator.models import ScorePrediction, TournamentPrediction


def mae(predictions: list[ScorePrediction], actuals: dict[str, float]) -> float:
    """Mean absolute error between predicted and actual scores.

    Args:
        predictions: List of score predictions.
        actuals: Mapping from solution_id to actual score.

    Returns:
        MAE value, or 0.0 if no valid predictions.
    """
    errors = []
    for p in predictions:
        actual = actuals.get(p.solution_id)
        if actual is not None:
            errors.append(abs(p.predicted_score - actual))
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def rmse(predictions: list[ScorePrediction], actuals: dict[str, float]) -> float:
    """Root mean squared error between predicted and actual scores.

    Args:
        predictions: List of score predictions.
        actuals: Mapping from solution_id to actual score.

    Returns:
        RMSE value, or 0.0 if no valid predictions.
    """
    squared_errors = []
    for p in predictions:
        actual = actuals.get(p.solution_id)
        if actual is not None:
            squared_errors.append((p.predicted_score - actual) ** 2)
    if not squared_errors:
        return 0.0
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def rank_correlation(
    predictions: list[ScorePrediction], actuals: dict[str, float]
) -> float:
    """Spearman's rank correlation between predicted and actual scores.

    Pure Python implementation (no scipy). Measures whether the predictor
    ranks solutions in the correct order.

    Args:
        predictions: List of score predictions.
        actuals: Mapping from solution_id to actual score.

    Returns:
        Spearman's rho in [-1, 1], or 0.0 if fewer than 2 valid predictions.
    """
    # Collect paired values
    pairs = []
    for p in predictions:
        actual = actuals.get(p.solution_id)
        if actual is not None:
            pairs.append((p.predicted_score, actual))

    n = len(pairs)
    if n < 2:
        return 0.0

    predicted_vals = [x[0] for x in pairs]
    actual_vals = [x[1] for x in pairs]

    pred_ranks = _compute_ranks(predicted_vals)
    actual_ranks = _compute_ranks(actual_vals)

    # Spearman's rho = 1 - 6 * sum(d^2) / (n * (n^2 - 1))
    d_squared_sum = sum(
        (pr - ar) ** 2 for pr, ar in zip(pred_ranks, actual_ranks)
    )
    return 1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1))


def _compute_ranks(values: list[float]) -> list[float]:
    """Compute ranks with average tie-breaking.

    Rank 1 = smallest value.
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        # Find tied group
        j = i + 1
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        # Average rank for tied group (1-based)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j

    return ranks


def tournament_rank_correlation(
    predictions: list[TournamentPrediction],
    actuals: dict[str, float],
) -> float:
    """Spearman rho of win-count ranking vs actual score ranking.

    Derives a ranking from tournament results (by win count), then
    computes Spearman correlation against the actual score ranking.

    Args:
        predictions: List of tournament predictions.
        actuals: Mapping from solution_id to actual score.

    Returns:
        Spearman's rho in [-1, 1], or 0.0 if fewer than 2 solutions.
    """
    # Count wins per solution
    win_counts: dict[str, int] = {}
    for p in predictions:
        if p.solution_a_id not in win_counts:
            win_counts[p.solution_a_id] = 0
        if p.solution_b_id not in win_counts:
            win_counts[p.solution_b_id] = 0
        if p.winner_id in win_counts:
            win_counts[p.winner_id] += 1

    # Filter to solutions that appear in both win_counts and actuals
    common_ids = [sid for sid in win_counts if sid in actuals]
    if len(common_ids) < 2:
        return 0.0

    # Convert to ScorePrediction objects (predicted_score = win_count)
    score_preds = [
        ScorePrediction(solution_id=sid, predicted_score=float(win_counts[sid]))
        for sid in common_ids
    ]
    common_actuals = {sid: actuals[sid] for sid in common_ids}

    return rank_correlation(score_preds, common_actuals)


def tournament_accuracy(
    predictions: list[TournamentPrediction],
    actuals: dict[str, float],
    is_lower_better: bool = False,
) -> float:
    """Fraction of correct winner predictions.

    Ties (equal actual scores) are skipped.

    Args:
        predictions: List of tournament predictions.
        actuals: Mapping from solution_id to actual score.
        is_lower_better: Whether lower metric values are better.

    Returns:
        Accuracy in [0, 1], or 0.0 if no valid non-tie predictions.
    """
    correct = 0
    total = 0
    for p in predictions:
        score_a = actuals.get(p.solution_a_id)
        score_b = actuals.get(p.solution_b_id)
        if score_a is None or score_b is None:
            continue
        if score_a == score_b:
            continue  # skip ties
        if is_lower_better:
            actual_winner = p.solution_a_id if score_a < score_b else p.solution_b_id
        else:
            actual_winner = p.solution_a_id if score_a > score_b else p.solution_b_id
        total += 1
        if p.winner_id == actual_winner:
            correct += 1

    if total == 0:
        return 0.0
    return correct / total


@dataclass
class BenchmarkResult:
    """Holds aggregate metrics for one predictor-task benchmark run.

    Attributes:
        predictor_name: Name/identifier of the predictor.
        task_name: Name of the task being evaluated.
        num_solutions: Number of solutions in the benchmark.
        mae: Mean absolute error (score prediction).
        rmse: Root mean squared error (score prediction).
        rank_correlation: Spearman's rho (score prediction).
        tournament_accuracy: Fraction correct (tournament prediction).
        extra: Additional metadata.
    """

    predictor_name: str = ""
    task_name: str = ""
    num_solutions: int = 0
    mae: float = 0.0
    rmse: float = 0.0
    rank_correlation: float = 0.0
    tournament_accuracy: float = 0.0
    extra: dict = field(default_factory=dict)
