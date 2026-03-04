"""Save and load predictions and benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

from evaluator.metrics import BenchmarkResult
from evaluator.models import ScorePrediction, Solution, TournamentPrediction


def save_predictions(
    predictions: list[ScorePrediction],
    solutions: list[Solution],
    path: str | Path,
) -> Path:
    """Write score predictions with ground truth to a JSONL file.

    Each line contains: solution_id, predicted_score, actual_score,
    confidence, reasoning.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build id -> actual score lookup
    actuals = {s.id: s.score for s in solutions if s.score is not None}

    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            record = {
                "solution_id": p.solution_id,
                "predicted_score": p.predicted_score,
                "actual_score": actuals.get(p.solution_id),
                "confidence": p.confidence,
                "reasoning": p.reasoning,
            }
            f.write(json.dumps(record) + "\n")

    return path


def save_tournament_predictions(
    predictions: list[TournamentPrediction],
    solutions: list[Solution],
    path: str | Path,
) -> Path:
    """Write tournament predictions with ground truth to a JSONL file.

    Each line contains: solution_a_id, solution_b_id, predicted_winner,
    actual_winner, confidence, reasoning.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    actuals = {s.id: s.score for s in solutions if s.score is not None}

    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            score_a = actuals.get(p.solution_a_id)
            score_b = actuals.get(p.solution_b_id)
            # Determine actual winner (assumes higher is better by default)
            actual_winner = None
            if score_a is not None and score_b is not None:
                if score_a > score_b:
                    actual_winner = p.solution_a_id
                elif score_b > score_a:
                    actual_winner = p.solution_b_id
                # else: tie, actual_winner stays None

            record = {
                "solution_a_id": p.solution_a_id,
                "solution_b_id": p.solution_b_id,
                "predicted_winner": p.winner_id,
                "actual_winner": actual_winner,
                "confidence": p.confidence,
                "reasoning": p.reasoning,
            }
            f.write(json.dumps(record) + "\n")

    return path


def save_benchmark_results(
    results: list[BenchmarkResult],
    path: str | Path,
) -> Path:
    """Write aggregate BenchmarkResult list to a JSON file.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for r in results:
        records.append({
            "predictor_name": r.predictor_name,
            "task_name": r.task_name,
            "num_solutions": r.num_solutions,
            "mae": r.mae,
            "rmse": r.rmse,
            "rank_correlation": r.rank_correlation,
            "tournament_accuracy": r.tournament_accuracy,
            "extra": r.extra,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return path


def load_predictions(path: str | Path) -> list[dict]:
    """Read saved predictions (score or tournament) from a JSONL file.

    Returns:
        List of dicts, one per prediction line.
    """
    path = Path(path)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_benchmark_results(path: str | Path) -> list[BenchmarkResult]:
    """Read saved benchmark results from a JSON file.

    Returns:
        List of BenchmarkResult objects.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    return [
        BenchmarkResult(
            predictor_name=r["predictor_name"],
            task_name=r["task_name"],
            num_solutions=r["num_solutions"],
            mae=r["mae"],
            rmse=r["rmse"],
            rank_correlation=r["rank_correlation"],
            tournament_accuracy=r["tournament_accuracy"],
            extra=r.get("extra", {}),
        )
        for r in records
    ]
