"""Benchmark harness: load solutions, run predictors, compute metrics, save results."""

from __future__ import annotations

import itertools
import random
from dataclasses import replace
from pathlib import Path

from evaluator.loaders import discover_runs, load_journal
from evaluator.metrics import (
    BenchmarkResult,
    mae,
    rank_correlation,
    rmse,
    tournament_accuracy,
    tournament_rank_correlation,
)
from evaluator.models import Solution, TaskContext
from evaluator.predictors import ScorePredictor, TournamentPredictor
from evaluator.results import (
    save_benchmark_results,
    save_predictions,
    save_tournament_predictions,
)
from evaluator.swiss import run_swiss_tournament


def _blind_copy(solution: Solution) -> Solution:
    """Create a copy of a solution with score stripped (prevents data leakage)."""
    return replace(solution, score=None)


def _scored_solutions(solutions: list[Solution]) -> list[Solution]:
    """Filter to non-buggy solutions with valid scores."""
    return [s for s in solutions if not s.is_buggy and s.score is not None]


def benchmark_score_predictor(
    predictor: ScorePredictor,
    solutions: list[Solution],
    task: TaskContext,
    predictor_name: str = "",
    output_dir: str | Path | None = None,
) -> BenchmarkResult:
    """Run a score predictor on solutions and compute metrics.

    Filters to non-buggy solutions with scores, creates blind copies,
    runs predictions, and computes MAE/RMSE/Spearman rho.

    Args:
        predictor: The score predictor to benchmark.
        solutions: List of solutions (with ground-truth scores).
        task: Task context for the solutions.
        predictor_name: Name for the predictor in results.
        output_dir: Optional directory to save raw predictions.

    Returns:
        BenchmarkResult with score prediction metrics.
    """
    scored = _scored_solutions(solutions)
    if not scored:
        return BenchmarkResult(
            predictor_name=predictor_name,
            task_name=task.name,
            num_solutions=0,
        )

    # Build ground truth lookup
    actuals = {s.id: s.score for s in scored}

    # Create blind copies and predict
    blind = [_blind_copy(s) for s in scored]
    predictions = predictor.predict_scores(blind, task)

    # Compute metrics
    result = BenchmarkResult(
        predictor_name=predictor_name,
        task_name=task.name,
        num_solutions=len(scored),
        mae=mae(predictions, actuals),
        rmse=rmse(predictions, actuals),
        rank_correlation=rank_correlation(predictions, actuals),
    )

    # Save raw predictions if output_dir specified
    if output_dir is not None:
        name = predictor_name or "predictor"
        save_predictions(
            predictions,
            scored,
            Path(output_dir) / f"score_{name}_{task.name}.jsonl",
        )

    return result


def benchmark_tournament_predictor(
    predictor: TournamentPredictor,
    solutions: list[Solution],
    task: TaskContext,
    predictor_name: str = "",
    max_pairs: int = 500,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> BenchmarkResult:
    """Run a tournament predictor on solution pairs and compute accuracy.

    Generates all pairs from scored solutions, subsamples if needed,
    creates blind copies, runs predictions, and computes accuracy.

    Args:
        predictor: The tournament predictor to benchmark.
        solutions: List of solutions (with ground-truth scores).
        task: Task context for the solutions.
        predictor_name: Name for the predictor in results.
        max_pairs: Maximum number of pairs to evaluate (subsamples if exceeded).
        seed: Random seed for subsampling.
        output_dir: Optional directory to save raw predictions.

    Returns:
        BenchmarkResult with tournament accuracy.
    """
    scored = _scored_solutions(solutions)
    if len(scored) < 2:
        return BenchmarkResult(
            predictor_name=predictor_name,
            task_name=task.name,
            num_solutions=len(scored),
        )

    # Build ground truth lookup
    actuals = {s.id: s.score for s in scored}

    # Generate all pairs
    all_pairs = list(itertools.combinations(scored, 2))

    # Subsample if too many pairs
    if len(all_pairs) > max_pairs:
        rng = random.Random(seed)
        all_pairs = rng.sample(all_pairs, max_pairs)

    # Create blind copies for each pair
    blind_pairs = [(_blind_copy(a), _blind_copy(b)) for a, b in all_pairs]
    predictions = predictor.predict_tournament(blind_pairs, task)

    # Compute metrics
    acc = tournament_accuracy(predictions, actuals, is_lower_better=task.is_lower_better)
    rho = tournament_rank_correlation(predictions, actuals)

    result = BenchmarkResult(
        predictor_name=predictor_name,
        task_name=task.name,
        num_solutions=len(scored),
        rank_correlation=rho,
        tournament_accuracy=acc,
        extra={"num_pairs": len(all_pairs)},
    )

    # Save raw predictions
    if output_dir is not None:
        name = predictor_name or "predictor"
        save_tournament_predictions(
            predictions,
            scored,
            Path(output_dir) / f"tournament_{name}_{task.name}.jsonl",
        )

    return result


def benchmark_swiss_tournament(
    predictor: TournamentPredictor,
    solutions: list[Solution],
    task: TaskContext,
    predictor_name: str = "",
    num_rounds: int | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> BenchmarkResult:
    """Run a Swiss-system tournament and compute metrics.

    Uses run_swiss_tournament() for efficient pairing, then computes
    tournament accuracy and rank correlation on the results.

    Args:
        predictor: The tournament predictor to benchmark.
        solutions: List of solutions (with ground-truth scores).
        task: Task context for the solutions.
        predictor_name: Name for the predictor in results.
        num_rounds: Number of Swiss rounds (default: ceil(log2(n))).
        seed: Random seed for pairing.
        output_dir: Optional directory to save raw predictions.

    Returns:
        BenchmarkResult with tournament metrics and swiss metadata.
    """
    scored = _scored_solutions(solutions)
    if len(scored) < 2:
        return BenchmarkResult(
            predictor_name=predictor_name,
            task_name=task.name,
            num_solutions=len(scored),
        )

    actuals = {s.id: s.score for s in scored}

    predictions = run_swiss_tournament(
        predictor, scored, task, num_rounds=num_rounds, seed=seed,
    )

    acc = tournament_accuracy(predictions, actuals, is_lower_better=task.is_lower_better)
    rho = tournament_rank_correlation(predictions, actuals)

    result = BenchmarkResult(
        predictor_name=predictor_name,
        task_name=task.name,
        num_solutions=len(scored),
        rank_correlation=rho,
        tournament_accuracy=acc,
        extra={
            "num_pairs": len(predictions),
            "num_rounds": num_rounds,
            "tournament_mode": "swiss",
        },
    )

    if output_dir is not None:
        name = predictor_name or "predictor"
        save_tournament_predictions(
            predictions,
            scored,
            Path(output_dir) / f"swiss_{name}_{task.name}.jsonl",
        )

    return result


def run_benchmark(
    score_predictors: list[tuple[str, ScorePredictor]] | None = None,
    tournament_predictors: list[tuple[str, TournamentPredictor]] | None = None,
    journal_paths: list[str | Path] | None = None,
    logs_dir: str | Path | None = None,
    prefix: str = "",
    data_dir: str | Path | None = None,
    max_pairs: int = 500,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> list[BenchmarkResult]:
    """Top-level benchmark entry point.

    Discovers journals, loads solutions, runs all predictors, returns results.

    Args:
        score_predictors: List of (name, predictor) tuples for score prediction.
        tournament_predictors: List of (name, predictor) tuples for tournament.
        journal_paths: Explicit list of JOURNAL.jsonl paths. If None, discovers
            from logs_dir.
        logs_dir: Root logs directory to discover journals from.
        prefix: Prefix filter for journal discovery.
        data_dir: MLE-bench data directory for task descriptions.
        max_pairs: Maximum tournament pairs per task.
        seed: Random seed for subsampling.
        output_dir: Directory to save all predictions and results.

    Returns:
        List of BenchmarkResult objects (one per predictor-task combination).
    """
    score_predictors = score_predictors or []
    tournament_predictors = tournament_predictors or []

    # Discover journals
    if journal_paths is None:
        if logs_dir is None:
            raise ValueError("Either journal_paths or logs_dir must be provided")
        journal_paths = discover_runs(logs_dir, prefix)

    results: list[BenchmarkResult] = []

    for jpath in journal_paths:
        solutions = load_journal(jpath, data_dir=data_dir)
        if not solutions:
            continue

        task = solutions[0].task or TaskContext(name="unknown")

        for name, predictor in score_predictors:
            result = benchmark_score_predictor(
                predictor, solutions, task,
                predictor_name=name,
                output_dir=output_dir,
            )
            results.append(result)

        for name, predictor in tournament_predictors:
            result = benchmark_tournament_predictor(
                predictor, solutions, task,
                predictor_name=name,
                max_pairs=max_pairs,
                seed=seed,
                output_dir=output_dir,
            )
            results.append(result)

    # Save aggregate results
    if output_dir is not None and results:
        save_benchmark_results(results, Path(output_dir) / "benchmark_summary.json")

    return results
