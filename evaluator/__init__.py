"""evaluator — Benchmark LLM-based prediction of ML solution quality.

Public API re-exports:

- Models: :class:`TaskContext`, :class:`Solution`, :class:`ScorePrediction`,
  :class:`TournamentPrediction`
- Predictors: :class:`ScorePredictor`, :class:`TournamentPredictor`,
  :class:`LLMScorePredictor`, :class:`LLMTournamentPredictor`,
  :class:`RandomScorePredictor`, :class:`RandomTournamentPredictor`
- Metrics: :func:`mae`, :func:`rmse`, :func:`rank_correlation`,
  :func:`tournament_accuracy`, :class:`BenchmarkResult`
- LLM: :class:`LLMBackend`, :class:`LiteLLMBackend`
- Loaders: :func:`load_journal`, :func:`discover_runs`, :func:`load_task_description`
- Results: :func:`save_predictions`, :func:`save_tournament_predictions`,
  :func:`save_benchmark_results`, :func:`load_predictions`,
  :func:`load_benchmark_results`
- Benchmark: :func:`benchmark_score_predictor`,
  :func:`benchmark_tournament_predictor`, :func:`benchmark_swiss_tournament`,
  :func:`run_benchmark`
- Swiss: :func:`run_swiss_tournament`, :class:`SwissStandings`
"""

from evaluator.benchmark import (
    benchmark_score_predictor,
    benchmark_swiss_tournament,
    benchmark_tournament_predictor,
    run_benchmark,
)
from evaluator.llm import LiteLLMBackend, LLMBackend
from evaluator.loaders import discover_runs, load_journal, load_task_description
from evaluator.metrics import (
    BenchmarkResult,
    mae,
    rank_correlation,
    rmse,
    tournament_accuracy,
    tournament_rank_correlation,
)
from evaluator.models import ScorePrediction, Solution, TaskContext, TournamentPrediction
from evaluator.predictors import (
    EnsembleScorePredictor,
    EnsembleTournamentPredictor,
    LLMScorePredictor,
    LLMTournamentPredictor,
    RandomScorePredictor,
    RandomTournamentPredictor,
    ScorePredictor,
    TournamentPredictor,
)
from evaluator.swiss import SwissStandings, run_swiss_tournament
from evaluator.results import (
    load_benchmark_results,
    load_predictions,
    save_benchmark_results,
    save_predictions,
    save_tournament_predictions,
)

__all__ = [
    "BenchmarkResult",
    "EnsembleScorePredictor",
    "EnsembleTournamentPredictor",
    "SwissStandings",
    "LLMBackend",
    "LLMScorePredictor",
    "LLMTournamentPredictor",
    "LiteLLMBackend",
    "RandomScorePredictor",
    "RandomTournamentPredictor",
    "ScorePrediction",
    "ScorePredictor",
    "Solution",
    "TaskContext",
    "TournamentPrediction",
    "TournamentPredictor",
    "benchmark_score_predictor",
    "benchmark_swiss_tournament",
    "benchmark_tournament_predictor",
    "discover_runs",
    "load_benchmark_results",
    "load_journal",
    "load_predictions",
    "load_task_description",
    "mae",
    "rank_correlation",
    "rmse",
    "run_benchmark",
    "run_swiss_tournament",
    "save_benchmark_results",
    "save_predictions",
    "save_tournament_predictions",
    "tournament_accuracy",
    "tournament_rank_correlation",
]
