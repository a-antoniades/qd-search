"""Data models for the evaluator module.

Four frozen dataclasses representing task context, solutions, and predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskContext:
    """Describes a competition task and what "good" performance looks like.

    Attributes:
        name: Competition/task identifier (e.g. "tabular-playground-series-dec-2021").
        description: Full task description text (from description.md).
        metric_name: Name of the evaluation metric (e.g. "RMSLE", "accuracy").
        is_lower_better: Whether lower metric values are better.
        gold_threshold: Score threshold for a gold medal.
        median_threshold: Score threshold for above-median performance.
    """

    name: str
    description: str = ""
    metric_name: str = ""
    is_lower_better: bool = False
    gold_threshold: float | None = None
    median_threshold: float | None = None


@dataclass(frozen=True)
class Solution:
    """A single ML solution to evaluate.

    Attributes:
        id: Unique identifier for this solution (journal node id).
        plan: Text description of the approach/idea.
        code: Python implementation source code.
        score: Ground-truth metric score (None when predicting).
        is_buggy: Whether execution failed.
        exit_code: Process exit code (0 = success).
        task: The task context this solution belongs to.
        operators_used: List of operators applied (draft, debug, improve, etc.).
        analysis: LLM feedback on execution results.
        term_out: Terminal output from execution.
    """

    id: str
    plan: str = ""
    code: str = ""
    score: float | None = None
    is_buggy: bool = False
    exit_code: int = 0
    task: TaskContext | None = None
    operators_used: list[str] = field(default_factory=list)
    analysis: str = ""
    term_out: str = ""


@dataclass(frozen=True)
class ScorePrediction:
    """Output of a score predictor.

    Attributes:
        solution_id: ID of the solution being predicted.
        predicted_score: The predicted metric score.
        confidence: Predictor's self-assessed confidence (0-1).
        reasoning: Explanation for the prediction.
    """

    solution_id: str
    predicted_score: float
    confidence: float = 0.0
    reasoning: str = ""
    prompt: str = ""
    raw_response: str = ""
    individual_scores: tuple[float, ...] = ()
    individual_confidences: tuple[float, ...] = ()
    num_votes: int = 1


@dataclass(frozen=True)
class TournamentPrediction:
    """Output of a tournament predictor.

    Attributes:
        solution_a_id: ID of the first solution.
        solution_b_id: ID of the second solution.
        winner_id: ID of the predicted winner.
        confidence: Predictor's self-assessed confidence (0-1).
        reasoning: Explanation for the prediction.
    """

    solution_a_id: str
    solution_b_id: str
    winner_id: str
    confidence: float = 0.0
    reasoning: str = ""
    prompt: str = ""
    raw_response: str = ""
    individual_winners: tuple[str, ...] = ()
    vote_counts: tuple[tuple[str, int], ...] = ()
    num_votes: int = 1
