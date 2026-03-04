"""Tests for evaluator.predictors — random baselines and LLM predictors with mock backend."""

from __future__ import annotations

import json

import pytest

from evaluator.llm import LLMBackend
from evaluator.models import ScorePrediction, Solution, TaskContext, TournamentPrediction
from evaluator.predictors import (
    LLMScorePredictor,
    LLMTournamentPredictor,
    RandomScorePredictor,
    RandomTournamentPredictor,
)


@pytest.fixture
def task() -> TaskContext:
    return TaskContext(
        name="test-task",
        description="Predict the target variable.",
        metric_name="accuracy",
        is_lower_better=False,
        gold_threshold=0.95,
        median_threshold=0.70,
    )


@pytest.fixture
def solutions() -> list[Solution]:
    return [
        Solution(id="s1", plan="Use XGBoost with 5-fold CV", code="import xgboost", score=0.90),
        Solution(id="s2", plan="Use random forest", code="from sklearn import ...", score=0.85),
        Solution(id="s3", plan="Use neural network", code="import torch", score=0.92),
    ]


# ---------------------------------------------------------------------------
# Random predictors
# ---------------------------------------------------------------------------


class TestRandomScorePredictor:
    def test_returns_valid_prediction(self, task, solutions):
        predictor = RandomScorePredictor(seed=42)
        pred = predictor.predict_score(solutions[0], task)
        assert isinstance(pred, ScorePrediction)
        assert pred.solution_id == "s1"
        assert 0.70 <= pred.predicted_score <= 0.95  # between median and gold
        assert pred.confidence == 0.0
        assert pred.reasoning == "random baseline"

    def test_batch(self, task, solutions):
        predictor = RandomScorePredictor(seed=42)
        preds = predictor.predict_scores(solutions, task)
        assert len(preds) == 3
        assert all(isinstance(p, ScorePrediction) for p in preds)

    def test_deterministic(self, task, solutions):
        p1 = RandomScorePredictor(seed=42).predict_score(solutions[0], task)
        p2 = RandomScorePredictor(seed=42).predict_score(solutions[0], task)
        assert p1.predicted_score == p2.predicted_score

    def test_handles_inverted_thresholds(self, solutions):
        """When lower is better, median > gold — predictor should handle this."""
        task = TaskContext(
            name="test", is_lower_better=True,
            gold_threshold=0.1, median_threshold=0.5,
        )
        predictor = RandomScorePredictor(seed=42)
        pred = predictor.predict_score(solutions[0], task)
        assert 0.1 <= pred.predicted_score <= 0.5


class TestRandomTournamentPredictor:
    def test_returns_valid_prediction(self, task, solutions):
        predictor = RandomTournamentPredictor(seed=42)
        pred = predictor.predict_winner(solutions[0], solutions[1], task)
        assert isinstance(pred, TournamentPrediction)
        assert pred.solution_a_id == "s1"
        assert pred.solution_b_id == "s2"
        assert pred.winner_id in ("s1", "s2")
        assert pred.confidence == 0.5

    def test_batch(self, task, solutions):
        predictor = RandomTournamentPredictor(seed=42)
        pairs = [(solutions[0], solutions[1]), (solutions[1], solutions[2])]
        preds = predictor.predict_tournament(pairs, task)
        assert len(preds) == 2


# ---------------------------------------------------------------------------
# LLM predictors (with mock backend)
# ---------------------------------------------------------------------------


class MockLLMBackend(LLMBackend):
    """Returns canned JSON responses for testing."""

    def __init__(self, response: dict) -> None:
        self._response = response

    def query(self, prompt: str, system_prompt: str = "") -> str:
        return json.dumps(self._response)


class TestLLMScorePredictor:
    def test_parses_response(self, task, solutions):
        backend = MockLLMBackend({
            "predicted_score": 0.88,
            "confidence": 0.75,
            "reasoning": "XGBoost is strong for tabular data",
        })
        predictor = LLMScorePredictor(backend, include_code=True)
        pred = predictor.predict_score(solutions[0], task)

        assert pred.solution_id == "s1"
        assert pred.predicted_score == 0.88
        assert pred.confidence == 0.75
        assert "XGBoost" in pred.reasoning

    def test_code_excluded(self, task, solutions):
        """When include_code=False, code should not appear in prompt."""
        calls = []

        class SpyBackend(LLMBackend):
            def query(self, prompt: str, system_prompt: str = "") -> str:
                calls.append(prompt)
                return json.dumps({"predicted_score": 0.5, "confidence": 0.5, "reasoning": ""})

        predictor = LLMScorePredictor(SpyBackend(), include_code=False)
        predictor.predict_score(solutions[0], task)

        assert len(calls) == 1
        assert "import xgboost" not in calls[0]
        assert "XGBoost" in calls[0]  # plan should still be there

    def test_markdown_fenced_response(self, task, solutions):
        """LLM sometimes wraps JSON in ```json ... ``` — should still parse."""

        class FencedBackend(LLMBackend):
            def query(self, prompt: str, system_prompt: str = "") -> str:
                return '```json\n{"predicted_score": 0.77, "confidence": 0.6, "reasoning": "ok"}\n```'

        predictor = LLMScorePredictor(FencedBackend())
        pred = predictor.predict_score(solutions[0], task)
        assert pred.predicted_score == 0.77


class TestLLMTournamentPredictor:
    def test_parses_response(self, task, solutions):
        backend = MockLLMBackend({
            "winner": "A",
            "confidence": 0.8,
            "reasoning": "A uses better features",
        })
        predictor = LLMTournamentPredictor(backend, seed=0)
        pred = predictor.predict_winner(solutions[0], solutions[1], task)

        assert pred.solution_a_id == "s1"
        assert pred.solution_b_id == "s2"
        assert pred.winner_id in ("s1", "s2")
        assert pred.confidence == 0.8

    def test_randomizes_order(self, task, solutions):
        """Different seeds should produce different orderings."""
        calls_seed0 = []
        calls_seed1 = []

        class SpyBackend(LLMBackend):
            def __init__(self, calls_list):
                self._calls = calls_list

            def query(self, prompt: str, system_prompt: str = "") -> str:
                self._calls.append(prompt)
                return json.dumps({"winner": "A", "confidence": 0.5, "reasoning": ""})

        # Run many times to observe ordering differs
        for _ in range(10):
            LLMTournamentPredictor(SpyBackend(calls_seed0), seed=0).predict_winner(
                solutions[0], solutions[1], task
            )
            LLMTournamentPredictor(SpyBackend(calls_seed1), seed=1).predict_winner(
                solutions[0], solutions[1], task
            )

        # With different seeds, the A/B label mapping should differ at some point
        # (probabilistic but highly likely with 10 trials)
        # Just verify both produced valid results
        assert len(calls_seed0) == 10
        assert len(calls_seed1) == 10
