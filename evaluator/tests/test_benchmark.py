"""Tests for evaluator.benchmark — integration tests with synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.benchmark import (
    _blind_copy,
    _scored_solutions,
    benchmark_score_predictor,
    benchmark_tournament_predictor,
)
from evaluator.metrics import BenchmarkResult
from evaluator.models import Solution, TaskContext
from evaluator.predictors import RandomScorePredictor, RandomTournamentPredictor


@pytest.fixture
def task() -> TaskContext:
    return TaskContext(
        name="test-task",
        metric_name="accuracy",
        is_lower_better=False,
        gold_threshold=0.95,
        median_threshold=0.70,
    )


@pytest.fixture
def solutions(task) -> list[Solution]:
    return [
        Solution(id="s1", plan="XGBoost", score=0.90, is_buggy=False, task=task),
        Solution(id="s2", plan="Random Forest", score=0.85, is_buggy=False, task=task),
        Solution(id="s3", plan="Neural Net", score=0.92, is_buggy=False, task=task),
        Solution(id="s4", plan="Buggy attempt", score=None, is_buggy=True, task=task),
        Solution(id="s5", plan="Linear Reg", score=0.75, is_buggy=False, task=task),
    ]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestBlindCopy:
    def test_score_stripped(self, solutions):
        blind = _blind_copy(solutions[0])
        assert blind.score is None
        assert blind.id == "s1"
        assert blind.plan == "XGBoost"

    def test_original_unchanged(self, solutions):
        _blind_copy(solutions[0])
        assert solutions[0].score == 0.90


class TestScoredSolutions:
    def test_filters_correctly(self, solutions):
        scored = _scored_solutions(solutions)
        assert len(scored) == 4  # s1, s2, s3, s5 (not s4 which is buggy)
        ids = [s.id for s in scored]
        assert "s4" not in ids

    def test_empty(self):
        assert _scored_solutions([]) == []


# ---------------------------------------------------------------------------
# Score predictor benchmark
# ---------------------------------------------------------------------------


class TestBenchmarkScorePredictor:
    def test_basic(self, task, solutions):
        predictor = RandomScorePredictor(seed=42)
        result = benchmark_score_predictor(
            predictor, solutions, task, predictor_name="random"
        )

        assert isinstance(result, BenchmarkResult)
        assert result.predictor_name == "random"
        assert result.task_name == "test-task"
        assert result.num_solutions == 4  # 4 valid scored solutions
        assert result.mae > 0  # random predictor won't be perfect
        assert result.rmse > 0
        # Spearman rho should be in [-1, 1]
        assert -1.0 <= result.rank_correlation <= 1.0

    def test_empty_solutions(self, task):
        predictor = RandomScorePredictor(seed=42)
        result = benchmark_score_predictor(predictor, [], task)
        assert result.num_solutions == 0
        assert result.mae == 0.0

    def test_all_buggy(self, task):
        buggy = [Solution(id="b1", is_buggy=True, task=task)]
        predictor = RandomScorePredictor(seed=42)
        result = benchmark_score_predictor(predictor, buggy, task)
        assert result.num_solutions == 0

    def test_saves_output(self, task, solutions, tmp_path):
        predictor = RandomScorePredictor(seed=42)
        result = benchmark_score_predictor(
            predictor, solutions, task,
            predictor_name="random",
            output_dir=tmp_path,
        )
        expected_path = tmp_path / "score_random_test-task.jsonl"
        assert expected_path.exists()

        # Verify file contents
        with open(expected_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 4
        assert all("predicted_score" in l for l in lines)
        assert all("actual_score" in l for l in lines)


# ---------------------------------------------------------------------------
# Tournament predictor benchmark
# ---------------------------------------------------------------------------


class TestBenchmarkTournamentPredictor:
    def test_basic(self, task, solutions):
        predictor = RandomTournamentPredictor(seed=42)
        result = benchmark_tournament_predictor(
            predictor, solutions, task, predictor_name="random_t"
        )

        assert isinstance(result, BenchmarkResult)
        assert result.predictor_name == "random_t"
        assert result.num_solutions == 4
        # 4 choose 2 = 6 pairs
        assert result.extra["num_pairs"] == 6
        # Tournament accuracy should be between 0 and 1
        assert 0.0 <= result.tournament_accuracy <= 1.0

    def test_max_pairs_subsampling(self, task, solutions):
        predictor = RandomTournamentPredictor(seed=42)
        result = benchmark_tournament_predictor(
            predictor, solutions, task,
            max_pairs=3, seed=42,
        )
        assert result.extra["num_pairs"] == 3

    def test_too_few_solutions(self, task):
        single = [Solution(id="s1", score=0.9, is_buggy=False, task=task)]
        predictor = RandomTournamentPredictor(seed=42)
        result = benchmark_tournament_predictor(predictor, single, task)
        assert result.num_solutions == 1
        assert result.tournament_accuracy == 0.0

    def test_saves_output(self, task, solutions, tmp_path):
        predictor = RandomTournamentPredictor(seed=42)
        benchmark_tournament_predictor(
            predictor, solutions, task,
            predictor_name="random_t",
            output_dir=tmp_path,
        )
        expected_path = tmp_path / "tournament_random_t_test-task.jsonl"
        assert expected_path.exists()


# ---------------------------------------------------------------------------
# Blind copy integration: verify predictors never see ground truth
# ---------------------------------------------------------------------------


class TestDataLeakagePrevention:
    def test_score_predictor_receives_blind_copies(self, task, solutions):
        """Verify that the predictor receives solutions with score=None."""
        received_solutions = []

        class SpyPredictor(RandomScorePredictor):
            def predict_scores(self, sols, task):
                received_solutions.extend(sols)
                return super().predict_scores(sols, task)

        predictor = SpyPredictor(seed=42)
        benchmark_score_predictor(predictor, solutions, task)

        assert len(received_solutions) == 4
        assert all(s.score is None for s in received_solutions)
