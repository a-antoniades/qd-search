"""Tests for evaluator.metrics — MAE, RMSE, Spearman, tournament accuracy."""

from __future__ import annotations

import math

import pytest

from evaluator.metrics import (
    BenchmarkResult,
    _compute_ranks,
    mae,
    rank_correlation,
    rmse,
    tournament_accuracy,
    tournament_rank_correlation,
)
from evaluator.models import ScorePrediction, TournamentPrediction


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------


class TestMAE:
    def test_perfect(self):
        preds = [
            ScorePrediction("s1", 0.5),
            ScorePrediction("s2", 0.8),
        ]
        actuals = {"s1": 0.5, "s2": 0.8}
        assert mae(preds, actuals) == pytest.approx(0.0)

    def test_known_error(self):
        preds = [
            ScorePrediction("s1", 0.6),  # error = 0.1
            ScorePrediction("s2", 1.0),  # error = 0.2
        ]
        actuals = {"s1": 0.5, "s2": 0.8}
        assert mae(preds, actuals) == pytest.approx(0.15)

    def test_missing_actual_skipped(self):
        preds = [
            ScorePrediction("s1", 0.5),
            ScorePrediction("s_missing", 0.9),
        ]
        actuals = {"s1": 0.5}
        assert mae(preds, actuals) == pytest.approx(0.0)

    def test_empty(self):
        assert mae([], {}) == 0.0


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


class TestRMSE:
    def test_perfect(self):
        preds = [ScorePrediction("s1", 0.5)]
        assert rmse(preds, {"s1": 0.5}) == pytest.approx(0.0)

    def test_known_error(self):
        preds = [
            ScorePrediction("s1", 1.0),  # error^2 = 1.0
            ScorePrediction("s2", 3.0),  # error^2 = 1.0
        ]
        actuals = {"s1": 0.0, "s2": 2.0}
        # RMSE = sqrt((1 + 1) / 2) = 1.0
        assert rmse(preds, actuals) == pytest.approx(1.0)

    def test_empty(self):
        assert rmse([], {}) == 0.0


# ---------------------------------------------------------------------------
# Rank correlation (Spearman's rho)
# ---------------------------------------------------------------------------


class TestRankCorrelation:
    def test_perfect_positive(self):
        """Predictions in same order as actuals → rho = 1.0."""
        preds = [
            ScorePrediction("s1", 1.0),
            ScorePrediction("s2", 2.0),
            ScorePrediction("s3", 3.0),
        ]
        actuals = {"s1": 10.0, "s2": 20.0, "s3": 30.0}
        assert rank_correlation(preds, actuals) == pytest.approx(1.0)

    def test_perfect_negative(self):
        """Predictions in reverse order → rho = -1.0."""
        preds = [
            ScorePrediction("s1", 3.0),
            ScorePrediction("s2", 2.0),
            ScorePrediction("s3", 1.0),
        ]
        actuals = {"s1": 10.0, "s2": 20.0, "s3": 30.0}
        assert rank_correlation(preds, actuals) == pytest.approx(-1.0)

    def test_no_correlation(self):
        """Known arrangement with rho = 0."""
        # Ranks: pred=[1,2,3,4,5], actual=[3,4,1,5,2]
        # d^2 = (1-3)^2 + (2-4)^2 + (3-1)^2 + (4-5)^2 + (5-2)^2 = 4+4+4+1+9 = 22
        # rho = 1 - 6*22/(5*24) = 1 - 132/120 = -0.1
        preds = [
            ScorePrediction("s1", 1.0),
            ScorePrediction("s2", 2.0),
            ScorePrediction("s3", 3.0),
            ScorePrediction("s4", 4.0),
            ScorePrediction("s5", 5.0),
        ]
        actuals = {"s1": 30.0, "s2": 40.0, "s3": 10.0, "s4": 50.0, "s5": 20.0}
        assert rank_correlation(preds, actuals) == pytest.approx(-0.1)

    def test_single_element(self):
        preds = [ScorePrediction("s1", 1.0)]
        assert rank_correlation(preds, {"s1": 1.0}) == 0.0

    def test_empty(self):
        assert rank_correlation([], {}) == 0.0

    def test_ties(self):
        """Tied values should get average ranks."""
        preds = [
            ScorePrediction("s1", 1.0),
            ScorePrediction("s2", 1.0),
            ScorePrediction("s3", 2.0),
        ]
        actuals = {"s1": 10.0, "s2": 10.0, "s3": 20.0}
        # Both have same ranking → rho should be 1.0
        assert rank_correlation(preds, actuals) == pytest.approx(1.0)


class TestComputeRanks:
    def test_no_ties(self):
        assert _compute_ranks([30, 10, 20]) == [3.0, 1.0, 2.0]

    def test_all_tied(self):
        assert _compute_ranks([5, 5, 5]) == [2.0, 2.0, 2.0]

    def test_partial_tie(self):
        # 10, 20, 20, 30 → ranks 1, 2.5, 2.5, 4
        assert _compute_ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


# ---------------------------------------------------------------------------
# Tournament rank correlation
# ---------------------------------------------------------------------------


class TestTournamentRankCorrelation:
    def test_perfect_ranking(self):
        """All matchups predicted correctly → rho should be 1.0."""
        # 3 solutions: s1=0.3, s2=0.5, s3=0.8 (higher is better)
        # Correct winners: s3>s2, s3>s1, s2>s1
        preds = [
            TournamentPrediction("s1", "s2", winner_id="s2"),
            TournamentPrediction("s1", "s3", winner_id="s3"),
            TournamentPrediction("s2", "s3", winner_id="s3"),
        ]
        actuals = {"s1": 0.3, "s2": 0.5, "s3": 0.8}
        # Win counts: s1=0, s2=1, s3=2 → ranking matches actual ranking
        rho = tournament_rank_correlation(preds, actuals)
        assert rho == pytest.approx(1.0)

    def test_reversed_ranking(self):
        """All matchups predicted wrong → rho should be -1.0."""
        preds = [
            TournamentPrediction("s1", "s2", winner_id="s1"),
            TournamentPrediction("s1", "s3", winner_id="s1"),
            TournamentPrediction("s2", "s3", winner_id="s2"),
        ]
        actuals = {"s1": 0.3, "s2": 0.5, "s3": 0.8}
        # Win counts: s1=2, s2=1, s3=0 → reversed actual ranking
        rho = tournament_rank_correlation(preds, actuals)
        assert rho == pytest.approx(-1.0)

    def test_empty(self):
        assert tournament_rank_correlation([], {}) == 0.0

    def test_single_pair(self):
        """Only one pair → fewer than 2 distinct solutions shouldn't happen,
        but 2 solutions is fine."""
        preds = [TournamentPrediction("s1", "s2", winner_id="s2")]
        actuals = {"s1": 0.3, "s2": 0.8}
        # Win counts: s1=0, s2=1 → correct order → rho = 1.0
        rho = tournament_rank_correlation(preds, actuals)
        assert rho == pytest.approx(1.0)

    def test_missing_actuals_filtered(self):
        """Solutions without actual scores are excluded."""
        preds = [
            TournamentPrediction("s1", "s2", winner_id="s2"),
            TournamentPrediction("s1", "s_missing", winner_id="s1"),
        ]
        actuals = {"s1": 0.3, "s2": 0.8}
        # s_missing filtered out; win counts: s1=1, s2=1 (tied)
        # Tied predicted ranks (1.5, 1.5), actual ranks (1, 2)
        # d^2 = 0.25 + 0.25 = 0.5, rho = 1 - 6*0.5/(2*3) = 0.5
        rho = tournament_rank_correlation(preds, actuals)
        assert rho == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tournament accuracy
# ---------------------------------------------------------------------------


class TestTournamentAccuracy:
    def test_all_correct(self):
        preds = [
            TournamentPrediction("s1", "s2", winner_id="s1"),
            TournamentPrediction("s2", "s3", winner_id="s3"),
        ]
        actuals = {"s1": 0.9, "s2": 0.5, "s3": 0.8}
        assert tournament_accuracy(preds, actuals) == pytest.approx(1.0)

    def test_all_wrong(self):
        preds = [
            TournamentPrediction("s1", "s2", winner_id="s2"),
        ]
        actuals = {"s1": 0.9, "s2": 0.5}
        assert tournament_accuracy(preds, actuals) == pytest.approx(0.0)

    def test_ties_skipped(self):
        preds = [
            TournamentPrediction("s1", "s2", winner_id="s1"),
        ]
        actuals = {"s1": 0.5, "s2": 0.5}  # tie
        assert tournament_accuracy(preds, actuals) == 0.0

    def test_lower_is_better(self):
        preds = [
            TournamentPrediction("s1", "s2", winner_id="s1"),
        ]
        actuals = {"s1": 0.1, "s2": 0.5}  # s1 wins when lower is better
        assert tournament_accuracy(preds, actuals, is_lower_better=True) == pytest.approx(1.0)

    def test_empty(self):
        assert tournament_accuracy([], {}) == 0.0

    def test_missing_actual(self):
        preds = [
            TournamentPrediction("s1", "s_missing", winner_id="s1"),
        ]
        actuals = {"s1": 0.9}
        assert tournament_accuracy(preds, actuals) == 0.0


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_defaults(self):
        r = BenchmarkResult()
        assert r.predictor_name == ""
        assert r.task_name == ""
        assert r.num_solutions == 0
        assert r.mae == 0.0
        assert r.extra == {}

    def test_construction(self):
        r = BenchmarkResult(
            predictor_name="random",
            task_name="tabular",
            num_solutions=10,
            mae=0.15,
            rmse=0.20,
            rank_correlation=0.3,
            tournament_accuracy=0.55,
        )
        assert r.predictor_name == "random"
        assert r.num_solutions == 10
