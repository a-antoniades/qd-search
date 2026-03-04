"""Tests for evaluator.models — frozen dataclass construction and defaults."""

from __future__ import annotations

import pytest

from evaluator.models import ScorePrediction, Solution, TaskContext, TournamentPrediction


class TestTaskContext:
    def test_minimal(self):
        t = TaskContext(name="test-task")
        assert t.name == "test-task"
        assert t.description == ""
        assert t.metric_name == ""
        assert t.is_lower_better is False
        assert t.gold_threshold is None
        assert t.median_threshold is None

    def test_full(self):
        t = TaskContext(
            name="tabular-playground",
            description="Predict the target",
            metric_name="RMSLE",
            is_lower_better=True,
            gold_threshold=0.95,
            median_threshold=0.80,
        )
        assert t.name == "tabular-playground"
        assert t.is_lower_better is True
        assert t.gold_threshold == 0.95

    def test_frozen(self):
        t = TaskContext(name="test")
        with pytest.raises(AttributeError):
            t.name = "modified"


class TestSolution:
    def test_minimal(self):
        s = Solution(id="abc123")
        assert s.id == "abc123"
        assert s.plan == ""
        assert s.code == ""
        assert s.score is None
        assert s.is_buggy is False
        assert s.exit_code == 0
        assert s.task is None
        assert s.operators_used == []
        assert s.analysis == ""
        assert s.term_out == ""

    def test_with_score(self):
        task = TaskContext(name="test")
        s = Solution(
            id="s1",
            plan="Use XGBoost",
            code="import xgboost",
            score=0.95,
            task=task,
            operators_used=["draft", "improve"],
        )
        assert s.score == 0.95
        assert s.task.name == "test"
        assert len(s.operators_used) == 2

    def test_frozen(self):
        s = Solution(id="s1")
        with pytest.raises(AttributeError):
            s.id = "s2"

    def test_default_list_independence(self):
        """Each instance gets its own default list."""
        s1 = Solution(id="s1")
        s2 = Solution(id="s2")
        assert s1.operators_used is not s2.operators_used


class TestScorePrediction:
    def test_construction(self):
        p = ScorePrediction(
            solution_id="s1",
            predicted_score=0.85,
            confidence=0.7,
            reasoning="looks good",
        )
        assert p.solution_id == "s1"
        assert p.predicted_score == 0.85
        assert p.confidence == 0.7

    def test_defaults(self):
        p = ScorePrediction(solution_id="s1", predicted_score=0.5)
        assert p.confidence == 0.0
        assert p.reasoning == ""


class TestTournamentPrediction:
    def test_construction(self):
        p = TournamentPrediction(
            solution_a_id="s1",
            solution_b_id="s2",
            winner_id="s1",
            confidence=0.9,
            reasoning="A is better",
        )
        assert p.winner_id == "s1"

    def test_defaults(self):
        p = TournamentPrediction(
            solution_a_id="s1",
            solution_b_id="s2",
            winner_id="s2",
        )
        assert p.confidence == 0.0
        assert p.reasoning == ""
