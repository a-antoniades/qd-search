"""Tests for evaluator.swiss — Swiss-system tournament logic."""

from __future__ import annotations

import math

import pytest

from evaluator.metrics import tournament_accuracy, tournament_rank_correlation
from evaluator.models import Solution, TaskContext, TournamentPrediction
from evaluator.predictors import RandomTournamentPredictor
from evaluator.swiss import SwissStandings, run_swiss_tournament, swiss_pair_round


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def task() -> TaskContext:
    return TaskContext(
        name="test-task",
        metric_name="accuracy",
        is_lower_better=False,
        gold_threshold=0.95,
        median_threshold=0.70,
    )


def _make_solutions(n: int, task: TaskContext) -> list[Solution]:
    """Create n solutions with scores from 0.5 to 0.9."""
    return [
        Solution(
            id=f"s{i}",
            plan=f"Approach {i}",
            score=0.5 + 0.4 * i / max(n - 1, 1),
            is_buggy=False,
            task=task,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# SwissStandings
# ---------------------------------------------------------------------------


class TestSwissStandings:
    def test_initial_state(self):
        s = SwissStandings()
        assert s.wins == {}
        assert s.played == set()
        assert s.bye_ids == set()

    def test_record_result(self):
        s = SwissStandings()
        s.record_result("a", "b")
        assert s.wins["a"] == 1
        assert s.wins["b"] == 0
        assert s.have_played("a", "b")
        assert s.have_played("b", "a")  # symmetric

    def test_record_multiple_results(self):
        s = SwissStandings()
        s.record_result("a", "b")
        s.record_result("a", "c")
        assert s.wins["a"] == 2
        assert s.wins["b"] == 0
        assert s.wins["c"] == 0

    def test_have_played_false(self):
        s = SwissStandings()
        s.record_result("a", "b")
        assert not s.have_played("a", "c")

    def test_record_bye(self):
        s = SwissStandings()
        s.wins["x"] = 0
        s.record_bye("x")
        assert s.wins["x"] == 1
        assert "x" in s.bye_ids

    def test_record_bye_initializes_wins(self):
        s = SwissStandings()
        s.record_bye("new")
        assert s.wins["new"] == 1


# ---------------------------------------------------------------------------
# swiss_pair_round
# ---------------------------------------------------------------------------


class TestSwissPairRound:
    def test_even_count(self):
        """4 solutions → 2 pairs, no bye."""
        import random

        standings = SwissStandings()
        for sid in ["a", "b", "c", "d"]:
            standings.wins[sid] = 0

        pairs, bye = swiss_pair_round(["a", "b", "c", "d"], standings, random.Random(42))
        assert bye is None
        assert len(pairs) == 2
        # All 4 IDs should appear exactly once across pairs
        flat = [x for p in pairs for x in p]
        assert sorted(flat) == ["a", "b", "c", "d"]

    def test_odd_count_gives_bye(self):
        """5 solutions → 2 pairs + 1 bye."""
        import random

        ids = ["a", "b", "c", "d", "e"]
        standings = SwissStandings()
        for sid in ids:
            standings.wins[sid] = 0

        pairs, bye = swiss_pair_round(ids, standings, random.Random(42))
        assert bye is not None
        assert len(pairs) == 2
        paired_ids = {x for p in pairs for x in p}
        assert bye not in paired_ids
        assert len(paired_ids) + 1 == 5

    def test_no_repeated_pairings(self):
        """Over multiple rounds, pairs should avoid rematches when possible."""
        import random

        ids = ["a", "b", "c", "d"]
        standings = SwissStandings()
        for sid in ids:
            standings.wins[sid] = 0
        rng = random.Random(42)

        all_pairs_seen: set[frozenset[str]] = set()
        for _ in range(3):
            pairs, _ = swiss_pair_round(ids, standings, rng)
            for a, b in pairs:
                pair_key = frozenset([a, b])
                # For 4 elements with C(4,2)=6 possible pairs and 2 per round,
                # 3 rounds = 6 pair slots, all 6 unique pairs should be usable
                all_pairs_seen.add(pair_key)
                standings.record_result(a, b)

        # With 4 players, 3 rounds of 2 pairs = 6 matches,
        # and C(4,2) = 6 possible pairs. Should use all unique pairs.
        assert len(all_pairs_seen) == 6

    def test_forced_re_pair(self):
        """When all opponents have been played, force a re-pair."""
        import random

        ids = ["a", "b"]
        standings = SwissStandings()
        standings.wins["a"] = 1
        standings.wins["b"] = 0
        standings.played.add(frozenset(["a", "b"]))

        pairs, bye = swiss_pair_round(ids, standings, random.Random(42))
        assert bye is None
        assert len(pairs) == 1
        assert set(pairs[0]) == {"a", "b"}  # forced re-pair

    def test_single_solution(self):
        """1 solution → no pairs, no bye."""
        import random

        standings = SwissStandings()
        standings.wins["a"] = 0
        pairs, bye = swiss_pair_round(["a"], standings, random.Random(42))
        assert pairs == []
        assert bye is None

    def test_bye_rotates(self):
        """With odd count, bye should go to different players across rounds."""
        import random

        ids = ["a", "b", "c"]
        standings = SwissStandings()
        for sid in ids:
            standings.wins[sid] = 0
        rng = random.Random(42)

        byes = []
        for _ in range(3):
            pairs, bye = swiss_pair_round(ids, standings, rng)
            assert bye is not None
            byes.append(bye)
            standings.record_bye(bye)
            for a, b in pairs:
                standings.record_result(a, b)

        # All 3 players should get a bye (one each)
        assert set(byes) == {"a", "b", "c"}

    def test_ranking_influences_pairing(self):
        """Higher-ranked solutions should be paired together."""
        import random

        ids = ["top1", "top2", "mid1", "mid2"]
        standings = SwissStandings()
        standings.wins["top1"] = 3
        standings.wins["top2"] = 3
        standings.wins["mid1"] = 0
        standings.wins["mid2"] = 0

        pairs, bye = swiss_pair_round(ids, standings, random.Random(42))
        assert bye is None
        # Top-ranked should be paired together, mid-ranked together
        pair_sets = [set(p) for p in pairs]
        assert {"top1", "top2"} in pair_sets
        assert {"mid1", "mid2"} in pair_sets


# ---------------------------------------------------------------------------
# run_swiss_tournament
# ---------------------------------------------------------------------------


class TestRunSwissTournament:
    def test_correct_prediction_count(self, task):
        """Total predictions = num_rounds * floor(n/2)."""
        solutions = _make_solutions(10, task)
        predictor = RandomTournamentPredictor(seed=42)

        preds = run_swiss_tournament(predictor, solutions, task, seed=42)

        n = 10
        num_rounds = math.ceil(math.log2(n))  # 4
        expected = num_rounds * (n // 2)
        assert len(preds) == expected

    def test_correct_count_odd(self, task):
        """Odd n: each round has floor(n/2) matches + 1 bye."""
        solutions = _make_solutions(7, task)
        predictor = RandomTournamentPredictor(seed=42)

        preds = run_swiss_tournament(predictor, solutions, task, seed=42)

        n = 7
        num_rounds = math.ceil(math.log2(n))  # 3
        expected = num_rounds * (n // 2)  # 3 * 3 = 9
        assert len(preds) == expected

    def test_deterministic_with_seed(self, task):
        """Same seed produces identical results."""
        solutions = _make_solutions(8, task)
        predictor1 = RandomTournamentPredictor(seed=42)
        predictor2 = RandomTournamentPredictor(seed=42)

        preds1 = run_swiss_tournament(predictor1, solutions, task, seed=99)
        preds2 = run_swiss_tournament(predictor2, solutions, task, seed=99)

        assert len(preds1) == len(preds2)
        for p1, p2 in zip(preds1, preds2):
            assert p1.solution_a_id == p2.solution_a_id
            assert p1.solution_b_id == p2.solution_b_id
            assert p1.winner_id == p2.winner_id

    def test_compatible_with_tournament_metrics(self, task):
        """Output works with tournament_accuracy() and tournament_rank_correlation()."""
        solutions = _make_solutions(10, task)
        predictor = RandomTournamentPredictor(seed=42)
        actuals = {s.id: s.score for s in solutions}

        preds = run_swiss_tournament(predictor, solutions, task, seed=42)

        acc = tournament_accuracy(preds, actuals, is_lower_better=task.is_lower_better)
        rho = tournament_rank_correlation(preds, actuals)

        assert 0.0 <= acc <= 1.0
        assert -1.0 <= rho <= 1.0

    def test_n_equals_2(self, task):
        """Minimum viable tournament: 2 solutions, 1 round, 1 match."""
        solutions = _make_solutions(2, task)
        predictor = RandomTournamentPredictor(seed=42)

        preds = run_swiss_tournament(predictor, solutions, task, seed=42)
        assert len(preds) == 1

    def test_n_equals_3_odd(self, task):
        """3 solutions: ceil(log2(3))=2 rounds, 1 match per round."""
        solutions = _make_solutions(3, task)
        predictor = RandomTournamentPredictor(seed=42)

        preds = run_swiss_tournament(predictor, solutions, task, seed=42)

        num_rounds = math.ceil(math.log2(3))  # 2
        expected = num_rounds * (3 // 2)  # 2 * 1 = 2
        assert len(preds) == expected

    def test_n_equals_1(self, task):
        """Single solution: no matches possible."""
        solutions = _make_solutions(1, task)
        predictor = RandomTournamentPredictor(seed=42)

        preds = run_swiss_tournament(predictor, solutions, task, seed=42)
        assert preds == []

    def test_filters_buggy_solutions(self, task):
        """Buggy and unscored solutions are excluded."""
        solutions = [
            Solution(id="good1", plan="A", score=0.9, is_buggy=False, task=task),
            Solution(id="good2", plan="B", score=0.8, is_buggy=False, task=task),
            Solution(id="buggy", plan="C", score=0.7, is_buggy=True, task=task),
            Solution(id="unscored", plan="D", score=None, is_buggy=False, task=task),
        ]
        predictor = RandomTournamentPredictor(seed=42)

        preds = run_swiss_tournament(predictor, solutions, task, seed=42)
        # Only 2 valid solutions → 1 round × 1 match
        assert len(preds) == 1
        ids_in_preds = {preds[0].solution_a_id, preds[0].solution_b_id}
        assert ids_in_preds == {"good1", "good2"}

    def test_custom_num_rounds(self, task):
        """Explicit num_rounds overrides the default."""
        solutions = _make_solutions(10, task)
        predictor = RandomTournamentPredictor(seed=42)

        preds = run_swiss_tournament(
            predictor, solutions, task, num_rounds=2, seed=42
        )
        expected = 2 * (10 // 2)  # 2 * 5 = 10
        assert len(preds) == expected

    def test_predictions_are_tournament_predictions(self, task):
        """All returned objects are TournamentPrediction instances."""
        solutions = _make_solutions(6, task)
        predictor = RandomTournamentPredictor(seed=42)

        preds = run_swiss_tournament(predictor, solutions, task, seed=42)
        for p in preds:
            assert isinstance(p, TournamentPrediction)
            assert p.solution_a_id
            assert p.solution_b_id
            assert p.winner_id in (p.solution_a_id, p.solution_b_id)

    def test_far_fewer_than_exhaustive(self, task):
        """Swiss should use far fewer comparisons than C(n,2)."""
        n = 30
        solutions = _make_solutions(n, task)
        predictor = RandomTournamentPredictor(seed=42)

        preds = run_swiss_tournament(predictor, solutions, task, seed=42)

        exhaustive = n * (n - 1) // 2  # 435
        swiss_count = len(preds)  # 5 * 15 = 75
        assert swiss_count < exhaustive * 0.25  # at least 75% reduction
