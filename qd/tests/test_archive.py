"""Tests for the qd package: archives, selection, CVT, and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from qd import (
    CVTArchive,
    EliteEntry,
    Feature,
    GridArchive,
    Selector,
    best_fitness,
    closest_centroid_idx,
    coverage,
    cvt,
    qd_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_features() -> list[Feature]:
    return [
        Feature("x", 0.0, 1.0, num_bins=5),
        Feature("y", 0.0, 1.0, num_bins=4),
    ]


@pytest.fixture
def grid(grid_features) -> GridArchive:
    return GridArchive(grid_features)


# ---------------------------------------------------------------------------
# GridArchive
# ---------------------------------------------------------------------------

class TestGridArchive:
    def test_cell_count(self, grid: GridArchive):
        assert grid.cell_count() == 20  # 5 * 4

    def test_empty(self, grid: GridArchive):
        assert grid.size == 0
        assert grid.elites() == []
        assert grid.elite_ids == []

    def test_add_single(self, grid: GridArchive):
        inserted = grid.add("a", 0.5, {"x": 0.1, "y": 0.2})
        assert inserted is True
        assert grid.size == 1
        assert grid.elite_ids == ["a"]

    def test_add_better_replaces(self, grid: GridArchive):
        grid.add("a", 0.5, {"x": 0.1, "y": 0.2})
        inserted = grid.add("b", 0.9, {"x": 0.1, "y": 0.2})
        assert inserted is True
        assert grid.size == 1
        assert grid.elite_ids == ["b"]

    def test_add_worse_rejected(self, grid: GridArchive):
        grid.add("a", 0.9, {"x": 0.1, "y": 0.2})
        inserted = grid.add("b", 0.3, {"x": 0.1, "y": 0.2})
        assert inserted is False
        assert grid.elite_ids == ["a"]

    def test_different_cells(self, grid: GridArchive):
        grid.add("a", 0.5, {"x": 0.1, "y": 0.1})
        grid.add("b", 0.6, {"x": 0.9, "y": 0.9})
        assert grid.size == 2
        assert set(grid.elite_ids) == {"a", "b"}

    def test_missing_feature_rejected(self, grid: GridArchive):
        inserted = grid.add("a", 0.5, {"x": 0.1})
        assert inserted is False
        assert grid.size == 0

    def test_clamp_to_bounds(self, grid: GridArchive):
        # Values outside [0, 1] should be clamped, not rejected.
        inserted = grid.add("a", 0.5, {"x": -5.0, "y": 10.0})
        assert inserted is True
        assert grid.size == 1

    def test_num_bins_required(self):
        feats = [Feature("x", 0.0, 1.0)]  # no num_bins
        with pytest.raises(ValueError, match="num_bins"):
            GridArchive(feats)

    def test_cell_placement_deterministic(self, grid_features):
        """Same features → same cell, regardless of insertion order."""
        a = GridArchive(grid_features)
        b = GridArchive(grid_features)
        feats = {"x": 0.45, "y": 0.75}
        a.add("p", 1.0, feats)
        b.add("q", 2.0, feats)
        # Both archives should have exactly one occupied cell at the same index.
        assert a.size == b.size == 1
        # Internally they should map to the same cell tuple.
        assert list(a._map.keys()) == list(b._map.keys())

    def test_features_stored_in_elite(self, grid: GridArchive):
        feats = {"x": 0.3, "y": 0.7}
        grid.add("a", 1.0, feats)
        elite = grid.elites()[0]
        assert elite.features == feats

    def test_minimize(self, grid_features):
        archive = GridArchive(grid_features, maximize=False)
        archive.add("a", 0.9, {"x": 0.1, "y": 0.2})
        archive.add("b", 0.3, {"x": 0.1, "y": 0.2})  # lower is better
        assert archive.elite_ids == ["b"]
        # Higher fitness rejected
        inserted = archive.add("c", 0.5, {"x": 0.1, "y": 0.2})
        assert inserted is False
        assert archive.elite_ids == ["b"]

    def test_cell_visits_and_improvements(self, grid: GridArchive):
        grid.add("a", 0.5, {"x": 0.1, "y": 0.2})  # insert (visit+improve)
        grid.add("b", 0.3, {"x": 0.1, "y": 0.2})  # rejected (visit only)
        grid.add("c", 0.9, {"x": 0.1, "y": 0.2})  # replace (visit+improve)
        grid.add("d", 0.8, {"x": 0.9, "y": 0.9})  # new cell (visit+improve)

        visits = grid.cell_visits()
        improvements = grid.cell_improvements()
        assert grid.total_visits == 4
        # The cell containing x=0.1/y=0.2 had 3 visits, 2 improvements
        cell_01 = grid._cell_index({"x": 0.1, "y": 0.2})
        assert visits[cell_01] == 3
        assert improvements[cell_01] == 2

    def test_add_batch(self, grid: GridArchive):
        results = grid.add_batch(
            ids=["a", "b", "c"],
            fitnesses=[0.5, 0.9, 0.3],
            features_list=[
                {"x": 0.1, "y": 0.2},
                {"x": 0.1, "y": 0.2},  # same cell, better
                {"x": 0.9, "y": 0.9},  # different cell
            ],
        )
        assert results == [True, True, True]
        assert grid.size == 2
        assert set(grid.elite_ids) == {"b", "c"}


# ---------------------------------------------------------------------------
# CVTArchive
# ---------------------------------------------------------------------------

class TestCVTArchive:
    def test_basic(self):
        np.random.seed(42)
        feats = [Feature("x", 0.0, 1.0), Feature("y", 0.0, 1.0)]
        archive = CVTArchive(feats, num_centroids=10, num_init_samples=500)
        assert archive.cell_count() == 10
        assert archive.size == 0

        archive.add("a", 0.5, {"x": 0.1, "y": 0.1})
        assert archive.size == 1

    def test_replacement(self):
        np.random.seed(42)
        feats = [Feature("x", 0.0, 1.0)]
        archive = CVTArchive(feats, num_centroids=5, num_init_samples=200)
        archive.add("a", 0.3, {"x": 0.5})
        archive.add("b", 0.9, {"x": 0.5})  # same region, higher fitness
        assert archive.size == 1
        assert archive.elite_ids == ["b"]

    def test_missing_feature(self):
        np.random.seed(42)
        feats = [Feature("x", 0.0, 1.0), Feature("y", 0.0, 1.0)]
        archive = CVTArchive(feats, num_centroids=5, num_init_samples=200)
        assert archive.add("a", 1.0, {"x": 0.5}) is False

    def test_multiple_cells(self):
        np.random.seed(42)
        feats = [Feature("x", 0.0, 1.0)]
        archive = CVTArchive(feats, num_centroids=10, num_init_samples=500)
        # Add solutions spread across the feature space.
        for i in range(10):
            archive.add(f"s{i}", float(i), {"x": i / 10.0})
        # Expect most centroids to be occupied.
        assert archive.size >= 5

    def test_features_stored(self):
        np.random.seed(42)
        feats = [Feature("x", 0.0, 1.0)]
        archive = CVTArchive(feats, num_centroids=5, num_init_samples=200)
        archive.add("a", 0.5, {"x": 0.3})
        elite = archive.elites()[0]
        assert elite.features == {"x": 0.3}

    def test_minimize(self):
        np.random.seed(42)
        feats = [Feature("x", 0.0, 1.0)]
        archive = CVTArchive(feats, num_centroids=5, num_init_samples=200, maximize=False)
        archive.add("a", 0.9, {"x": 0.5})
        archive.add("b", 0.3, {"x": 0.5})  # lower is better
        assert archive.size == 1
        assert archive.elite_ids == ["b"]

    def test_cell_visits(self):
        np.random.seed(42)
        feats = [Feature("x", 0.0, 1.0)]
        archive = CVTArchive(feats, num_centroids=5, num_init_samples=200)
        archive.add("a", 0.5, {"x": 0.5})
        archive.add("b", 0.3, {"x": 0.5})  # rejected
        archive.add("c", 0.9, {"x": 0.5})  # replaces
        assert archive.total_visits == 3
        visits = archive.cell_visits()
        improvements = archive.cell_improvements()
        # Single cell visited 3 times with 2 improvements
        assert sum(visits.values()) == 3
        assert sum(improvements.values()) == 2


# ---------------------------------------------------------------------------
# CVT utilities
# ---------------------------------------------------------------------------

class TestCVT:
    def test_closest_centroid(self):
        centroids = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert closest_centroid_idx(np.array([0.1, 0.1]), centroids) == 0
        assert closest_centroid_idx(np.array([1.1, 0.9]), centroids) == 1
        assert closest_centroid_idx(np.array([1.9, 2.1]), centroids) == 2

    def test_cvt_shape(self):
        np.random.seed(0)
        centroids = cvt(8, 500, [(0, 1), (0, 1)])
        assert centroids.shape == (8, 2)
        # All centroids should be within bounds.
        assert np.all(centroids >= 0) and np.all(centroids <= 1)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class TestSelector:
    @pytest.fixture
    def populated_grid(self, grid: GridArchive) -> GridArchive:
        for i in range(10):
            grid.add(f"s{i}", float(i), {"x": i / 10.0, "y": i / 10.0})
        return grid

    def test_random(self, populated_grid):
        sel = Selector(seed=0)
        sel.update(archive=populated_grid)
        ids = sel.random(k=3)
        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)

    def test_roulette_rank(self, populated_grid):
        sel = Selector(seed=0)
        sel.update(archive=populated_grid)
        ids = sel.roulette(k=5, by_rank=True)
        assert len(ids) == 5

    def test_roulette_fitness(self, populated_grid):
        sel = Selector(seed=0)
        sel.update(archive=populated_grid)
        ids = sel.roulette(k=5, by_rank=False)
        assert len(ids) == 5

    def test_tournament(self, populated_grid):
        sel = Selector(seed=0)
        sel.update(archive=populated_grid)
        ids = sel.tournament(k=2, tournament_size=5)
        assert len(ids) == 2

    def test_best(self, populated_grid):
        sel = Selector(seed=0)
        sel.update(archive=populated_grid)
        ids = sel.best(k=3)
        assert ids[0] == "s9"  # highest fitness
        # s8 shares a cell with s9 (grid discretization), so s7 is second best.
        assert ids[1] == "s7"

    def test_sample_dispatch(self, populated_grid):
        sel = Selector(seed=0)
        sel.update(archive=populated_grid)
        ids = sel.sample("best", k=1)
        assert ids == ["s9"]

    def test_sample_unknown_policy(self, populated_grid):
        sel = Selector(seed=0)
        sel.update(archive=populated_grid)
        with pytest.raises(ValueError, match="Unknown policy"):
            sel.sample("nonexistent", k=1)

    def test_empty_pool(self):
        sel = Selector(seed=0)
        feats = [Feature("x", 0.0, 1.0, num_bins=5)]
        sel.update(archive=GridArchive(feats))
        assert sel.random(k=3) == []
        assert sel.best(k=1) == []

    def test_update_with_explicit_pool(self):
        sel = Selector(seed=0)
        pool = [EliteEntry("a", 1.0), EliteEntry("b", 2.0), EliteEntry("c", 0.5)]
        sel.update(pool=pool)
        assert sel.best(k=1) == ["b"]

    def test_update_requires_arg(self):
        sel = Selector()
        with pytest.raises(ValueError):
            sel.update()

    def test_best_biased(self, populated_grid):
        """Best selection should heavily favour the top IDs over many draws."""
        sel = Selector(seed=42)
        sel.update(archive=populated_grid)
        draws = [sel.best(k=1)[0] for _ in range(50)]
        assert all(d == "s9" for d in draws)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_coverage(self, grid: GridArchive):
        assert coverage(grid) == 0.0
        grid.add("a", 1.0, {"x": 0.5, "y": 0.5})
        assert coverage(grid) == pytest.approx(1 / 20)

    def test_qd_score(self, grid: GridArchive):
        assert qd_score(grid) == 0.0
        grid.add("a", 1.0, {"x": 0.1, "y": 0.1})
        grid.add("b", 2.0, {"x": 0.9, "y": 0.9})
        assert qd_score(grid) == pytest.approx(3.0)

    def test_best_fitness_empty(self, grid: GridArchive):
        assert best_fitness(grid) == float("-inf")

    def test_best_fitness(self, grid: GridArchive):
        grid.add("a", 1.0, {"x": 0.1, "y": 0.1})
        grid.add("b", 5.0, {"x": 0.9, "y": 0.9})
        assert best_fitness(grid) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Round-trip integration test
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_add_select_metrics(self):
        """Full pipeline: create archive → add solutions → select → metrics."""
        feats = [
            Feature("complexity", 0.0, 10.0, num_bins=5),
            Feature("speed", 0.0, 100.0, num_bins=4),
        ]
        archive = GridArchive(feats)
        assert archive.cell_count() == 20

        solutions = [
            ("s0", 0.8, {"complexity": 1.0, "speed": 90.0}),
            ("s1", 0.6, {"complexity": 5.0, "speed": 50.0}),
            ("s2", 0.9, {"complexity": 9.0, "speed": 10.0}),
            ("s3", 0.7, {"complexity": 1.0, "speed": 90.0}),  # same cell as s0, worse
            ("s4", 0.95, {"complexity": 1.0, "speed": 90.0}),  # same cell as s0, better
        ]
        for sid, fit, f in solutions:
            archive.add(sid, fit, f)

        # s0 replaced by s4 (same cell, better fitness). s3 rejected.
        assert archive.size == 3
        assert set(archive.elite_ids) == {"s4", "s1", "s2"}

        # Selection.
        sel = Selector(seed=123)
        sel.update(archive=archive)
        top = sel.best(k=1)
        assert top == ["s4"]  # 0.95 is highest

        # Metrics.
        assert coverage(archive) == pytest.approx(3 / 20)
        assert qd_score(archive) == pytest.approx(0.95 + 0.6 + 0.9)
        assert best_fitness(archive) == pytest.approx(0.95)
