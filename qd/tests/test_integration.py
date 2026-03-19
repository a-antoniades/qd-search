"""Tests for qd.integration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from qd import (
    DEFAULT_FEATURES,
    Feature,
    GridArchive,
    Selector,
)
from qd.integration import (
    check_qd_ready,
    format_archive_context,
    rebuild_archive,
    select_parents,
)


# ---------------------------------------------------------------------------
# Lightweight stubs (avoid importing research_agent in unit tests)
# ---------------------------------------------------------------------------


@dataclass
class FakeRunRecord:
    """Mimics research_agent.models.RunRecord for testing."""

    run_id: str
    idea: str | None = None
    score: float | None = None
    valid: bool = True
    workspace: Path = Path(".")
    metadata: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)


class FakeTask:
    """Mimics research_agent.tasks.base.Task for testing."""

    task_path = "fake/test-task"

    def qd_features(self):
        return DEFAULT_FEATURES

    def extract_features(self, idea, workspace):
        return {"model_family": 1.0, "data_strategy": 2.0}


class FakeTaskNoQD:
    """Task without QD support."""

    task_path = "fake/no-qd-task"

    def qd_features(self):
        return []


# ---------------------------------------------------------------------------
# check_qd_ready
# ---------------------------------------------------------------------------


class TestCheckQDReady:
    def test_passes_for_qd_task(self):
        check_qd_ready(FakeTask())  # should not raise

    def test_raises_for_non_qd_task(self):
        with pytest.raises(ValueError, match="does not define QD features"):
            check_qd_ready(FakeTaskNoQD())


# ---------------------------------------------------------------------------
# rebuild_archive
# ---------------------------------------------------------------------------


class TestRebuildArchive:
    def test_empty_runs(self):
        archive = rebuild_archive([], DEFAULT_FEATURES)
        assert archive.size == 0
        assert archive.cell_count() == 70  # 10 x 7

    def test_populates_from_metadata(self):
        runs = [
            FakeRunRecord(
                run_id="r0",
                score=0.8,
                valid=True,
                metadata={"qd_features": {"model_family": 1.0, "data_strategy": 2.0}},
            ),
            FakeRunRecord(
                run_id="r1",
                score=0.9,
                valid=True,
                metadata={"qd_features": {"model_family": 4.0, "data_strategy": 0.0}},
            ),
        ]
        archive = rebuild_archive(runs, DEFAULT_FEATURES)
        assert archive.size == 2
        assert set(archive.elite_ids) == {"r0", "r1"}

    def test_skips_invalid_runs(self):
        runs = [
            FakeRunRecord(
                run_id="r0",
                score=0.8,
                valid=True,
                metadata={"qd_features": {"model_family": 1.0, "data_strategy": 2.0}},
            ),
            FakeRunRecord(
                run_id="r1",
                score=None,
                valid=False,
                metadata={"qd_features": {"model_family": 4.0, "data_strategy": 0.0}},
            ),
            FakeRunRecord(
                run_id="r2",
                score=0.5,
                valid=True,
                metadata={},  # no qd_features
            ),
        ]
        archive = rebuild_archive(runs, DEFAULT_FEATURES)
        assert archive.size == 1
        assert archive.elite_ids == ["r0"]

    def test_replacement_on_same_cell(self):
        """Better fitness replaces worse in the same cell."""
        runs = [
            FakeRunRecord(
                run_id="r0",
                score=0.5,
                valid=True,
                metadata={"qd_features": {"model_family": 1.0, "data_strategy": 2.0}},
            ),
            FakeRunRecord(
                run_id="r1",
                score=0.9,
                valid=True,
                metadata={"qd_features": {"model_family": 1.0, "data_strategy": 2.0}},
            ),
        ]
        archive = rebuild_archive(runs, DEFAULT_FEATURES)
        assert archive.size == 1
        assert archive.elite_ids == ["r1"]


# ---------------------------------------------------------------------------
# format_archive_context
# ---------------------------------------------------------------------------


class TestFormatArchiveContext:
    def test_empty_archive(self):
        archive = GridArchive(DEFAULT_FEATURES)
        text = format_archive_context(archive)
        assert "Coverage: 0/70" in text
        assert "QD Score: 0.0000" in text

    def test_with_elites(self):
        archive = GridArchive(DEFAULT_FEATURES)
        archive.add("r0", 0.85, {"model_family": 1.0, "data_strategy": 2.0})
        runs = [
            FakeRunRecord(
                run_id="r0",
                idea="## Strategy\nUse XGBoost with augmentation",
                score=0.85,
                metadata={"qd_features": {"model_family": 1.0, "data_strategy": 2.0}},
            ),
        ]
        text = format_archive_context(archive, runs)
        assert "Coverage: 1/70" in text
        assert "r0: score=0.8500" in text
        assert "Unexplored regions" in text

    def test_features_from_elite_no_runs_needed(self):
        """format_archive_context reads features from elite directly."""
        archive = GridArchive(DEFAULT_FEATURES)
        archive.add("r0", 0.85, {"model_family": 1.0, "data_strategy": 2.0})
        # No runs passed — features should still appear via elite.features
        text = format_archive_context(archive)
        assert "GBDT" in text  # model_family=1.0 maps to GBDT
        assert "Augmentation" in text  # data_strategy=2.0 maps to Augmentation

    def test_unexplored_regions_shown(self):
        archive = GridArchive(DEFAULT_FEATURES)
        archive.add("r0", 0.85, {"model_family": 0.0, "data_strategy": 0.0})
        text = format_archive_context(archive)
        assert "Unexplored regions to target:" in text
        # Should show human-readable names
        assert "GBDT" in text or "CNN" in text or "RNN" in text


# ---------------------------------------------------------------------------
# select_parents
# ---------------------------------------------------------------------------


class TestSelectParents:
    def test_empty_archive_returns_first_k_runs(self):
        archive = GridArchive(DEFAULT_FEATURES)
        runs = [FakeRunRecord(run_id=f"r{i}") for i in range(10)]
        selector = Selector(seed=0)
        parents = select_parents(archive, runs, selector, k=3)
        assert len(parents) == 3

    def test_selects_from_archive(self):
        archive = GridArchive(DEFAULT_FEATURES)
        archive.add("r0", 0.9, {"model_family": 1.0, "data_strategy": 2.0})
        archive.add("r2", 0.8, {"model_family": 4.0, "data_strategy": 0.0})

        runs = [
            FakeRunRecord(run_id="r0", score=0.9),
            FakeRunRecord(run_id="r1", score=0.5),
            FakeRunRecord(run_id="r2", score=0.8),
        ]

        selector = Selector(seed=42)
        parents = select_parents(archive, runs, selector, k=2, policy="best")
        # Should return the 2 archive elites, not r1
        parent_ids = {p.run_id for p in parents}
        assert parent_ids == {"r0", "r2"}

    def test_deduplicates(self):
        """Selection with replacement shouldn't return duplicate RunRecords."""
        archive = GridArchive(DEFAULT_FEATURES)
        archive.add("r0", 0.9, {"model_family": 1.0, "data_strategy": 2.0})

        runs = [FakeRunRecord(run_id="r0", score=0.9)]

        selector = Selector(seed=0)
        # Random with k=5 on 1-element archive will pick "r0" repeatedly
        parents = select_parents(archive, runs, selector, k=5, policy="random")
        assert len(parents) == 1  # deduplicated to 1
