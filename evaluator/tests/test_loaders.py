"""Tests for evaluator.loaders — JOURNAL.jsonl parsing and discovery."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from evaluator.loaders import (
    _extract_metric_info_field,
    discover_runs,
    load_journal,
    load_task_description,
)
from evaluator.models import TaskContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_journal_entry(step: int, **overrides) -> dict:
    """Create a minimal journal entry."""
    data = {
        "step": step,
        "id": f"id_{step}",
        "plan": f"Plan for step {step}",
        "code": f"# code for step {step}",
        "metric": None,
        "is_buggy": True,
        "exit_code": 1,
        "analysis": "",
        "operators_used": ["draft"],
        "term_out": "",
        "_term_out": [],
    }
    data.update(overrides)
    return {"timestamp": "2026-01-01T00:00:00", "step": step, "data": data}


def _make_scored_entry(step: int, score: float, competition_id: str = "test-comp") -> dict:
    """Create a journal entry with metric_info in flattened format."""
    return _make_journal_entry(
        step,
        metric=score,
        is_buggy=False,
        exit_code=0,
        **{
            "metric_info/score": score,
            "metric_info/competition_id": competition_id,
            "metric_info/gold_threshold": 0.95,
            "metric_info/median_threshold": 0.70,
            "metric_info/is_lower_better": 0.0,
            "metric_info/gold_medal": 1.0 if score >= 0.95 else 0.0,
        },
    )


@pytest.fixture
def journal_dir(tmp_path) -> Path:
    """Create a temp directory with a JOURNAL.jsonl file."""
    run_dir = tmp_path / "logs" / "user_test_issue_1" / "cfg_abc" / "json"
    run_dir.mkdir(parents=True)

    entries = [
        _make_journal_entry(0),  # root sentinel
        _make_journal_entry(1, metric=None, is_buggy=True),
        _make_scored_entry(2, 0.85),
        _make_scored_entry(3, 0.92),
        _make_scored_entry(4, 0.96),
    ]

    journal_path = run_dir / "JOURNAL.jsonl"
    with open(journal_path, "w") as f:
        for entry in entries:
            # Flatten: move metric_info/* keys into data
            data = entry["data"]
            flat_data = {}
            for k, v in data.items():
                flat_data[k] = v
            entry["data"] = flat_data
            f.write(json.dumps(entry) + "\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: _extract_metric_info_field
# ---------------------------------------------------------------------------


class TestExtractMetricInfoField:
    def test_flattened_format(self):
        data = {"metric_info/score": 0.85, "metric_info/gold_threshold": 0.95}
        assert _extract_metric_info_field(data, "score") == 0.85
        assert _extract_metric_info_field(data, "gold_threshold") == 0.95

    def test_nested_format(self):
        data = {"metric_info": {"score": 0.85, "gold_threshold": 0.95}}
        assert _extract_metric_info_field(data, "score") == 0.85

    def test_missing_returns_default(self):
        data = {}
        assert _extract_metric_info_field(data, "score") is None
        assert _extract_metric_info_field(data, "score", -1) == -1


# ---------------------------------------------------------------------------
# Tests: load_journal
# ---------------------------------------------------------------------------


class TestLoadJournal:
    def test_basic_loading(self, journal_dir):
        journal_path = list(journal_dir.rglob("JOURNAL.jsonl"))[0]
        solutions = load_journal(journal_path)

        # Step 0 (root) should be skipped
        assert len(solutions) == 4

    def test_step_zero_skipped(self, journal_dir):
        journal_path = list(journal_dir.rglob("JOURNAL.jsonl"))[0]
        solutions = load_journal(journal_path)
        step_ids = [s.id for s in solutions]
        assert "id_0" not in step_ids

    def test_score_extraction(self, journal_dir):
        journal_path = list(journal_dir.rglob("JOURNAL.jsonl"))[0]
        solutions = load_journal(journal_path)

        # Step 1 has no score (buggy)
        assert solutions[0].score is None
        # Steps 2-4 have scores
        assert solutions[1].score == 0.85
        assert solutions[2].score == 0.92
        assert solutions[3].score == 0.96

    def test_task_context_inferred(self, journal_dir):
        journal_path = list(journal_dir.rglob("JOURNAL.jsonl"))[0]
        solutions = load_journal(journal_path)

        task = solutions[0].task
        assert task is not None
        assert task.name == "test-comp"
        assert task.gold_threshold == 0.95
        assert task.median_threshold == 0.70
        assert task.is_lower_better is False

    def test_explicit_task_overrides(self, journal_dir):
        journal_path = list(journal_dir.rglob("JOURNAL.jsonl"))[0]
        custom_task = TaskContext(name="custom", gold_threshold=0.99)
        solutions = load_journal(journal_path, task=custom_task)

        assert solutions[0].task.name == "custom"
        assert solutions[0].task.gold_threshold == 0.99

    def test_buggy_flag(self, journal_dir):
        journal_path = list(journal_dir.rglob("JOURNAL.jsonl"))[0]
        solutions = load_journal(journal_path)

        assert solutions[0].is_buggy is True  # step 1
        assert solutions[1].is_buggy is False  # step 2

    def test_operators_preserved(self, journal_dir):
        journal_path = list(journal_dir.rglob("JOURNAL.jsonl"))[0]
        solutions = load_journal(journal_path)
        assert solutions[0].operators_used == ["draft"]

    def test_empty_file(self, tmp_path):
        empty_journal = tmp_path / "JOURNAL.jsonl"
        empty_journal.write_text("")
        solutions = load_journal(empty_journal)
        assert solutions == []


# ---------------------------------------------------------------------------
# Tests: load_task_description
# ---------------------------------------------------------------------------


class TestLoadTaskDescription:
    def test_existing_description(self, tmp_path):
        desc_path = tmp_path / "my-comp" / "prepared" / "public"
        desc_path.mkdir(parents=True)
        (desc_path / "description.md").write_text("# My Competition\nPredict X.")

        result = load_task_description(tmp_path, "my-comp")
        assert "My Competition" in result
        assert "Predict X." in result

    def test_missing_description(self, tmp_path):
        result = load_task_description(tmp_path, "nonexistent")
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: discover_runs
# ---------------------------------------------------------------------------


class TestDiscoverRuns:
    def test_finds_journals(self, journal_dir):
        journals = discover_runs(journal_dir / "logs")
        assert len(journals) == 1
        assert journals[0].name == "JOURNAL.jsonl"

    def test_prefix_filter(self, tmp_path):
        # Create two run directories
        for name in ["QD_STUDY_evo_run1", "QD_STUDY_greedy_run1", "OTHER_run"]:
            d = tmp_path / name / "json"
            d.mkdir(parents=True)
            (d / "JOURNAL.jsonl").write_text("{}\n")

        all_journals = discover_runs(tmp_path)
        assert len(all_journals) == 3

        evo_journals = discover_runs(tmp_path, prefix="QD_STUDY_evo")
        assert len(evo_journals) == 1

        qd_journals = discover_runs(tmp_path, prefix="QD_STUDY")
        assert len(qd_journals) == 2

    def test_empty_dir(self, tmp_path):
        assert discover_runs(tmp_path) == []
