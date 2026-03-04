"""Load JOURNAL.jsonl files into Solution objects.

Standalone loader — does not depend on the aira-dojo package.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluator.models import Solution, TaskContext


def load_task_description(data_dir: str | Path, competition_id: str) -> str:
    """Read the task description markdown from MLE-bench data.

    Expected path: {data_dir}/{competition_id}/prepared/public/description.md

    Returns:
        The description text, or empty string if file not found.
    """
    desc_path = Path(data_dir) / competition_id / "prepared" / "public" / "description.md"
    if desc_path.exists():
        return desc_path.read_text(encoding="utf-8")
    return ""


def _extract_metric_info_field(
    data: dict, field: str, default=None
):
    """Extract a field from metric_info, handling both flattened and nested formats.

    Flattened: data["metric_info/score"]
    Nested:    data["metric_info"]["score"]
    """
    # Try flattened format first (used in JOURNAL.jsonl)
    flat_key = f"metric_info/{field}"
    if flat_key in data:
        return data[flat_key]
    # Try nested format
    metric_info = data.get("metric_info")
    if isinstance(metric_info, dict) and field in metric_info:
        return metric_info[field]
    return default


def _infer_task_context(
    data: dict, data_dir: str | Path | None = None
) -> TaskContext:
    """Infer TaskContext from a journal entry's metric_info fields."""
    competition_id = _extract_metric_info_field(data, "competition_id", "")
    is_lower_raw = _extract_metric_info_field(data, "is_lower_better", 0.0)
    is_lower_better = bool(float(is_lower_raw)) if is_lower_raw is not None else False

    gold_raw = _extract_metric_info_field(data, "gold_threshold")
    gold_threshold = float(gold_raw) if gold_raw is not None else None

    median_raw = _extract_metric_info_field(data, "median_threshold")
    median_threshold = float(median_raw) if median_raw is not None else None

    description = ""
    if data_dir and competition_id:
        description = load_task_description(data_dir, competition_id)

    return TaskContext(
        name=competition_id or "unknown",
        description=description,
        is_lower_better=is_lower_better,
        gold_threshold=gold_threshold,
        median_threshold=median_threshold,
    )


def load_journal(
    path: str | Path,
    task: TaskContext | None = None,
    data_dir: str | Path | None = None,
) -> list[Solution]:
    """Parse a JOURNAL.jsonl file into Solution objects.

    Args:
        path: Path to the JOURNAL.jsonl file.
        task: Optional pre-built TaskContext. If None, inferred from the
              first entry with metric_info.
        data_dir: Optional MLE-bench data directory for loading task descriptions.
            Defaults to None (no description loaded).

    Returns:
        List of Solution objects (step 0 / root sentinel is skipped).
    """
    path = Path(path)
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    # Infer task context from first entry with metric_info if not provided.
    # Prefer entries with competition_id (graded submissions with full metadata)
    # over early entries that only have validation_score/validity_feedback.
    if task is None:
        fallback_data = None
        for entry in entries:
            data = entry.get("data", entry)
            has_metric_info = any(
                k.startswith("metric_info/") for k in data
            ) or isinstance(data.get("metric_info"), dict)
            if not has_metric_info:
                continue
            if fallback_data is None:
                fallback_data = data
            if _extract_metric_info_field(data, "competition_id"):
                task = _infer_task_context(data, data_dir)
                break
        if task is None and fallback_data is not None:
            task = _infer_task_context(fallback_data, data_dir)
        if task is None:
            task = TaskContext(name="unknown")

    solutions = []
    for entry in entries:
        data = entry.get("data", entry)
        step = data.get("step", 0)
        if step == 0:
            continue  # skip root sentinel

        # Extract score from metric_info/score (flattened) or metric field
        score_raw = _extract_metric_info_field(data, "score")
        if score_raw is None:
            score_raw = data.get("metric")
        score = float(score_raw) if score_raw is not None else None

        is_buggy = data.get("is_buggy", False)
        if isinstance(is_buggy, (int, float)):
            is_buggy = bool(is_buggy)

        # Combine terminal output
        term_out_parts = data.get("_term_out", [])
        if isinstance(term_out_parts, list):
            term_out = "".join(str(t) for t in term_out_parts)
        else:
            term_out = str(data.get("term_out", ""))

        solutions.append(
            Solution(
                id=data.get("id", f"step_{step}"),
                plan=data.get("plan", ""),
                code=data.get("code", ""),
                score=score,
                is_buggy=is_buggy,
                exit_code=data.get("exit_code", 0),
                task=task,
                operators_used=data.get("operators_used", []),
                analysis=data.get("analysis", ""),
                term_out=term_out,
            )
        )

    return solutions


def discover_runs(
    logs_dir: str | Path, prefix: str = ""
) -> list[Path]:
    """Find all JOURNAL.jsonl files under a logs directory.

    Args:
        logs_dir: Root directory to search (e.g. "aira-dojo/logs/aira-dojo").
        prefix: Optional prefix filter on run directory names.

    Returns:
        Sorted list of paths to JOURNAL.jsonl files.
    """
    logs_dir = Path(logs_dir)
    journals = []
    for journal_path in sorted(logs_dir.rglob("JOURNAL.jsonl")):
        if prefix:
            # Check if any parent directory name starts with prefix
            if not any(
                part.startswith(prefix) for part in journal_path.parts
            ):
                continue
        journals.append(journal_path)
    return journals
