"""Helpers for using qd archives with research_agent experiments.

These are standalone functions, not a framework.  Import what you need
into your loop script.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from qd.features import DEFAULT_FEATURES, feature_names
from qd.map_elites import Archive, EliteEntry, Feature, GridArchive
from qd.metrics import best_fitness, coverage, qd_score

if TYPE_CHECKING:
    from research_agent.models import RunRecord
    from research_agent.tasks.base import Task


def check_qd_ready(task: Task) -> None:
    """Validate that a task supports QD.  Call at loop startup.

    Raises ``ValueError`` with an actionable message if *task* does not
    define QD features.
    """
    features = task.qd_features()
    if not features:
        raise ValueError(
            f"Task '{task.task_path}' does not define QD features. "
            f"Implement qd_features() and extract_features() on the Task class "
            f"to use QD-guided loops."
        )


def rebuild_archive(
    runs: list[RunRecord],
    features: list[Feature],
    *,
    archive_cls: type[Archive] = GridArchive,
    **archive_kwargs: Any,
) -> Archive:
    """Reconstruct an archive from persisted runs.

    Reads ``metadata["qd_features"]`` from each run.  Skips runs without
    features or with invalid scores.

    Returns:
        A new archive populated with all valid past elites.
    """
    archive = archive_cls(features, **archive_kwargs)
    for run in runs:
        qd_feats = run.metadata.get("qd_features")
        if qd_feats and run.valid and run.score is not None:
            archive.add(run.run_id, run.score, qd_feats)
    return archive


def format_archive_context(
    archive: Archive,
    runs: list[RunRecord] | None = None,
    feature_name_fn: Callable[[dict[str, float]], dict[str, str]] | None = None,
) -> str:
    """Format archive state as a text block for the ideator prompt.

    Includes coverage/QD-score stats, occupied cells with elite scores,
    and empty cell regions as exploration targets.
    """
    if feature_name_fn is None:
        feature_name_fn = feature_names

    total = archive.cell_count()
    occupied = archive.size
    lines = [
        "=== Quality-Diversity Archive ===",
        f"Coverage: {occupied}/{total} cells ({coverage(archive):.0%})",
        f"QD Score: {qd_score(archive):.4f}",
        f"Best Fitness: {best_fitness(archive):.4f}",
    ]

    # Occupied cells with elite info
    if occupied > 0:
        lines.append("")
        lines.append("Elite solutions (one per explored region):")

        # Build ID → run lookup for strategy extraction
        run_map: dict[str, RunRecord] = {}
        if runs:
            run_map = {r.run_id: r for r in runs}

        for elite in sorted(archive.elites(), key=lambda e: e.fitness, reverse=True):
            run = run_map.get(elite.id)
            feat_str = ""
            if elite.features:
                names = feature_name_fn(elite.features)
                feat_str = " | ".join(f"{k}: {v}" for k, v in names.items())

            strategy = ""
            if run and run.idea:
                # Extract first meaningful line from idea as strategy summary
                for line in run.idea.splitlines():
                    stripped = line.strip().lstrip("#").strip()
                    if stripped and not stripped.startswith("---"):
                        strategy = stripped[:80]
                        break

            parts = [f"- {elite.id}: score={elite.fitness:.4f}"]
            if feat_str:
                parts.append(f"[{feat_str}]")
            if strategy:
                parts.append(f'"{strategy}"')
            lines.append(" ".join(parts))

    # Empty regions as exploration targets (for GridArchive)
    if isinstance(archive, GridArchive) and occupied < total:
        lines.append("")
        lines.append("Unexplored regions to target:")
        # Collect occupied cell indices
        occupied_indices = {cell for cell, _ in archive.occupied_cells()}
        # Generate all possible cell indices
        dims = [f.num_bins for f in archive.features]
        empty_cells = _enumerate_empty_cells(dims, occupied_indices)
        # Show up to 10 empty cells with human-readable names
        for cell_idx in empty_cells[:10]:
            feat_dict = {}
            for i, f in enumerate(archive.features):
                feat_dict[f.name] = float(cell_idx[i])
            names = feature_name_fn(feat_dict)
            label = " x ".join(names.values())
            lines.append(f"  - {label}")
        if len(empty_cells) > 10:
            lines.append(f"  ... and {len(empty_cells) - 10} more")

    return "\n".join(lines)


def _enumerate_empty_cells(
    dims: list[int],
    occupied: set[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Enumerate all empty cell indices for a grid archive."""
    import itertools

    all_cells = list(itertools.product(*(range(d) for d in dims)))
    return [c for c in all_cells if c not in occupied]


def select_parents(
    archive: Archive,
    runs: list[RunRecord],
    selector: Any,
    *,
    k: int = 5,
    policy: str = "tournament",
    **kwargs: Any,
) -> list[RunRecord]:
    """Select parent runs from the archive using the given selection policy.

    Maps elite IDs back to RunRecords.  Returns a subset of *runs*
    suitable for passing to ``task.ideator_prompt()``.

    Args:
        archive: The QD archive to select from.
        runs: All completed runs (for ID → RunRecord mapping).
        selector: A ``qd.Selector`` instance (already updated or will be
                  updated here).
        k: Number of parents to select.
        policy: Selection policy name (``"random"``, ``"roulette"``,
                ``"tournament"``, ``"best"``).
        **kwargs: Extra arguments forwarded to the selector method.
    """
    if archive.size == 0:
        return runs[:k]

    selector.update(archive=archive)
    selected_ids = selector.sample(policy, k=k, **kwargs)

    run_map = {r.run_id: r for r in runs}
    parents = []
    seen = set()
    for sid in selected_ids:
        if sid in run_map and sid not in seen:
            parents.append(run_map[sid])
            seen.add(sid)
    return parents
