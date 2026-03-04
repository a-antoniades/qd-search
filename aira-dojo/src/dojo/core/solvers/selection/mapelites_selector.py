# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MAP-Elites diversity-aware node selection.

Uses a GridArchive (model_family x data_strategy = 30 cells) to track
solution diversity alongside the island-based population. Parents are
selected from the archive using configurable QD selection policies
(tournament, roulette, random, best).

The archive runs parallel to the island system: islands manage the
population and generation loop, the archive drives diversity-aware
parent selection.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dojo.core.solvers.selection.base import NodeSelector, SelectedNode, SelectionResult
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.config_dataclasses.selector.mapelites import MAPElitesSelectorConfig

log = logging.getLogger(__name__)


def _ensure_qd_importable():
    """Add qd/ package to sys.path if needed (monorepo layout, not pip-installed)."""
    qd_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "qd"
    if qd_root.is_dir() and str(qd_root.parent) not in sys.path:
        sys.path.insert(0, str(qd_root.parent))


class MAPElitesNodeSelector(NodeSelector):
    """
    MAP-Elites diversity-aware node selector.

    Maintains a GridArchive that maps solutions into a 2D feature space
    (model_family x data_strategy). On each select() call, syncs new
    journal nodes into the archive and selects parents using QD selection
    policies to promote diverse exploration.
    """

    def __init__(
        self,
        cfg: MAPElitesSelectorConfig,
        lower_is_better: bool,
    ):
        self.cfg = cfg
        self.lower_is_better = lower_is_better

        # Lazy-init archive and selector on first use
        self._archive = None
        self._selector = None
        self._fitness_selector = None

        # Track which node IDs we've already processed (incremental sync)
        self._processed_node_ids: set = set()

    @property
    def selector_type(self) -> str:
        return "mapelites"

    def _init_qd(self):
        """Lazy initialization of QD archive and selector."""
        _ensure_qd_importable()
        from qd.map_elites import GridArchive
        from qd.features import DEFAULT_FEATURES
        from qd.selection import Selector

        self._archive = GridArchive(DEFAULT_FEATURES)
        self._selector = Selector(seed=self.cfg.seed)
        log.info(
            f"MAP-Elites archive initialized: {self._archive.cell_count()} cells, "
            f"policy={self.cfg.selection_policy}"
        )

    def _get_fitness_selector(self):
        """Get fitness selector for fallback."""
        if self._fitness_selector is None:
            from dojo.core.solvers.selection.fitness_selector import FitnessNodeSelector
            self._fitness_selector = FitnessNodeSelector(
                lower_is_better=self.lower_is_better,
                verbose=self.cfg.verbose,
            )
        return self._fitness_selector

    def _node_fitness(self, node: Node) -> Optional[float]:
        """Extract fitness from a node, negating if lower_is_better."""
        if node.metric is None or node.metric.value is None:
            return None
        value = node.metric.value
        # Archive always maximizes — negate if lower is better
        return -value if self.lower_is_better else value

    def _sync_archive(self, journal: Journal) -> int:
        """
        Sync new non-buggy nodes from the journal into the archive.

        Returns the number of newly added elites.
        """
        _ensure_qd_importable()
        from qd.features import extract_features

        added = 0
        for node in journal.nodes:
            if node.id in self._processed_node_ids:
                continue
            self._processed_node_ids.add(node.id)

            # Skip buggy nodes and nodes without valid metrics
            if node.is_buggy:
                continue
            fitness = self._node_fitness(node)
            if fitness is None:
                continue

            # Extract features from plan + code (keyword-based, no LLM calls)
            features = extract_features(node.plan or "", node.code or "")

            inserted = self._archive.add(
                id=node.id,
                fitness=fitness,
                features=features,
            )
            if inserted:
                added += 1
                if self.cfg.verbose:
                    from qd.features import feature_names
                    names = feature_names(features)
                    log.info(
                        f"Archive: added node {node.step} (id={node.id[:8]}...) "
                        f"fitness={fitness:.4f} -> [{names['model_family']}, {names['data_strategy']}]"
                    )

        return added

    def _find_island_for_node(self, node_id: str, islands: list) -> int:
        """Look up which island a node belongs to. Returns 0 if not found."""
        for idx, island in enumerate(islands):
            for node in island.nodes:
                if node.id == node_id:
                    return idx
        return 0

    # ── diversity steering helpers ────────────────────────────────────────

    def _display_fitness(self, archive_fitness: float) -> float:
        """Convert archive fitness back to original score for display."""
        return -archive_fitness if self.lower_is_better else archive_fitness

    def _get_archive_summary(self) -> tuple:
        """Get occupied and empty cells with human-readable names.

        Returns:
            (occupied, empty) where occupied is [(model_name, data_name, display_score)]
            and empty is [(model_name, data_name)].
        """
        _ensure_qd_importable()
        from qd.features import DEFAULT_FEATURES, MODEL_FAMILY_NAMES, DATA_STRATEGY_NAMES

        n_model = DEFAULT_FEATURES[0].num_bins
        n_data = DEFAULT_FEATURES[1].num_bins

        occupied = []
        occupied_set = set()
        for (m, d), elite in self._archive.occupied_cells():
            occupied.append((
                MODEL_FAMILY_NAMES[m],
                DATA_STRATEGY_NAMES[d],
                self._display_fitness(elite.fitness),
            ))
            occupied_set.add((m, d))

        empty = []
        for m in range(n_model):
            for d in range(n_data):
                if (m, d) not in occupied_set:
                    empty.append((MODEL_FAMILY_NAMES[m], DATA_STRATEGY_NAMES[d]))

        return occupied, empty

    def _format_archive_context(
        self,
        occupied: list,
        empty: list,
        max_occupied: int = 10,
        max_empty: int = 5,
    ) -> str:
        """Format archive state as a string for LLM prompt injection."""
        if not occupied:
            return ""

        total = self._archive.cell_count()
        n_occupied = len(occupied)
        coverage_pct = n_occupied / total * 100

        occupied_sorted = sorted(occupied, key=lambda x: x[2], reverse=True)[:max_occupied]
        occ_lines = [
            f"  - {mf} + {ds} (score: {score:.4f})"
            for mf, ds, score in occupied_sorted
        ]

        parts = [
            f"Archive: {n_occupied}/{total} cells occupied ({coverage_pct:.0f}% coverage).",
            "Explored approaches:\n" + "\n".join(occ_lines),
        ]

        if empty:
            empty_lines = [f"  - {mf} + {ds}" for mf, ds in empty[:max_empty]]
            remaining = len(empty) - max_empty
            if remaining > 0:
                empty_lines.append(f"  ... and {remaining} more unexplored regions")
            parts.append("Unexplored regions:\n" + "\n".join(empty_lines))
        else:
            parts.append("All regions explored.")

        return "\n".join(parts)

    def _pick_target_cell(self) -> tuple:
        """Pick a random empty cell to target for exploration drafts.

        Prioritizes empty cells. When all cells are occupied, picks a random occupied cell.
        Returns (model_family_name, data_strategy_name).
        """
        _, empty = self._get_archive_summary()
        if empty:
            return random.choice(empty)
        # All cells occupied — pick random occupied
        _ensure_qd_importable()
        from qd.features import MODEL_FAMILY_NAMES, DATA_STRATEGY_NAMES

        cells = [cell for cell, _ in self._archive.occupied_cells()]
        m, d = random.choice(cells)
        return MODEL_FAMILY_NAMES[m], DATA_STRATEGY_NAMES[d]

    def _node_cell_name(self, node_id: str) -> Optional[tuple]:
        """Look up the archive cell for a node ID.

        Returns (model_family_name, data_strategy_name, display_score) or None.
        """
        _ensure_qd_importable()
        from qd.features import MODEL_FAMILY_NAMES, DATA_STRATEGY_NAMES

        for (m, d), elite in self._archive.occupied_cells():
            if elite.id == node_id:
                return (
                    MODEL_FAMILY_NAMES[m],
                    DATA_STRATEGY_NAMES[d],
                    self._display_fitness(elite.fitness),
                )
        return None

    def _compose_reasoning(
        self,
        operator: str,
        selected_nodes: List[SelectedNode],
        target_cell: Optional[tuple] = None,
    ) -> str:
        """Build actionable diversity guidance for the LLM operator prompt."""
        occupied, empty = self._get_archive_summary()
        archive_context = self._format_archive_context(occupied, empty)

        if operator == "draft" and target_cell:
            model_family, data_strategy = target_cell
            return (
                f"## Diversity Target\n"
                f"Generate a solution using **{model_family}** as the primary modeling "
                f"approach with **{data_strategy}** as the data handling strategy.\n\n"
                f"{archive_context}"
            )

        if operator == "improve" and selected_nodes:
            cell = self._node_cell_name(selected_nodes[0].node.id)
            if cell:
                mf, ds, score = cell
                return (
                    f"## Diversity Guidance\n"
                    f"Your parent solution uses **{mf}** with **{ds}** "
                    f"(score: {score:.4f}).\n"
                    f"Improve this solution while keeping the same general "
                    f"modeling approach and data strategy.\n\n"
                    f"{archive_context}"
                )

        if operator == "crossover" and len(selected_nodes) >= 2:
            cell_a = self._node_cell_name(selected_nodes[0].node.id)
            cell_b = self._node_cell_name(selected_nodes[1].node.id)
            if cell_a and cell_b:
                mf_a, ds_a, score_a = cell_a
                mf_b, ds_b, score_b = cell_b
                return (
                    f"## Diversity Guidance\n"
                    f"Parent A: **{mf_a}** with **{ds_a}** (score: {score_a:.4f})\n"
                    f"Parent B: **{mf_b}** with **{ds_b}** (score: {score_b:.4f})\n"
                    f"Combine the best ideas from both approaches — you may blend "
                    f"architectures, borrow preprocessing from one and modeling "
                    f"from the other.\n\n"
                    f"{archive_context}"
                )

        # Fallback: just archive context
        return archive_context

    def select(
        self,
        journal: Journal,
        context: Dict[str, Any],
    ) -> SelectionResult:
        """
        Select nodes using MAP-Elites diversity-aware selection.

        Syncs new journal nodes into the archive, then selects parents
        using the configured QD selection policy. Enriches the reasoning
        with actionable diversity guidance for the LLM operator.

        Exploration pressure: when archive coverage is low, biases toward
        draft operations targeting empty cells. draft_prob = max(0, 1 - 2*coverage).
        """
        # Lazy init
        if self._archive is None:
            self._init_qd()

        # Sync new nodes into archive
        added = self._sync_archive(journal)

        _ensure_qd_importable()
        from qd.metrics import coverage, qd_score

        archive_size = self._archive.size
        archive_coverage = coverage(self._archive)
        archive_qd_score = qd_score(self._archive)

        if self.cfg.verbose or added > 0:
            log.info(
                f"Archive state: {archive_size}/{self._archive.cell_count()} cells occupied "
                f"(coverage={archive_coverage:.1%}, qd_score={archive_qd_score:.4f})"
            )

        # Fallback to fitness selector if archive is empty
        if archive_size == 0:
            if self.cfg.fallback_to_fitness:
                log.info("Archive empty, falling back to fitness selector")
                return self._get_fitness_selector().select(journal, context)
            return SelectionResult(
                selected_nodes=[],
                operator="draft",
                reasoning="Archive empty, requesting draft",
            )

        # ── Exploration pressure: bias toward draft when coverage is low ──
        draft_prob = max(0.0, 1.0 - 2.0 * archive_coverage)
        if draft_prob > 0 and random.random() < draft_prob:
            target_cell = self._pick_target_cell()
            log.info(
                f"DRAFT -> targeting {target_cell[0]} + {target_cell[1]} "
                f"(draft_prob={draft_prob:.2f}, coverage={archive_coverage:.1%})"
            )
            reasoning = self._compose_reasoning("draft", [], target_cell=target_cell)
            return SelectionResult(
                selected_nodes=[],
                operator="draft",
                reasoning=reasoning,
                metadata={
                    "qd_coverage": archive_coverage,
                    "qd_score": archive_qd_score,
                    "qd_archive_size": archive_size,
                    "diversity_action": "explore",
                    "target_cell": list(target_cell),
                },
            )

        # ── Exploit: improve or crossover ──
        crossover_prob = context.get("crossover_prob", 0.0)
        operator = "improve" if random.random() >= crossover_prob else "crossover"
        num_parents = 2 if operator == "crossover" else 1

        # If crossover but archive has fewer than 2 elites, fall back to improve
        if operator == "crossover" and archive_size < 2:
            operator = "improve"
            num_parents = 1

        # Select parent IDs from archive
        if operator == "crossover" and archive_size >= 2:
            # Crossover: pick parents from different cells for diversity
            occupied_cells = list(self._archive.occupied_cells())
            cell_pair = random.sample(occupied_cells, 2)
            selected_ids = [cell_pair[0][1].id, cell_pair[1][1].id]
        else:
            # Improve: use configured selection policy
            self._selector.update(archive=self._archive)
            selected_ids = self._selector.sample(
                policy=self.cfg.selection_policy,
                k=num_parents,
            )

        # Map selected IDs back to journal nodes
        nodes_by_id = {node.id: node for node in journal.nodes}
        islands = context.get("islands", [])
        selected_nodes = []
        island_id = 0

        for priority, node_id in enumerate(selected_ids, start=1):
            node = nodes_by_id.get(node_id)
            if node is None:
                log.warning(f"Selected node {node_id} not found in journal")
                continue
            selected_nodes.append(
                SelectedNode(
                    step=node.step,
                    priority=priority,
                    reason=f"MAP-Elites {self.cfg.selection_policy} selection "
                           f"(fitness={self._node_fitness(node):.4f})",
                    node=node,
                )
            )
            if priority == 1:
                island_id = self._find_island_for_node(node_id, islands)

        if not selected_nodes:
            log.warning("No valid nodes selected from archive, falling back to fitness")
            return self._get_fitness_selector().select(journal, context)

        # Build rich diversity guidance for the LLM
        reasoning = self._compose_reasoning(operator, selected_nodes)
        log.info(
            f"{operator.upper()} -> {len(selected_nodes)} parent(s) "
            f"(coverage={archive_coverage:.1%})"
        )

        return SelectionResult(
            selected_nodes=selected_nodes,
            operator=operator,
            island_id=island_id,
            reasoning=reasoning,
            metadata={
                "qd_coverage": archive_coverage,
                "qd_score": archive_qd_score,
                "qd_archive_size": archive_size,
                "diversity_action": operator,
            },
        )

    def get_archive_state(self) -> Dict[str, Any]:
        """Return archive state for checkpointing/logging."""
        if self._archive is None:
            return {"initialized": False}

        _ensure_qd_importable()
        from qd.metrics import coverage, qd_score, best_fitness
        from qd.features import feature_names

        elites = []
        for cell_idx, entry in self._archive.occupied_cells():
            elites.append({
                "cell": list(cell_idx),
                "id": entry.id,
                "fitness": entry.fitness,
            })

        return {
            "initialized": True,
            "cell_count": self._archive.cell_count(),
            "archive_size": self._archive.size,
            "coverage": coverage(self._archive),
            "qd_score": qd_score(self._archive),
            "best_fitness": best_fitness(self._archive),
            "elites": elites,
        }
