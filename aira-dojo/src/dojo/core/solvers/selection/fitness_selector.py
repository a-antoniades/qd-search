# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Fitness-based node selection.

This selector implements the original EVO solver selection logic:
- Select island based on average fitness (softmax weighted)
- Select nodes within island based on individual fitness
- Support for improve and crossover operators
"""

import logging
import random
from typing import Any, Dict, List, Optional

import numpy

from dojo.core.solvers.selection.base import NodeSelector, SelectedNode, SelectionResult
from dojo.core.solvers.utils.journal import Journal, Node
import dojo.core.solvers.utils.search_utils as utils

log = logging.getLogger(__name__)


class FitnessNodeSelector(NodeSelector):
    """
    Fitness-based node selector.

    Selects nodes using softmax-weighted sampling based on normalized
    fitness scores. This preserves the original EVO solver behavior.
    """

    def __init__(
        self,
        lower_is_better: bool,
        verbose: bool = False,
    ):
        """
        Initialize the fitness selector.

        Args:
            lower_is_better: If True, lower metric values are better.
            verbose: If True, log detailed selection information.
        """
        self.lower_is_better = lower_is_better
        self.verbose = verbose

        # Track global fitness range for normalization
        self.global_min_fitness = float("inf")
        self.global_max_fitness = float("-inf")

    @property
    def selector_type(self) -> str:
        return "fitness"

    def update_fitness_range(self, score: float) -> None:
        """Update the global min/max fitness scores."""
        if numpy.isfinite(score):
            self.global_min_fitness = min(self.global_min_fitness, score)
            self.global_max_fitness = max(self.global_max_fitness, score)

    def get_normalized_score(self, score: Optional[float]) -> float:
        """
        Normalize a fitness score to [0, 1], where 1.0 is always best.

        Args:
            score: Raw fitness score (can be None).

        Returns:
            Normalized score in [0, 1].
        """
        if score is None or not numpy.isfinite(score):
            return 0.0

        if (
            not numpy.isfinite(self.global_min_fitness)
            or not numpy.isfinite(self.global_max_fitness)
            or self.global_min_fitness == self.global_max_fitness
        ):
            return 0.5

        if self.lower_is_better:
            normalized = (self.global_max_fitness - score) / (
                self.global_max_fitness - self.global_min_fitness
            )
        else:
            normalized = (score - self.global_min_fitness) / (
                self.global_max_fitness - self.global_min_fitness
            )

        return float(numpy.clip(normalized, 0.0, 1.0))

    def select(
        self,
        journal: Journal,
        context: Dict[str, Any],
    ) -> SelectionResult:
        """
        Select nodes using fitness-weighted sampling.

        Expected context keys:
            - temperature: Sampling temperature (float)
            - crossover_prob: Probability of crossover (float)
            - num_samples: Dict mapping operator -> num samples
            - islands: List of Island objects

        Returns:
            SelectionResult with selected nodes, operator, and island_id.
        """
        temperature = context.get("temperature", 1.0)
        crossover_prob = context.get("crossover_prob", 0.0)
        num_samples = context.get("num_samples", {"improve": 1, "crossover": 2})
        islands = context.get("islands", [])

        # Update fitness range from islands
        for island in islands:
            for node in island.nodes:
                if node.metric is not None and node.metric.value is not None:
                    self.update_fitness_range(node.metric.value)

        # Check if any islands have nodes
        if not any(island.size > 0 for island in islands):
            log.warning("All islands empty, falling back to draft")
            return SelectionResult(
                selected_nodes=[],
                operator="draft",
                island_id=0,
                reasoning="All islands empty",
            )

        # Calculate island sampling weights based on average fitness
        island_avg_scores = []
        for island in islands:
            if island.size > 0:
                island_avg_scores.append(island.average_fitness_score)
            else:
                worst = float("inf") if self.lower_is_better else float("-inf")
                island_avg_scores.append(worst)

        normalized_island_scores = [
            self.get_normalized_score(s) for s in island_avg_scores
        ]
        island_sampling_weights = utils.normalized(normalized_island_scores, temp=temperature)

        if sum(island_sampling_weights) == 0:
            log.warning("Island weights all zero, using uniform")
            island_sampling_weights = [1.0 / len(islands)] * len(islands)

        # Determine operator
        operator = "improve" if random.random() >= crossover_prob else "crossover"
        num_in_context = num_samples.get(operator, 1)

        # Sample island and nodes
        attempts = 0
        max_attempts = len(islands) * 2

        while attempts < max_attempts:
            attempts += 1

            # Sample island
            sampled_island_id = random.choices(
                range(len(islands)), weights=island_sampling_weights, k=1
            )[0]
            sampled_island = islands[sampled_island_id]

            # Check island has enough nodes
            if sampled_island.size < num_in_context:
                if self.verbose:
                    log.debug(
                        f"Island {sampled_island_id} size {sampled_island.size} < {num_in_context}"
                    )
                continue

            # Sample nodes within island
            island_node_scores = sampled_island.fitness_scores
            normalized_node_scores = [
                self.get_normalized_score(s) for s in island_node_scores
            ]
            node_sampling_weights = utils.normalized(normalized_node_scores, temperature)

            if sum(node_sampling_weights) == 0:
                log.warning(f"Node weights zero on island {sampled_island_id}, uniform fallback")
                indices = numpy.random.choice(
                    range(sampled_island.size), size=num_in_context, replace=False
                )
            else:
                try:
                    indices = numpy.random.choice(
                        range(sampled_island.size),
                        p=node_sampling_weights,
                        size=num_in_context,
                        replace=False,
                    )
                except ValueError as e:
                    log.error(f"Sampling error: {e}, falling back to uniform")
                    indices = numpy.random.choice(
                        range(sampled_island.size), size=num_in_context, replace=False
                    )

            in_context_nodes = [sampled_island.nodes[i] for i in indices]

            # Build result
            selected_nodes = [
                SelectedNode(
                    step=node.step,
                    priority=i + 1,
                    reason=f"Fitness-weighted selection (score: {node.metric.value if node.metric else None})",
                    node=node,
                )
                for i, node in enumerate(in_context_nodes)
            ]

            return SelectionResult(
                selected_nodes=selected_nodes,
                operator=operator,
                island_id=sampled_island_id,
                reasoning=f"Sampled {len(selected_nodes)} nodes from island {sampled_island_id} for {operator}",
                metadata={
                    "temperature": temperature,
                    "island_weights": island_sampling_weights,
                },
            )

        # Should not reach here
        raise RuntimeError(
            f"Failed to sample nodes after {max_attempts} attempts"
        )
