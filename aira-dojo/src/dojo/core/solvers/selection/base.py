# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Base classes for node selection in evolutionary search.

This module provides the abstraction layer for node selection strategies,
allowing different selection methods (fitness-based, RLM-based) to be
swapped interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dojo.core.solvers.utils.journal import Journal, Node


@dataclass
class SelectedNode:
    """A node selected for expansion with metadata about the selection."""

    step: int
    priority: int
    reason: str
    node: Optional[Node] = None

    def __post_init__(self):
        if self.node is not None and self.step != self.node.step:
            raise ValueError(f"Step mismatch: SelectedNode.step={self.step} != Node.step={self.node.step}")


@dataclass
class SelectionResult:
    """
    Result of a node selection operation.

    Contains the selected nodes, the operator to apply, and metadata
    about the selection decision.
    """

    selected_nodes: List[SelectedNode]
    operator: str  # "improve" | "crossover" | "draft"
    island_id: Optional[int] = None
    reasoning: str = ""
    tree_insights: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def nodes(self) -> List[Node]:
        """Return the actual Node objects from selection."""
        return [sn.node for sn in self.selected_nodes if sn.node is not None]


class NodeSelector(ABC):
    """
    Abstract base class for node selection strategies.

    Implementations decide which nodes from the search tree should be
    selected for expansion (improvement, crossover, etc.).
    """

    @abstractmethod
    def select(
        self,
        journal: Journal,
        context: Dict[str, Any],
    ) -> SelectionResult:
        """
        Select nodes for the next evolutionary operation.

        Args:
            journal: The search journal containing all nodes.
            context: Additional context for selection, may include:
                - temperature: Sampling temperature
                - crossover_prob: Probability of crossover operation
                - num_samples: Dict with number of samples per operator
                - islands: List of Island objects (for fitness selector)
                - operator: Requested operator type (for RLM selector)

        Returns:
            SelectionResult containing selected nodes and operator.
        """
        pass

    @property
    @abstractmethod
    def selector_type(self) -> str:
        """Return the type identifier for this selector."""
        pass
