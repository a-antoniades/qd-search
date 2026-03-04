# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Node selection module for evolutionary search.

This module provides different strategies for selecting which nodes
in the search tree should be expanded next.

Available selectors:
- FitnessNodeSelector: Softmax-weighted selection based on fitness scores
- MAPElitesNodeSelector: Diversity-aware selection using a MAP-Elites archive
- RLMNodeSelector: Intelligent selection using a language model

Usage:
    from dojo.core.solvers.selection import get_selector

    selector = get_selector(cfg, lower_is_better=False)
    result = selector.select(journal, context)
"""

from typing import Union

from dojo.core.solvers.selection.base import (
    NodeSelector,
    SelectedNode,
    SelectionResult,
)
from dojo.core.solvers.selection.fitness_selector import FitnessNodeSelector
from dojo.core.solvers.selection.mapelites_selector import MAPElitesNodeSelector
from dojo.core.solvers.selection.rlm_selector import RLMNodeSelector
from dojo.core.solvers.selection.tree_serializer import (
    serialize_for_rlm,
    generate_ascii_tree,
)
from dojo.core.solvers.selection.replay import (
    JournalSnapshot,
    ReplayContext,
    ReplayResult,
    load_journal_from_jsonl,
    snapshot_at_step,
    build_replay_context,
    replay_selection,
    load_selection_history,
    compare_selections,
)

from dojo.config_dataclasses.selector.base import SelectorConfig
from dojo.config_dataclasses.selector.fitness import FitnessSelectorConfig
from dojo.config_dataclasses.selector.mapelites import MAPElitesSelectorConfig
from dojo.config_dataclasses.selector.rlm import RLMSelectorConfig


def get_selector(
    cfg: Union[SelectorConfig, FitnessSelectorConfig, MAPElitesSelectorConfig, RLMSelectorConfig, None],
    lower_is_better: bool,
) -> NodeSelector:
    """
    Factory function to create a node selector from configuration.

    Args:
        cfg: Selector configuration (SelectorConfig, FitnessSelectorConfig, RLMSelectorConfig, or None).
              If None, defaults to fitness selector.
        lower_is_better: If True, lower metric values are better.

    Returns:
        NodeSelector instance.

    Raises:
        ValueError: If selector_type is unknown.
    """
    # Default to fitness selector if no config provided
    if cfg is None:
        return FitnessNodeSelector(
            lower_is_better=lower_is_better,
            verbose=False,
        )

    selector_type = getattr(cfg, "selector_type", "fitness")

    if selector_type == "fitness":
        return FitnessNodeSelector(
            lower_is_better=lower_is_better,
            verbose=getattr(cfg, "verbose", False),
        )
    elif selector_type == "mapelites":
        if not isinstance(cfg, MAPElitesSelectorConfig):
            raise ValueError(
                f"MAP-Elites selector requires MAPElitesSelectorConfig, got {type(cfg).__name__}"
            )
        return MAPElitesNodeSelector(
            cfg=cfg,
            lower_is_better=lower_is_better,
        )
    elif selector_type == "rlm":
        if not isinstance(cfg, RLMSelectorConfig):
            raise ValueError(
                f"RLM selector requires RLMSelectorConfig, got {type(cfg).__name__}"
            )
        return RLMNodeSelector(
            cfg=cfg,
            lower_is_better=lower_is_better,
        )
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")


__all__ = [
    # Base classes
    "NodeSelector",
    "SelectedNode",
    "SelectionResult",
    # Selector implementations
    "FitnessNodeSelector",
    "MAPElitesNodeSelector",
    "RLMNodeSelector",
    # Factory
    "get_selector",
    # Serialization utilities
    "serialize_for_rlm",
    "generate_ascii_tree",
    # Replay system
    "JournalSnapshot",
    "ReplayContext",
    "ReplayResult",
    "load_journal_from_jsonl",
    "snapshot_at_step",
    "build_replay_context",
    "replay_selection",
    "load_selection_history",
    "compare_selections",
]
