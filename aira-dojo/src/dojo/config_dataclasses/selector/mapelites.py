# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for MAP-Elites-based node selector."""

from dataclasses import dataclass, field

from dojo.config_dataclasses.selector.base import SelectorConfig


@dataclass
class MAPElitesSelectorConfig(SelectorConfig):
    """
    Configuration for MAP-Elites diversity-aware node selection.

    Uses a GridArchive (model_family x data_strategy = 30 cells) to track
    solution diversity and selects parents from the archive using a
    configurable selection policy.
    """

    selector_type: str = field(
        default="mapelites",
        metadata={
            "description": "Selector type identifier",
        },
    )

    selection_policy: str = field(
        default="tournament",
        metadata={
            "description": "Selection policy: 'tournament', 'roulette', 'random', or 'best'",
            "example": "tournament",
        },
    )

    seed: int = field(
        default=42,
        metadata={
            "description": "Random seed for selection",
            "example": 42,
        },
    )

    fallback_to_fitness: bool = field(
        default=True,
        metadata={
            "description": "Fall back to fitness selector if archive is empty",
            "example": True,
        },
    )

    verbose: bool = field(
        default=False,
        metadata={
            "description": "Enable verbose logging for archive state and selection decisions",
            "example": False,
        },
    )

    def validate(self) -> None:
        super().validate()
        valid_policies = ["tournament", "roulette", "random", "best"]
        if self.selection_policy not in valid_policies:
            raise ValueError(
                f"selection_policy must be one of {valid_policies}, got {self.selection_policy}"
            )
