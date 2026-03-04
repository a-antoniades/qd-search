# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for fitness-based node selector."""

from dataclasses import dataclass, field

from dojo.config_dataclasses.selector.base import SelectorConfig


@dataclass
class FitnessSelectorConfig(SelectorConfig):
    """
    Configuration for fitness-based node selection.

    This selector uses softmax-weighted sampling based on normalized
    fitness scores, preserving the original EVO solver behavior.
    """

    selector_type: str = field(
        default="fitness",
        metadata={
            "description": "Selector type identifier",
        },
    )

    def validate(self) -> None:
        super().validate()
