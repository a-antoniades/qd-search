# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base configuration for node selectors."""

from dataclasses import dataclass, field

from aira_core.config.base import BaseConfig


@dataclass
class SelectorConfig(BaseConfig):
    """Base configuration for node selection strategies."""

    selector_type: str = field(
        default="fitness",
        metadata={
            "description": "Type of node selector: 'fitness' or 'rlm'",
            "example": "fitness",
        },
    )

    verbose: bool = field(
        default=False,
        metadata={
            "description": "Enable verbose logging for selection process",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()
        valid_types = ["fitness", "rlm", "mapelites"]
        if self.selector_type not in valid_types:
            raise ValueError(f"selector_type must be one of {valid_types}, got {self.selector_type}")
