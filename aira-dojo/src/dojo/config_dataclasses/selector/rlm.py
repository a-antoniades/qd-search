# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for RLM-based node selector."""

from dataclasses import dataclass, field

from omegaconf import SI

from dojo.config_dataclasses.selector.base import SelectorConfig


@dataclass
class RLMSelectorConfig(SelectorConfig):
    """
    Configuration for RLM-based intelligent node selection.

    This selector uses a language model to analyze the search tree
    and select promising nodes based on plans, architectures, and metrics.
    """

    selector_type: str = field(
        default="rlm",
        metadata={
            "description": "Selector type identifier",
        },
    )

    # RLM backend configuration
    backend: str = field(
        default="gemini",
        metadata={
            "description": "RLM backend: 'gemini', 'openai', or 'anthropic'",
            "example": "gemini",
        },
    )

    model_name: str = field(
        default="gemini-3-flash-preview",
        metadata={
            "description": "Model name for the RLM backend",
            "example": "gemini-3-flash-preview",
        },
    )

    api_key_env: str = field(
        default="GEMINI_API_KEY",
        metadata={
            "description": "Environment variable name containing the API key",
            "exclude_from_hash": True,
        },
    )

    # RLM execution parameters
    max_iterations: int = field(
        default=10,
        metadata={
            "description": "Maximum iterations for RLM execution",
            "example": 10,
        },
    )

    timeout: int = field(
        default=120,
        metadata={
            "description": "Timeout in seconds for RLM execution",
            "example": 120,
            "exclude_from_hash": True,
        },
    )

    # Context configuration
    max_nodes: int = field(
        default=100,
        metadata={
            "description": "Maximum number of nodes to include in RLM context",
            "example": 100,
        },
    )

    full_context: bool = field(
        default=True,
        metadata={
            "description": "If True, include all node data (plan, code, term_out). If False, just essentials.",
            "example": True,
        },
    )

    # Fallback behavior
    fallback_to_fitness: bool = field(
        default=True,
        metadata={
            "description": "Fall back to fitness selector if RLM fails",
            "example": True,
        },
    )

    def validate(self) -> None:
        super().validate()
        valid_backends = ["gemini", "openai", "anthropic"]
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got {self.backend}")
