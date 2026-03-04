#!/usr/bin/env python3
"""
Test script for the node selection module.

Usage:
    cd aira-dojo
    conda run -n aira-dojo python -m dojo.core.solvers.selection.test_selection
"""

import sys
from pathlib import Path


def test_imports():
    """Test all module imports."""
    print("Testing imports...")

    from dojo.core.solvers.selection import (
        get_selector,
        NodeSelector,
        SelectedNode,
        SelectionResult,
        FitnessNodeSelector,
        RLMNodeSelector,
        serialize_for_rlm,
        extract_architectures,
        extract_error_info,
        generate_ascii_tree,
    )
    from dojo.config_dataclasses.selector import (
        SelectorConfig,
        FitnessSelectorConfig,
        RLMSelectorConfig,
    )

    print("  All imports OK")


def test_fitness_selector():
    """Test FitnessNodeSelector."""
    print("Testing FitnessNodeSelector...")

    from dojo.core.solvers.selection import FitnessNodeSelector

    selector = FitnessNodeSelector(lower_is_better=False, verbose=False)
    assert selector.selector_type == "fitness"
    assert selector.lower_is_better is False

    # Test normalization
    selector.update_fitness_range(0.5)
    selector.update_fitness_range(1.0)
    assert selector.get_normalized_score(0.5) == 0.0  # worst
    assert selector.get_normalized_score(1.0) == 1.0  # best
    assert selector.get_normalized_score(0.75) == 0.5  # middle
    assert selector.get_normalized_score(None) == 0.0  # None -> worst

    print("  FitnessNodeSelector OK")


def test_get_selector_factory():
    """Test the get_selector factory function."""
    print("Testing get_selector factory...")

    from dojo.core.solvers.selection import get_selector
    from dojo.config_dataclasses.selector import FitnessSelectorConfig, RLMSelectorConfig

    # Test with None (should default to fitness)
    s1 = get_selector(None, lower_is_better=False)
    assert s1.selector_type == "fitness"

    # Test with FitnessSelectorConfig
    cfg2 = FitnessSelectorConfig(verbose=True)
    s2 = get_selector(cfg2, lower_is_better=True)
    assert s2.selector_type == "fitness"
    assert s2.lower_is_better is True

    # Test with RLMSelectorConfig
    cfg3 = RLMSelectorConfig(backend="gemini", model_name="gemini-2.0-flash")
    s3 = get_selector(cfg3, lower_is_better=False)
    assert s3.selector_type == "rlm"

    print("  get_selector factory OK")


def test_tree_serializer():
    """Test tree serialization utilities."""
    print("Testing tree serializer...")

    from dojo.core.solvers.selection.tree_serializer import (
        extract_architectures,
        extract_error_info,
    )

    # Test architecture extraction
    text = "Using ConvNeXt with K-Fold cross-validation and AdamW optimizer"
    archs = extract_architectures(text)
    assert "ConvNeXt" in archs
    assert "K-Fold" in archs
    assert "AdamW" in archs

    # Test error extraction
    error_out = "RuntimeError: CUDA out of memory\nTraceback..."
    error_info = extract_error_info(error_out)
    assert error_info is not None
    assert "CUDA" in error_info

    print("  Tree serializer OK")


def test_evo_config():
    """Test that EVO config includes selector field."""
    print("Testing EVO config...")

    from dojo.config_dataclasses.solver.evo import EvolutionarySolverConfig

    fields = EvolutionarySolverConfig.__dataclass_fields__
    assert "selector" in fields
    print(f"  EVO config has selector field: {fields['selector'].default}")
    print("  EVO config OK")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Node Selection Module Tests")
    print("=" * 60)

    test_imports()
    test_fitness_selector()
    test_get_selector_factory()
    test_tree_serializer()
    test_evo_config()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    # Add src to path if running directly
    src_path = Path(__file__).parent.parent.parent.parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    main()
