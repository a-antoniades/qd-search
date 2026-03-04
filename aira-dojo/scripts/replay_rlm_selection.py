#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CLI tool for replaying RLM selection decisions.

Enables prototyping and debugging RLM selection using saved journal data.

Examples:
    # Interactive: replay single step
    python scripts/replay_rlm_selection.py \\
        --journal logs/.../checkpoint/journal.jsonl \\
        --step 15

    # Batch: replay multiple steps, save results
    python scripts/replay_rlm_selection.py \\
        --journal logs/.../checkpoint/journal.jsonl \\
        --steps 5,10,15,20 \\
        --output replay_results.jsonl

    # Compare against original decisions
    python scripts/replay_rlm_selection.py \\
        --journal logs/.../checkpoint/journal.jsonl \\
        --history logs/.../checkpoint/selection_history.jsonl \\
        --compare

    # Use different model
    python scripts/replay_rlm_selection.py \\
        --journal logs/.../checkpoint/journal.jsonl \\
        --step 15 \\
        --model gemini-2.5-pro-preview

    # Dry run: show what would be sent to RLM without calling
    python scripts/replay_rlm_selection.py \\
        --journal logs/.../checkpoint/journal.jsonl \\
        --step 15 \\
        --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dojo.config_dataclasses.selector.rlm import RLMSelectorConfig
from dojo.core.solvers.selection.replay import (
    JournalSnapshot,
    ReplayContext,
    ReplayResult,
    build_replay_context,
    compare_selections,
    load_journal_from_jsonl,
    load_selection_history,
    replay_selection,
    snapshot_at_step,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def parse_steps(steps_str: str) -> list[int]:
    """Parse comma-separated step numbers."""
    return [int(s.strip()) for s in steps_str.split(",") if s.strip()]


def print_snapshot_info(snapshot: JournalSnapshot) -> None:
    """Print summary of snapshot state."""
    print(f"\n{'=' * 60}")
    print(f"Snapshot at step {snapshot.step}")
    print(f"{'=' * 60}")
    print(f"Total nodes: {snapshot.num_nodes}")
    print(f"Leaf nodes ({len(snapshot.leaf_steps)}): {snapshot.leaf_steps}")

    # Show top nodes by metric
    nodes = snapshot.journal.nodes
    valid_nodes = [n for n in nodes if not n.is_buggy and n.metric is not None]
    if valid_nodes:
        sorted_nodes = sorted(
            valid_nodes,
            key=lambda n: n.metric.value if hasattr(n.metric, "value") and n.metric.value is not None else float("-inf"),
            reverse=True,
        )
        print(f"\nTop 5 nodes by metric:")
        for n in sorted_nodes[:5]:
            metric_val = n.metric.value if hasattr(n.metric, "value") else None
            is_leaf = "LEAF" if n.step in snapshot.leaf_steps else ""
            metric_str = f"{metric_val:.4f}" if metric_val is not None else "N/A"
            print(f"  Step {n.step}: {metric_str} {is_leaf}")


def print_context_info(context: ReplayContext) -> None:
    """Print summary of replay context."""
    print(f"\nReplay Context:")
    print(f"  Step: {context.step}")
    print(f"  Leaf steps: {context.leaf_steps}")
    print(f"  Num samples: {context.num_samples}")
    print(f"  Tree summary: {context.tree_context.get('summary', {})}")
    print(f"\nRoot prompt:\n{context.root_prompt}")


def print_replay_result(result: ReplayResult) -> None:
    """Print replay result."""
    print(f"\n{'=' * 60}")
    print(f"Replay Result - Step {result.step}")
    print(f"{'=' * 60}")
    print(f"Model: {result.model}")
    print(f"Elapsed: {result.elapsed_s:.2f}s")

    if result.error:
        print(f"ERROR: {result.error}")
    else:
        print(f"Selected steps: {result.selected_steps}")
        print(f"Operator: {result.operator}")
        print(f"Reasoning: {result.reasoning}")


def print_comparison(comparison: dict) -> None:
    """Print comparison between original and replay."""
    print(f"\n{'=' * 60}")
    print(f"Comparison - Step {comparison['step']}")
    print(f"{'=' * 60}")
    print(f"Steps match: {'YES' if comparison['steps_match'] else 'NO'}")
    print(f"Operator match: {'YES' if comparison['operator_match'] else 'NO'}")
    print(f"Original selected: {comparison['original_selected']}")
    print(f"Replay selected: {comparison['replay_selected']}")
    print(f"Original operator: {comparison['original_operator']}")
    print(f"Replay operator: {comparison['replay_operator']}")


def run_single_replay(
    journal_path: Path,
    step: int,
    cfg: RLMSelectorConfig,
    num_samples: int = 1,
    dry_run: bool = False,
    verbose: bool = False,
) -> ReplayResult:
    """Run replay for a single step."""
    log.info(f"Loading journal from {journal_path}")
    journal = load_journal_from_jsonl(journal_path)
    log.info(f"Loaded {len(journal.nodes)} nodes")

    log.info(f"Creating snapshot at step {step}")
    snapshot = snapshot_at_step(journal, step)
    print_snapshot_info(snapshot)

    log.info("Building replay context")
    context = build_replay_context(snapshot, num_samples=num_samples, full_context=True)
    if verbose:
        print_context_info(context)

    if dry_run:
        print("\n[DRY RUN] Would send to RLM:")
        print(f"  Model: {cfg.model_name}")
        print(f"  Backend: {cfg.backend}")
        print(f"  Leaf steps: {context.leaf_steps}")
        print(f"  Prompt: {context.root_prompt[:200]}...")
        return ReplayResult(
            step=step,
            model=cfg.model_name,
            leaf_steps=context.leaf_steps,
            selected_steps=[],
            operator="",
            reasoning="[DRY RUN - no actual call made]",
            elapsed_s=0.0,
        )

    log.info("Running RLM selection")
    result = replay_selection(context, cfg)
    print_replay_result(result)
    return result


def run_batch_replay(
    journal_path: Path,
    steps: list[int],
    cfg: RLMSelectorConfig,
    output_path: Path = None,
    num_samples: int = 1,
) -> list[ReplayResult]:
    """Run replay for multiple steps."""
    log.info(f"Loading journal from {journal_path}")
    journal = load_journal_from_jsonl(journal_path)
    log.info(f"Loaded {len(journal.nodes)} nodes")

    results = []
    for step in steps:
        log.info(f"Replaying step {step}")
        try:
            snapshot = snapshot_at_step(journal, step)
            context = build_replay_context(snapshot, num_samples=num_samples, full_context=True)
            result = replay_selection(context, cfg)
            results.append(result)
            print_replay_result(result)
        except Exception as e:
            log.error(f"Failed to replay step {step}: {e}")
            results.append(
                ReplayResult(
                    step=step,
                    model=cfg.model_name,
                    leaf_steps=[],
                    selected_steps=[],
                    operator="",
                    reasoning="",
                    elapsed_s=0.0,
                    error=str(e),
                )
            )

    if output_path:
        log.info(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            for r in results:
                record = {
                    "step": r.step,
                    "model": r.model,
                    "leaf_steps": r.leaf_steps,
                    "selected_steps": r.selected_steps,
                    "operator": r.operator,
                    "reasoning": r.reasoning,
                    "elapsed_s": r.elapsed_s,
                    "error": r.error,
                }
                f.write(json.dumps(record) + "\n")

    return results


def run_comparison(
    journal_path: Path,
    history_path: Path,
    cfg: RLMSelectorConfig,
    output_path: Path = None,
) -> list[dict]:
    """Compare replay results against original selections."""
    log.info(f"Loading journal from {journal_path}")
    journal = load_journal_from_jsonl(journal_path)
    log.info(f"Loaded {len(journal.nodes)} nodes")

    log.info(f"Loading selection history from {history_path}")
    history = load_selection_history(history_path)
    log.info(f"Loaded {len(history)} selection records")

    comparisons = []
    for record in history:
        # Get the step at which selection was made
        # Use selection_index to determine roughly what step we were at
        # The tree_summary has total_nodes which approximates step
        tree_summary = record.get("tree_summary", {})
        step = tree_summary.get("total_nodes", 0)
        if step == 0:
            continue

        log.info(f"Replaying selection at ~step {step}")
        try:
            snapshot = snapshot_at_step(journal, step)
            context = build_replay_context(snapshot, full_context=True)
            result = replay_selection(context, cfg)
            comparison = compare_selections(result, record)
            comparisons.append(comparison)
            print_comparison(comparison)
        except Exception as e:
            log.error(f"Failed to compare at step {step}: {e}")

    if output_path:
        log.info(f"Saving comparisons to {output_path}")
        with open(output_path, "w") as f:
            for c in comparisons:
                f.write(json.dumps(c) + "\n")

    # Summary statistics
    if comparisons:
        steps_matches = sum(1 for c in comparisons if c["steps_match"])
        operator_matches = sum(1 for c in comparisons if c["operator_match"])
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"Total comparisons: {len(comparisons)}")
        print(f"Steps match: {steps_matches}/{len(comparisons)} ({100*steps_matches/len(comparisons):.1f}%)")
        print(f"Operator match: {operator_matches}/{len(comparisons)} ({100*operator_matches/len(comparisons):.1f}%)")

    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description="Replay RLM selection decisions using saved journal data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--journal",
        type=Path,
        required=True,
        help="Path to journal.jsonl checkpoint file",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Single step to replay",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of steps to replay (e.g., '5,10,15,20')",
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="Path to selection_history.jsonl for comparison mode",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare replay results against original selections",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for results (JSONL format)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Model name for RLM (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        choices=["gemini", "openai", "anthropic"],
        help="RLM backend (default: gemini)",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="GEMINI_API_KEY",
        help="Environment variable for API key (default: GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of nodes to select (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent to RLM without making actual call",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including full context",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.journal.exists():
        log.error(f"Journal file not found: {args.journal}")
        sys.exit(1)

    if args.compare and not args.history:
        log.error("--compare requires --history")
        sys.exit(1)

    if not args.step and not args.steps and not args.compare:
        log.error("Must specify --step, --steps, or --compare")
        sys.exit(1)

    # Build config
    cfg = RLMSelectorConfig(
        backend=args.backend,
        model_name=args.model,
        api_key_env=args.api_key_env,
        verbose=args.verbose,
    )

    # Run appropriate mode
    if args.compare:
        run_comparison(
            args.journal,
            args.history,
            cfg,
            args.output,
        )
    elif args.steps:
        steps = parse_steps(args.steps)
        run_batch_replay(
            args.journal,
            steps,
            cfg,
            args.output,
            args.num_samples,
        )
    else:
        run_single_replay(
            args.journal,
            args.step,
            cfg,
            args.num_samples,
            args.dry_run,
            args.verbose,
        )


if __name__ == "__main__":
    main()
