#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Run RLM selection replays at identified key junctions.

Takes a list of junctions from identify_key_junctions.py and runs RLM
selection at each point, comparing against what the fitness selector
originally chose.

Usage:
    python -m scripts.run_rlm_replay \
        --junctions experiments/rlm_replay_results/junctions.json \
        --output experiments/rlm_replay_results/replays.json
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dojo.config_dataclasses.selector.rlm import RLMSelectorConfig
from dojo.core.solvers.selection.replay import (
    build_replay_context,
    load_journal_from_jsonl,
    replay_selection,
    snapshot_at_step,
)
from dojo.core.solvers.utils.journal import Journal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


@dataclass
class OriginalSelection:
    """What the fitness selector originally chose."""
    operator: str
    selected_steps: List[int]
    child_step: int
    child_metric: Optional[float]
    child_is_buggy: bool


@dataclass
class ReplayComparison:
    """Comparison between RLM and original selection."""
    task: str
    step: int
    junction_type: str
    description: str

    # Original selection (from journal parent relationships)
    original: Dict[str, Any]

    # RLM selection (from replay)
    rlm: Dict[str, Any]

    # Comparison
    different: bool
    elapsed_s: float
    error: Optional[str] = None


def infer_original_selection(journal: Journal, target_step: int) -> Optional[OriginalSelection]:
    """
    Infer what the fitness selector originally chose at a given step.

    We look at the node at target_step and check its parents to understand
    what was selected.
    """
    # Find the node that was created at/after target_step
    # This represents the result of the selection at target_step
    child_node = None
    for node in journal.nodes:
        if node.step > target_step:
            child_node = node
            break

    if not child_node:
        # If no child after target_step, this might be a plateau at the end
        # Try to use the node at target_step itself as context
        for node in journal.nodes:
            if node.step == target_step:
                # Get parent info from this node
                parent_steps = [p.step for p in node.parents if p] if node.parents else []
                operator = "improve" if len(parent_steps) == 1 else ("crossover" if len(parent_steps) > 1 else "draft")

                child_metric = None
                if hasattr(node.metric, 'value') and node.metric.value is not None:
                    child_metric = node.metric.value

                return OriginalSelection(
                    operator=operator,
                    selected_steps=parent_steps,
                    child_step=node.step,
                    child_metric=child_metric,
                    child_is_buggy=node.is_buggy,
                )
        return None

    # Get parent steps
    parent_steps = [p.step for p in child_node.parents if p] if child_node.parents else []

    # Infer operator from number of parents
    if len(parent_steps) == 0:
        operator = "draft"
    elif len(parent_steps) == 1:
        # Check if parent was buggy (debug) or good (improve)
        parent = child_node.parents[0]
        if parent and parent.is_buggy:
            operator = "debug"
        else:
            operator = "improve"
    else:
        operator = "crossover"

    # Get child metric
    child_metric = None
    if hasattr(child_node.metric, 'value') and child_node.metric.value is not None:
        child_metric = child_node.metric.value

    return OriginalSelection(
        operator=operator,
        selected_steps=parent_steps,
        child_step=child_node.step,
        child_metric=child_metric,
        child_is_buggy=child_node.is_buggy,
    )


def replay_junction(
    junction: Dict[str, Any],
    cfg: RLMSelectorConfig,
    max_retries: int = 3,
) -> ReplayComparison:
    """
    Run RLM replay at a single junction and compare with original.
    """
    task = junction["task"]
    step = junction["step"]
    junction_type = junction["junction_type"]
    description = junction["description"]
    journal_path = junction["journal_path"]
    lower_is_better = junction.get("lower_is_better", False)

    log.info(f"Replaying {task} at step {step} ({junction_type})")

    start_time = time.time()

    try:
        # Load journal
        journal = load_journal_from_jsonl(Path(journal_path))

        # Get original selection
        original = infer_original_selection(journal, step)
        if not original:
            raise ValueError(f"Could not infer original selection at step {step}")

        # Create snapshot at step
        snapshot = snapshot_at_step(journal, step)

        # Build replay context
        # Determine step_limit from journal
        step_limit = max(n.step for n in journal.nodes) + 1
        context = build_replay_context(
            snapshot,
            num_samples=1,
            full_context=True,
            lower_is_better=lower_is_better,
            step_limit=step_limit,
        )

        # Run RLM selection with retries for transient API errors
        result = None
        last_error = None
        for attempt in range(max_retries):
            try:
                result = replay_selection(context, cfg, lower_is_better=lower_is_better)
                if not result.error:
                    break
                # If it's a 500 error, retry
                if "500" in str(result.error) or "INTERNAL" in str(result.error):
                    log.warning(f"Attempt {attempt + 1}/{max_retries} failed with API error, retrying...")
                    last_error = result.error
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                break
            except Exception as e:
                last_error = str(e)
                if "500" in str(e) or "INTERNAL" in str(e):
                    log.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                raise

        if result is None or (result.error and last_error):
            raise Exception(last_error or "RLM selection failed after retries")

        elapsed = time.time() - start_time

        # Check if selections differ
        rlm_steps = set(result.selected_steps)
        original_steps = set(original.selected_steps)

        # Normalize operator names
        rlm_operator = result.operator
        original_operator = original.operator
        if original_operator == "debug":
            original_operator = "improve"  # Normalize for comparison

        different = (rlm_steps != original_steps) or (rlm_operator != original_operator)

        return ReplayComparison(
            task=task,
            step=step,
            junction_type=junction_type,
            description=description,
            original={
                "operator": original.operator,
                "selected_steps": original.selected_steps,
                "child_step": original.child_step,
                "child_metric": original.child_metric,
                "child_is_buggy": original.child_is_buggy,
            },
            rlm={
                "operator": result.operator,
                "selected_steps": result.selected_steps,
                "reasoning": result.reasoning,
                "raw_response": result.raw_response[:1000] if result.raw_response else "",
            },
            different=different,
            elapsed_s=elapsed,
            error=result.error,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"Failed to replay {task} at step {step}: {e}")
        import traceback
        traceback.print_exc()

        return ReplayComparison(
            task=task,
            step=step,
            junction_type=junction_type,
            description=description,
            original={},
            rlm={},
            different=False,
            elapsed_s=elapsed,
            error=str(e),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run RLM selection replays at identified key junctions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--junctions",
        type=Path,
        required=True,
        help="Path to junctions JSON from identify_key_junctions.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for replay results JSON",
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
        "--limit",
        type=int,
        help="Maximum number of junctions to replay",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making RLM calls",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    if not args.junctions.exists():
        log.error(f"Junctions file not found: {args.junctions}")
        sys.exit(1)

    # Load junctions
    with open(args.junctions) as f:
        data = json.load(f)

    junctions = data.get("junctions", [])
    log.info(f"Loaded {len(junctions)} junctions")

    if args.limit:
        junctions = junctions[:args.limit]
        log.info(f"Limited to {len(junctions)} junctions")

    if not junctions:
        log.error("No junctions to replay")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Would replay these junctions:")
        for j in junctions:
            print(f"  {j['task']} step {j['step']}: {j['junction_type']}")
            print(f"    {j['description']}")
        return

    # Build RLM config
    cfg = RLMSelectorConfig(
        backend=args.backend,
        model_name=args.model,
        api_key_env=args.api_key_env,
        verbose=args.verbose,
    )

    # Run replays
    results = []
    for i, junction in enumerate(junctions):
        log.info(f"[{i+1}/{len(junctions)}] Processing {junction['task']}")
        result = replay_junction(junction, cfg)
        results.append(result)

        # Print result
        print(f"\n{'=' * 60}")
        print(f"{result.task} - Step {result.step}")
        print(f"{'=' * 60}")
        print(f"Junction type: {result.junction_type}")
        print(f"Description: {result.description}")
        print()

        if result.error:
            print(f"ERROR: {result.error}")
        else:
            print(f"Original selection:")
            print(f"  Operator: {result.original.get('operator', 'N/A')}")
            print(f"  Selected steps: {result.original.get('selected_steps', [])}")
            print(f"  Child step: {result.original.get('child_step', 'N/A')}")
            print(f"  Child metric: {result.original.get('child_metric', 'N/A')}")
            print(f"  Child buggy: {result.original.get('child_is_buggy', 'N/A')}")
            print()
            print(f"RLM selection:")
            print(f"  Operator: {result.rlm.get('operator', 'N/A')}")
            print(f"  Selected steps: {result.rlm.get('selected_steps', [])}")
            print(f"  Reasoning: {result.rlm.get('reasoning', 'N/A')[:200]}")
            print()
            print(f"Different: {'YES' if result.different else 'NO'}")
            print(f"Elapsed: {result.elapsed_s:.2f}s")

        # Small delay between replays to avoid rate limits
        if i < len(junctions) - 1:
            time.sleep(1)

    # Save results
    output_data = {
        "config": {
            "model": args.model,
            "backend": args.backend,
        },
        "summary": {
            "total_replays": len(results),
            "successful": len([r for r in results if not r.error]),
            "different_choices": len([r for r in results if r.different and not r.error]),
            "errors": len([r for r in results if r.error]),
        },
        "replays": [asdict(r) for r in results],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    log.info(f"Saved {len(results)} replay results to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("Replay Summary")
    print("=" * 60)
    print(f"Total replays: {len(results)}")
    print(f"Successful: {len([r for r in results if not r.error])}")
    print(f"Different choices: {len([r for r in results if r.different and not r.error])}")
    print(f"Errors: {len([r for r in results if r.error])}")


if __name__ == "__main__":
    main()
