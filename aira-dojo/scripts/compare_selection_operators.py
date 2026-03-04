#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare EVO vs RLM selection by generating operator outputs for both.

Finds junctions where:
1. Original operation was improve (not debug) or crossover
2. RLM would have chosen differently

Then generates the plan/idea for both choices to compare.

Usage:
    python -m scripts.compare_selection_operators \
        --logs_dir logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm \
        --output experiments/rlm_replay_results/comparison.md \
        --limit 5
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dojo.core.solvers.selection.replay import (
    load_journal_from_jsonl,
    snapshot_at_step,
    build_replay_context,
    replay_selection,
)
from dojo.config_dataclasses.selector.rlm import RLMSelectorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


@dataclass
class SelectionJunction:
    """A junction point suitable for comparison."""
    task: str
    journal_path: str
    step: int
    original_operator: str  # improve, crossover, draft
    original_parents: List[int]
    original_child_step: int
    original_child_metric: Optional[float]
    best_metric_at_step: float
    best_node_at_step: int
    num_leaves: int
    lower_is_better: bool = False


def get_task_name(journal_path: Path) -> str:
    """Get task name from config file."""
    config_path = journal_path.parent.parent / "dojo_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("task", {}).get("name", "unknown")
    return "unknown"


def find_improve_crossover_junctions(
    journal_path: Path,
    min_step: int = 5,
    max_junctions: int = 3,
) -> List[SelectionJunction]:
    """
    Find junctions where original operation was improve or crossover (not debug).

    Returns junctions where:
    - Parent was not buggy (so it's improve, not debug)
    - OR there were 2 parents (crossover)
    """
    journal = load_journal_from_jsonl(journal_path)
    task_name = get_task_name(journal_path)

    junctions = []
    nodes_by_step = {n.step: n for n in journal.nodes}

    for node in journal.nodes:
        if node.step < min_step:
            continue
        if not node.parents:
            continue

        parent_steps = [p.step for p in node.parents if p]
        if not parent_steps:
            continue

        # Determine original operator
        if len(parent_steps) == 0:
            operator = "draft"
        elif len(parent_steps) >= 2:
            operator = "crossover"
        else:
            # Check if parent was buggy
            parent = node.parents[0]
            if parent and parent.is_buggy:
                operator = "debug"
            else:
                operator = "improve"

        # Skip debug operations
        if operator == "debug":
            continue

        # Get child metric
        child_metric = None
        if hasattr(node.metric, 'value') and node.metric.value is not None:
            child_metric = node.metric.value

        # Find best node at the time of selection (before this node was created)
        valid_nodes = [
            n for n in journal.nodes
            if n.step < node.step
            and not n.is_buggy
            and hasattr(n.metric, 'value')
            and n.metric.value is not None
        ]

        if not valid_nodes:
            continue

        best_node = max(valid_nodes, key=lambda n: n.metric.value)

        # Count leaves at this step
        leaves_at_step = []
        for n in journal.nodes:
            if n.step >= node.step:
                continue
            is_leaf = all(c.step >= node.step for c in n.children) if n.children else True
            if is_leaf and not n.is_buggy:
                leaves_at_step.append(n)

        junction = SelectionJunction(
            task=task_name,
            journal_path=str(journal_path),
            step=node.step,
            original_operator=operator,
            original_parents=parent_steps,
            original_child_step=node.step,
            original_child_metric=child_metric,
            best_metric_at_step=best_node.metric.value,
            best_node_at_step=best_node.step,
            num_leaves=len(leaves_at_step),
        )
        junctions.append(junction)

        if len(junctions) >= max_junctions:
            break

    return junctions


def run_rlm_selection(
    journal_path: Path,
    step: int,
    cfg: RLMSelectorConfig,
    lower_is_better: bool = False,
) -> Dict[str, Any]:
    """Run RLM selection at a specific step."""
    journal = load_journal_from_jsonl(journal_path)
    snapshot = snapshot_at_step(journal, step - 1)  # State before this step

    # Get step limit from journal
    step_limit = max(n.step for n in journal.nodes) + 1

    context = build_replay_context(
        snapshot,
        num_samples=1,
        full_context=True,
        lower_is_better=lower_is_better,
        step_limit=step_limit,
    )

    result = replay_selection(context, cfg, lower_is_better=lower_is_better)

    return {
        "operator": result.operator,
        "selected_steps": result.selected_steps,
        "reasoning": result.reasoning,
        "error": result.error,
    }


def generate_operator_plan(
    journal_path: Path,
    operator: str,
    parent_steps: List[int],
    step: int,
    model: str = "gemini-3-flash-preview",
) -> str:
    """Generate the plan that would result from an operator choice."""
    import litellm

    journal = load_journal_from_jsonl(journal_path)
    nodes_by_step = {n.step: n for n in journal.nodes}

    # Get task description
    task_desc = get_task_name(journal_path)

    if operator == "draft":
        prompt = f"""You are a Kaggle Grandmaster. Generate a NEW solution idea for:

Task: {task_desc}

Propose a novel approach that hasn't been tried yet. Focus on the key idea and why it would work.

# Proposed Approach (3-5 sentences)
"""
    elif operator == "improve" and parent_steps:
        parent = nodes_by_step.get(parent_steps[0])
        if not parent:
            return "Error: Parent node not found"

        prompt = f"""You are a Kaggle Grandmaster. Improve this existing solution:

Task: {task_desc}

Current approach (Step {parent.step}, metric: {parent.metric.value if hasattr(parent.metric, 'value') else 'N/A'}):
{parent.plan[:1500] if parent.plan else '(No plan)'}

Propose specific improvements. Focus on what to change and why.

# Improvement Plan (3-5 sentences)
"""
    elif operator == "crossover" and len(parent_steps) >= 2:
        parent1 = nodes_by_step.get(parent_steps[0])
        parent2 = nodes_by_step.get(parent_steps[1])
        if not parent1 or not parent2:
            return "Error: Parent nodes not found"

        prompt = f"""You are a Kaggle Grandmaster. Combine these two solutions:

Task: {task_desc}

Solution 1 (Step {parent1.step}, metric: {parent1.metric.value if hasattr(parent1.metric, 'value') else 'N/A'}):
{parent1.plan[:1000] if parent1.plan else '(No plan)'}

Solution 2 (Step {parent2.step}, metric: {parent2.metric.value if hasattr(parent2.metric, 'value') else 'N/A'}):
{parent2.plan[:1000] if parent2.plan else '(No plan)'}

Propose how to combine them effectively. Focus on the synergy.

# Crossover Plan (3-5 sentences)
"""
    else:
        return f"Unknown operator: {operator}"

    try:
        response = litellm.completion(
            model=f"gemini/{model}",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating plan: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare EVO vs RLM selection by generating operator outputs.",
    )

    parser.add_argument(
        "--logs_dir",
        type=Path,
        required=True,
        help="Directory containing EVO run logs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output markdown file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of comparisons",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Model for RLM and plan generation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find junctions without making LLM calls",
    )

    args = parser.parse_args()

    # Find all journals
    journals = list(args.logs_dir.rglob("checkpoint/journal.jsonl"))
    log.info(f"Found {len(journals)} journals")

    # Find junctions from each journal
    all_junctions = []
    for journal_path in journals:
        junctions = find_improve_crossover_junctions(journal_path, min_step=10, max_junctions=2)
        all_junctions.extend(junctions)
        if junctions:
            log.info(f"{get_task_name(journal_path)}: found {len(junctions)} improve/crossover junctions")

    log.info(f"Total junctions found: {len(all_junctions)}")

    # Sort by diversity (different tasks first)
    seen_tasks = set()
    diverse_junctions = []
    for j in all_junctions:
        if j.task not in seen_tasks:
            diverse_junctions.append(j)
            seen_tasks.add(j.task)
    # Add remaining
    for j in all_junctions:
        if j not in diverse_junctions:
            diverse_junctions.append(j)

    junctions = diverse_junctions[:args.limit]
    log.info(f"Selected {len(junctions)} junctions for comparison")

    if args.dry_run:
        print("\n[DRY RUN] Would compare these junctions:\n")
        for j in junctions:
            print(f"  {j.task} step {j.step}: {j.original_operator} {j.original_parents}")
            print(f"    Best at step: {j.best_node_at_step} ({j.best_metric_at_step:.4f})")
            print(f"    Child metric: {j.original_child_metric}")
            print()
        return

    # RLM config
    cfg = RLMSelectorConfig(
        backend="gemini",
        model_name=args.model,
        api_key_env="GEMINI_API_KEY",
    )

    # Generate comparisons
    results = []

    for i, junction in enumerate(junctions):
        log.info(f"[{i+1}/{len(junctions)}] Processing {junction.task} step {junction.step}")

        result = {
            "junction": asdict(junction),
            "evo": {},
            "rlm": {},
            "gold": {},
        }

        journal_path = Path(junction.journal_path)

        # EVO's actual choice
        result["evo"]["operator"] = junction.original_operator
        result["evo"]["selected"] = junction.original_parents
        result["evo"]["child_metric"] = junction.original_child_metric

        # Generate EVO's plan
        log.info(f"  Generating EVO plan ({junction.original_operator})...")
        result["evo"]["plan"] = generate_operator_plan(
            journal_path,
            junction.original_operator,
            junction.original_parents,
            junction.step,
            args.model,
        )
        time.sleep(1)

        # RLM's choice
        log.info(f"  Running RLM selection...")
        try:
            rlm_result = run_rlm_selection(journal_path, junction.step, cfg)
            result["rlm"]["operator"] = rlm_result["operator"]
            result["rlm"]["selected"] = rlm_result["selected_steps"]
            result["rlm"]["reasoning"] = rlm_result["reasoning"]
            result["rlm"]["error"] = rlm_result["error"]

            if not rlm_result["error"]:
                # Generate RLM's plan
                log.info(f"  Generating RLM plan ({rlm_result['operator']})...")
                result["rlm"]["plan"] = generate_operator_plan(
                    journal_path,
                    rlm_result["operator"],
                    rlm_result["selected_steps"],
                    junction.step,
                    args.model,
                )
                time.sleep(1)
        except Exception as e:
            log.error(f"  RLM failed: {e}")
            result["rlm"]["error"] = str(e)

        # Gold: what was the best outcome in the full run?
        journal = load_journal_from_jsonl(journal_path)
        best_node = max(
            [n for n in journal.nodes if not n.is_buggy and hasattr(n.metric, 'value') and n.metric.value],
            key=lambda n: n.metric.value,
            default=None
        )
        if best_node:
            result["gold"]["best_step"] = best_node.step
            result["gold"]["best_metric"] = best_node.metric.value
            result["gold"]["plan_preview"] = (best_node.plan[:500] + "...") if best_node.plan and len(best_node.plan) > 500 else best_node.plan

        results.append(result)
        time.sleep(2)  # Rate limiting

    # Generate report
    lines = []
    lines.append("# EVO vs RLM Selection Comparison")
    lines.append("")
    lines.append("Comparing operator choices and resulting plans at key decision points.")
    lines.append("")

    for r in results:
        j = r["junction"]
        lines.append(f"## {j['task']} - Step {j['step']}")
        lines.append("")

        # Summary table
        lines.append("| Aspect | EVO (Original) | RLM | Gold (Best in Run) |")
        lines.append("|--------|----------------|-----|-------------------|")

        evo_sel = f"{r['evo']['operator']} {r['evo']['selected']}"
        rlm_sel = f"{r['rlm'].get('operator', 'error')} {r['rlm'].get('selected', [])}" if not r['rlm'].get('error') else "Error"
        gold_info = f"Step {r['gold'].get('best_step', '?')} ({r['gold'].get('best_metric', 0):.4f})"

        lines.append(f"| Selection | {evo_sel} | {rlm_sel} | {gold_info} |")

        evo_metric = f"{r['evo']['child_metric']:.4f}" if r['evo']['child_metric'] else "N/A"
        lines.append(f"| Result Metric | {evo_metric} | (not executed) | {r['gold'].get('best_metric', 0):.4f} |")
        lines.append("")

        # EVO Plan
        lines.append("### EVO's Plan")
        lines.append(f"**Operator**: {r['evo']['operator']} on {r['evo']['selected']}")
        lines.append("")
        lines.append(r['evo'].get('plan', 'No plan'))
        lines.append("")

        # RLM Plan
        lines.append("### RLM's Plan")
        if r['rlm'].get('error'):
            lines.append(f"**Error**: {r['rlm']['error']}")
        else:
            lines.append(f"**Operator**: {r['rlm']['operator']} on {r['rlm']['selected']}")
            lines.append(f"**Reasoning**: {r['rlm'].get('reasoning', 'N/A')}")
            lines.append("")
            lines.append(r['rlm'].get('plan', 'No plan'))
        lines.append("")

        # Gold reference
        lines.append("### Gold (Best Solution in Run)")
        lines.append(f"**Step {r['gold'].get('best_step', '?')}** with metric **{r['gold'].get('best_metric', 0):.4f}**")
        lines.append("")
        lines.append(f"> {r['gold'].get('plan_preview', 'No plan')[:300]}...")
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)

    log.info(f"Saved comparison to {args.output}")
    print(report)


if __name__ == "__main__":
    main()
