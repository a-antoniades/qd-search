#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate crossover plans for RLM-selected node pairs.

Takes successful replay results and generates the crossover plan/idea
that would have been produced, without actually executing the code.

Usage:
    python -m scripts.generate_crossover_plans \
        --replays experiments/rlm_replay_results/replays.json \
        --output experiments/rlm_replay_results/crossover_plans.md
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dojo.core.solvers.selection.replay import load_journal_from_jsonl, snapshot_at_step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# Crossover prompt template (simplified from the full AIRA template)
CROSSOVER_PROMPT = """You are a Kaggle Grandmaster. Your goal is to combine two previously developed solutions to increase performance.

# TASK DESCRIPTION
{task_desc}

# SOLUTION 1 (Step {step1}, Metric: {metric1})
## Plan:
{plan1}

## Code (first 3000 chars):
```python
{code1}
```

# SOLUTION 2 (Step {step2}, Metric: {metric2})
## Plan:
{plan2}

## Code (first 3000 chars):
```python
{code2}
```

# INSTRUCTIONS
Propose a **Crossover Plan** explaining:
1. What are the key strengths of each solution?
2. How can they be effectively combined?
3. Why would this combination likely improve performance?

Focus on the IDEAS and APPROACH, not implementation details.

# Crossover Plan
"""


def get_task_description(journal_path: Path) -> str:
    """Extract task description from dojo_config.json."""
    config_path = journal_path.parent.parent / "dojo_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        task_name = config.get("task", {}).get("name", "Unknown task")
        # Try to get more details
        return f"Kaggle competition: {task_name}"
    return "ML competition task"


def generate_crossover_plan(
    journal_path: Path,
    step: int,
    node1_step: int,
    node2_step: int,
    model: str = "gemini-3-flash-preview",
) -> Dict[str, Any]:
    """
    Generate crossover plan for two nodes using LLM.

    Returns dict with plan text and metadata.
    """
    log.info(f"Loading journal from {journal_path}")
    journal = load_journal_from_jsonl(journal_path)

    # Get the snapshot at the junction step
    snapshot = snapshot_at_step(journal, step)

    # Find the two nodes
    nodes_by_step = {n.step: n for n in snapshot.journal.nodes}

    node1 = nodes_by_step.get(node1_step)
    node2 = nodes_by_step.get(node2_step)

    if not node1 or not node2:
        return {
            "error": f"Could not find nodes {node1_step} and/or {node2_step}",
            "plan": None,
        }

    # Get task description
    task_desc = get_task_description(journal_path)

    # Get metrics
    metric1 = node1.metric.value if hasattr(node1.metric, 'value') else None
    metric2 = node2.metric.value if hasattr(node2.metric, 'value') else None

    # Build prompt
    prompt = CROSSOVER_PROMPT.format(
        task_desc=task_desc,
        step1=node1_step,
        metric1=f"{metric1:.4f}" if metric1 else "N/A",
        plan1=node1.plan[:2000] if node1.plan else "(No plan)",
        code1=node1.code[:3000] if node1.code else "(No code)",
        step2=node2_step,
        metric2=f"{metric2:.4f}" if metric2 else "N/A",
        plan2=node2.plan[:2000] if node2.plan else "(No plan)",
        code2=node2.code[:3000] if node2.code else "(No code)",
    )

    # Call LLM
    try:
        import litellm

        log.info(f"Calling {model} to generate crossover plan...")
        start_time = time.time()

        response = litellm.completion(
            model=f"gemini/{model}",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7,
        )

        elapsed = time.time() - start_time
        plan_text = response.choices[0].message.content

        return {
            "error": None,
            "plan": plan_text,
            "node1": {
                "step": node1_step,
                "metric": metric1,
                "plan_preview": (node1.plan[:500] + "...") if node1.plan and len(node1.plan) > 500 else node1.plan,
            },
            "node2": {
                "step": node2_step,
                "metric": metric2,
                "plan_preview": (node2.plan[:500] + "...") if node2.plan and len(node2.plan) > 500 else node2.plan,
            },
            "elapsed_s": elapsed,
            "model": model,
        }

    except Exception as e:
        log.error(f"Failed to generate crossover plan: {e}")
        return {
            "error": str(e),
            "plan": None,
        }


def generate_report(results: List[Dict], output_path: Path) -> str:
    """Generate markdown report with crossover plans."""
    lines = []

    lines.append("# RLM Crossover Plans Analysis")
    lines.append("")
    lines.append("This report shows the crossover plans that RLM's selected node pairs would have generated.")
    lines.append("")

    for r in results:
        lines.append(f"## {r['task']} - Step {r['step']}")
        lines.append("")
        lines.append(f"**Junction type**: {r['junction_type']}")
        lines.append(f"**RLM selected**: crossover [{r['node1_step']}, {r['node2_step']}]")
        lines.append(f"**Original choice**: {r['original_operator']} {r['original_steps']}")
        lines.append("")

        if r.get("error"):
            lines.append(f"**Error**: {r['error']}")
            lines.append("")
            continue

        plan_result = r.get("plan_result", {})

        if plan_result.get("error"):
            lines.append(f"**Error generating plan**: {plan_result['error']}")
            lines.append("")
            continue

        # Show the two nodes being combined
        lines.append("### Input Nodes")
        lines.append("")

        node1 = plan_result.get("node1", {})
        node2 = plan_result.get("node2", {})

        lines.append(f"**Node {node1.get('step')}** (metric: {node1.get('metric', 'N/A')})")
        lines.append(f"> {node1.get('plan_preview', 'No plan')[:300]}...")
        lines.append("")

        lines.append(f"**Node {node2.get('step')}** (metric: {node2.get('metric', 'N/A')})")
        lines.append(f"> {node2.get('plan_preview', 'No plan')[:300]}...")
        lines.append("")

        # Show the generated crossover plan
        lines.append("### Generated Crossover Plan")
        lines.append("")
        lines.append(plan_result.get("plan", "No plan generated"))
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate crossover plans for RLM-selected node pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--replays",
        type=Path,
        required=True,
        help="Path to replay results JSON",
    )
    parser.add_argument(
        "--junctions",
        type=Path,
        help="Path to junctions JSON (for journal paths)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for crossover plans markdown",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Model to use for generating plans",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making LLM calls",
    )

    args = parser.parse_args()

    if not args.replays.exists():
        log.error(f"Replays file not found: {args.replays}")
        sys.exit(1)

    # Load replays
    with open(args.replays) as f:
        data = json.load(f)

    replays = data.get("replays", [])

    # Load junctions for journal paths
    junctions_path = args.junctions or args.replays.parent / "junctions.json"
    journal_paths = {}
    if junctions_path.exists():
        with open(junctions_path) as f:
            junctions_data = json.load(f)
        for j in junctions_data.get("junctions", []):
            task = j.get("task", "")
            journal_paths[task] = j.get("journal_path", "")

    # Filter to successful crossover replays
    crossover_replays = [
        r for r in replays
        if not r.get("error")
        and r.get("rlm", {}).get("operator") == "crossover"
        and len(r.get("rlm", {}).get("selected_steps", [])) == 2
    ]

    log.info(f"Found {len(crossover_replays)} successful crossover replays")

    if not crossover_replays:
        log.error("No successful crossover replays found")
        sys.exit(1)

    results = []

    for replay in crossover_replays:
        task = replay.get("task", "Unknown")
        step = replay.get("step", 0)
        rlm_steps = replay.get("rlm", {}).get("selected_steps", [])

        log.info(f"Processing {task} step {step}: crossover {rlm_steps}")

        result = {
            "task": task,
            "step": step,
            "junction_type": replay.get("junction_type", ""),
            "node1_step": rlm_steps[0],
            "node2_step": rlm_steps[1],
            "original_operator": replay.get("original", {}).get("operator", ""),
            "original_steps": replay.get("original", {}).get("selected_steps", []),
        }

        journal_path = journal_paths.get(task)
        if not journal_path:
            result["error"] = "Could not find journal path"
            results.append(result)
            continue

        # Convert to absolute path
        journal_path = Path(journal_path)
        if not journal_path.is_absolute():
            journal_path = Path("/share/edc/home/antonis/qd-search/aira-dojo") / journal_path

        if not journal_path.exists():
            result["error"] = f"Journal not found: {journal_path}"
            results.append(result)
            continue

        if args.dry_run:
            result["plan_result"] = {"plan": "[DRY RUN - no LLM call made]"}
            results.append(result)
            continue

        # Generate crossover plan
        plan_result = generate_crossover_plan(
            journal_path=journal_path,
            step=step,
            node1_step=rlm_steps[0],
            node2_step=rlm_steps[1],
            model=args.model,
        )

        result["plan_result"] = plan_result
        results.append(result)

        # Small delay between calls
        time.sleep(1)

    # Generate report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, args.output)

    log.info(f"Saved crossover plans to {args.output}")
    print(report)


if __name__ == "__main__":
    main()
