#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Analyze RLM replay results and generate comparison report.

Takes replay results from run_rlm_replay.py and generates a markdown
report comparing RLM selections vs original fitness-based selections.

Usage:
    python -m scripts.analyze_replay_results \
        --replays experiments/rlm_replay_results/replays.json \
        --output experiments/rlm_replay_results/analysis.md
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dojo.core.solvers.selection.replay import load_journal_from_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def assess_reasoning_quality(reasoning: str, junction_type: str, description: str, raw_response: str = "") -> str:
    """
    Assess the quality of RLM reasoning.

    Returns: "Good", "Partial", or "Poor"
    """
    if not reasoning and not raw_response:
        return "Poor"

    # Combine reasoning and raw_response for analysis
    full_text = (reasoning + " " + raw_response).lower()

    # Check for key reasoning indicators based on junction type
    quality_indicators = {
        "best_abandonment": ["best", "score", "metric", "abandon", "select", "leading", "top", "improve", "crossover", "combine"],
        "debug_chain": ["debug", "error", "bug", "fix", "stuck", "pattern", "different", "approach"],
        "score_plateau": ["plateau", "stuck", "improvement", "explore", "new", "fresh", "crossover", "combine", "diverse", "stagnating"],
        "crossover_candidate": ["combine", "crossover", "merge", "diverse", "complement", "synergy", "different", "approach"],
        "repeated_failure": ["error", "pattern", "same", "different", "approach", "stuck", "draft", "new"],
    }

    indicators = quality_indicators.get(junction_type, [])
    matches = sum(1 for ind in indicators if ind in full_text)

    if matches >= 3:
        return "Good"
    elif matches >= 1:
        return "Partial"
    else:
        return "Poor"


def assess_outcome_potential(replay: Dict, journal_path: str) -> str:
    """
    Assess whether the RLM choice would have helped.

    Checks if the RLM-selected node(s) had better descendants than
    what was actually selected.

    Returns: "Likely", "Maybe", "Unlikely", or "Unknown"
    """
    if replay.get("error"):
        return "Unknown"

    rlm_steps = replay.get("rlm", {}).get("selected_steps", [])
    original_steps = replay.get("original", {}).get("selected_steps", [])

    if not rlm_steps or set(rlm_steps) == set(original_steps):
        return "Unknown"

    try:
        journal = load_journal_from_jsonl(Path(journal_path))
        nodes_by_step = {n.step: n for n in journal.nodes}

        # Get best descendant metric for each selected node
        def get_best_descendant_metric(step: int, lower_is_better: bool = False) -> Optional[float]:
            """Recursively find best metric among descendants."""
            node = nodes_by_step.get(step)
            if not node:
                return None

            best = None
            if hasattr(node.metric, 'value') and node.metric.value is not None and not node.is_buggy:
                best = node.metric.value

            for child in (node.children or []):
                child_best = get_best_descendant_metric(child.step, lower_is_better)
                if child_best is not None:
                    if best is None:
                        best = child_best
                    elif lower_is_better:
                        best = min(best, child_best)
                    else:
                        best = max(best, child_best)

            return best

        # Compare best descendants
        rlm_best = None
        for step in rlm_steps:
            b = get_best_descendant_metric(step)
            if b is not None:
                if rlm_best is None:
                    rlm_best = b
                else:
                    rlm_best = max(rlm_best, b)

        original_best = None
        for step in original_steps:
            b = get_best_descendant_metric(step)
            if b is not None:
                if original_best is None:
                    original_best = b
                else:
                    original_best = max(original_best, b)

        if rlm_best is None or original_best is None:
            return "Unknown"

        # Compare
        if rlm_best > original_best * 1.01:  # RLM choice led to better outcome
            return "Likely"
        elif rlm_best > original_best * 0.99:  # Similar
            return "Maybe"
        else:
            return "Unlikely"

    except Exception as e:
        log.warning(f"Could not assess outcome: {e}")
        return "Unknown"


def generate_markdown_table(replays: List[Dict], journal_paths: Dict[str, str]) -> str:
    """Generate markdown table comparing RLM vs original selections."""
    rows = []

    for r in replays:
        task = r.get("task", "Unknown")
        step = r.get("step", 0)
        junction_type = r.get("junction_type", "Unknown")

        if r.get("error"):
            rows.append({
                "task": task,
                "junction_type": junction_type,
                "original": "Error",
                "rlm": "Error",
                "different": "-",
                "reasoning_quality": "-",
                "would_have_helped": "-",
            })
            continue

        original = r.get("original", {})
        rlm = r.get("rlm", {})

        original_str = f"{original.get('operator', 'N/A')} {original.get('selected_steps', [])}"
        rlm_str = f"{rlm.get('operator', 'N/A')} {rlm.get('selected_steps', [])}"

        reasoning = rlm.get("reasoning", "")
        raw_response = rlm.get("raw_response", "")
        reasoning_quality = assess_reasoning_quality(reasoning, junction_type, r.get("description", ""), raw_response)

        # Get journal path for this task
        journal_path = journal_paths.get(task)
        if journal_path:
            would_have_helped = assess_outcome_potential(r, journal_path)
        else:
            would_have_helped = "Unknown"

        rows.append({
            "task": task,
            "junction_type": junction_type,
            "original": original_str,
            "rlm": rlm_str,
            "different": "Yes" if r.get("different") else "No",
            "reasoning_quality": reasoning_quality,
            "would_have_helped": would_have_helped,
            "reasoning_excerpt": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
        })

    # Generate markdown
    lines = []
    lines.append("| Task | Type | Original Selection | RLM Selection | Different? | Reasoning Quality | Would Have Helped? |")
    lines.append("|------|------|-------------------|---------------|------------|-------------------|-------------------|")

    for row in rows:
        lines.append(
            f"| {row['task'][:20]} | {row['junction_type']} | {row['original']} | "
            f"{row['rlm']} | {row['different']} | {row['reasoning_quality']} | "
            f"{row['would_have_helped']} |"
        )

    return "\n".join(lines)


def generate_report(
    replays: List[Dict],
    config: Dict,
    summary: Dict,
    junctions_path: Path,
) -> str:
    """Generate full markdown analysis report."""
    lines = []

    # Header
    lines.append("# RLM Selection Replay Analysis")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report compares RLM-based node selection against the original fitness-based selection")
    lines.append("at key decision points (junctions) in EVO search runs.")
    lines.append("")

    # Summary stats
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- **Model**: {config.get('model', 'N/A')}")
    lines.append(f"- **Backend**: {config.get('backend', 'N/A')}")
    lines.append(f"- **Total replays**: {summary.get('total_replays', 0)}")
    lines.append(f"- **Successful**: {summary.get('successful', 0)}")
    lines.append(f"- **Different choices**: {summary.get('different_choices', 0)}")
    lines.append(f"- **Errors**: {summary.get('errors', 0)}")
    lines.append("")

    # Success rate
    if summary.get('successful', 0) > 0:
        diff_rate = summary.get('different_choices', 0) / summary.get('successful', 1) * 100
        lines.append(f"RLM chose differently in **{diff_rate:.1f}%** of successful replays.")
    lines.append("")

    # Build journal_paths mapping
    journal_paths = {}
    for r in replays:
        task = r.get("task", "")
        # Try to extract journal path from junctions file
        try:
            with open(junctions_path) as f:
                junctions_data = json.load(f)
            for j in junctions_data.get("junctions", []):
                if j.get("task") == task:
                    journal_paths[task] = j.get("journal_path", "")
                    break
        except Exception:
            pass

    # Comparison table
    lines.append("## Detailed Comparison")
    lines.append("")
    lines.append(generate_markdown_table(replays, journal_paths))
    lines.append("")

    # Reasoning excerpts
    lines.append("## RLM Reasoning Excerpts")
    lines.append("")

    for r in replays:
        if r.get("error"):
            continue

        task = r.get("task", "Unknown")
        step = r.get("step", 0)
        reasoning = r.get("rlm", {}).get("reasoning", "")

        lines.append(f"### {task} (Step {step})")
        lines.append("")
        lines.append(f"**Junction type**: {r.get('junction_type', 'Unknown')}")
        lines.append("")
        lines.append(f"**Description**: {r.get('description', 'N/A')}")
        lines.append("")
        lines.append(f"**Original**: {r.get('original', {}).get('operator', 'N/A')} "
                     f"{r.get('original', {}).get('selected_steps', [])}")
        lines.append(f"**RLM**: {r.get('rlm', {}).get('operator', 'N/A')} "
                     f"{r.get('rlm', {}).get('selected_steps', [])}")
        lines.append("")
        lines.append("**RLM Reasoning**:")
        lines.append("")
        lines.append(f"> {reasoning}")
        lines.append("")

    # Conclusions
    lines.append("## Conclusions")
    lines.append("")

    different_count = sum(1 for r in replays if r.get("different") and not r.get("error"))
    total_successful = sum(1 for r in replays if not r.get("error"))

    if total_successful == 0:
        lines.append("No successful replays to analyze.")
    else:
        lines.append(f"1. **Selection divergence**: RLM chose differently in {different_count}/{total_successful} cases.")
        lines.append("")

        # Analyze reasoning quality
        good_reasoning = sum(1 for r in replays if not r.get("error") and
                             assess_reasoning_quality(r.get("rlm", {}).get("reasoning", ""),
                                                      r.get("junction_type", ""),
                                                      r.get("description", ""),
                                                      r.get("rlm", {}).get("raw_response", "")) == "Good")
        lines.append(f"2. **Reasoning quality**: {good_reasoning}/{total_successful} replays had good reasoning quality.")
        lines.append("")

        if different_count >= 3:
            lines.append("3. **Success criteria met**: At least 3 junctions where RLM chose differently.")
        else:
            lines.append(f"3. **Success criteria not met**: Only {different_count}/3 required junctions with different choices.")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by analyze_replay_results.py*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RLM replay results and generate comparison report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--replays",
        type=Path,
        required=True,
        help="Path to replay results JSON from run_rlm_replay.py",
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
        help="Output path for analysis markdown",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    args = parser.parse_args()

    if not args.replays.exists():
        log.error(f"Replays file not found: {args.replays}")
        sys.exit(1)

    # Load replay results
    with open(args.replays) as f:
        data = json.load(f)

    config = data.get("config", {})
    summary = data.get("summary", {})
    replays = data.get("replays", [])

    log.info(f"Loaded {len(replays)} replay results")

    # Determine junctions path
    junctions_path = args.junctions
    if not junctions_path:
        # Try to find it in same directory
        junctions_path = args.replays.parent / "junctions.json"

    # Generate report
    if args.format == "markdown":
        report = generate_report(replays, config, summary, junctions_path)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)

        log.info(f"Saved analysis to {args.output}")

        # Also print to stdout
        print(report)

    else:  # JSON
        analysis = {
            "config": config,
            "summary": summary,
            "detailed_analysis": [],
        }

        # Build journal_paths mapping
        journal_paths = {}
        if junctions_path and junctions_path.exists():
            try:
                with open(junctions_path) as f:
                    junctions_data = json.load(f)
                for j in junctions_data.get("junctions", []):
                    task = j.get("task", "")
                    journal_paths[task] = j.get("journal_path", "")
            except Exception:
                pass

        for r in replays:
            task = r.get("task", "Unknown")
            reasoning = r.get("rlm", {}).get("reasoning", "")
            junction_type = r.get("junction_type", "")

            journal_path = journal_paths.get(task)
            would_have_helped = assess_outcome_potential(r, journal_path) if journal_path else "Unknown"

            analysis["detailed_analysis"].append({
                "task": task,
                "step": r.get("step"),
                "junction_type": junction_type,
                "different": r.get("different"),
                "reasoning_quality": assess_reasoning_quality(reasoning, junction_type, r.get("description", "")),
                "would_have_helped": would_have_helped,
                "error": r.get("error"),
            })

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)

        log.info(f"Saved analysis to {args.output}")


if __name__ == "__main__":
    main()
