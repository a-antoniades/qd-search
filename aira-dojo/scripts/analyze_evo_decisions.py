#!/usr/bin/env python3
"""
Analyze EVO selection decisions vs gold outcomes from journal data.

No LLM calls needed - just extracts and compares actual plans from journals.

Usage:
    python -m scripts.analyze_evo_decisions \
        --logs_dir logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm \
        --output experiments/rlm_replay_results/evo_analysis.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dojo.core.solvers.selection.replay import load_journal_from_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def get_task_name(journal_path: Path) -> str:
    config_path = journal_path.parent.parent / "dojo_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f).get("task", {}).get("name", "unknown")
    return "unknown"


def extract_architecture_keywords(text: str) -> List[str]:
    """Extract ML architecture keywords from plan text."""
    if not text:
        return []
    text_lower = text.lower()
    keywords = {
        "transformer": ["transformer", "bert", "distilbert", "roberta", "deberta", "attention"],
        "cnn": ["convnext", "resnet", "efficientnet", "cnn", "convolutional"],
        "rnn": ["lstm", "gru", "rnn", "recurrent"],
        "lightgbm": ["lightgbm", "lgbm"],
        "xgboost": ["xgboost", "xgb"],
        "catboost": ["catboost"],
        "random_forest": ["random forest", "randomforest"],
        "ensemble": ["ensemble", "stacking", "blending"],
        "gnn": ["gnn", "graph neural", "gatv2", "graph attention"],
        "tfidf": ["tfidf", "tf-idf"],
        "ridge": ["ridge"],
    }
    found = []
    for arch, kws in keywords.items():
        if any(kw in text_lower for kw in kws):
            found.append(arch)
    return found


def analyze_journal(journal_path: Path) -> Dict[str, Any]:
    """Analyze a single journal to understand selection patterns."""
    journal = load_journal_from_jsonl(journal_path)
    task = get_task_name(journal_path)

    nodes_by_step = {n.step: n for n in journal.nodes}

    # Find gold (best) solution
    valid_nodes = [n for n in journal.nodes
                   if not n.is_buggy
                   and hasattr(n.metric, 'value')
                   and n.metric.value is not None]

    if not valid_nodes:
        return {"task": task, "error": "No valid nodes"}

    gold = max(valid_nodes, key=lambda n: n.metric.value)

    # Analyze gold's lineage
    gold_lineage = []
    current = gold
    while current:
        gold_lineage.append(current.step)
        if current.parents:
            current = current.parents[0]
        else:
            break
    gold_lineage.reverse()

    # Find decision points where a different choice could have reached gold faster
    missed_opportunities = []

    for node in journal.nodes:
        if node.step == 0 or not node.parents:
            continue

        parent_steps = [p.step for p in node.parents if p]

        # Check if this node is NOT on the gold path
        if node.step not in gold_lineage:
            # Was there a better option available?
            available_at_time = [n for n in valid_nodes if n.step < node.step]
            if available_at_time:
                best_available = max(available_at_time, key=lambda n: n.metric.value)

                # If we chose something not on gold path but best_available was on gold path
                if best_available.step in gold_lineage and best_available.step not in parent_steps:
                    node_metric = node.metric.value if hasattr(node.metric, 'value') and node.metric.value else None
                    missed_opportunities.append({
                        "step": node.step,
                        "chose_parents": parent_steps,
                        "missed_node": best_available.step,
                        "missed_metric": best_available.metric.value,
                        "actual_metric": node_metric,
                        "chose_archs": extract_architecture_keywords(node.plan or ""),
                        "missed_archs": extract_architecture_keywords(best_available.plan or ""),
                    })

    # Architecture diversity analysis
    arch_counts = {}
    for node in valid_nodes:
        archs = extract_architecture_keywords(node.plan or "")
        for arch in archs:
            arch_counts[arch] = arch_counts.get(arch, 0) + 1

    return {
        "task": task,
        "journal_path": str(journal_path),
        "total_nodes": len(journal.nodes),
        "valid_nodes": len(valid_nodes),
        "gold": {
            "step": gold.step,
            "metric": gold.metric.value,
            "plan_preview": (gold.plan[:500] + "...") if gold.plan and len(gold.plan) > 500 else gold.plan,
            "architectures": extract_architecture_keywords(gold.plan or ""),
            "lineage_length": len(gold_lineage),
            "lineage": gold_lineage[:10],  # First 10 steps
        },
        "missed_opportunities": missed_opportunities[:5],  # Top 5
        "architecture_diversity": arch_counts,
    }


def generate_report(analyses: List[Dict], output_path: Path):
    """Generate markdown report."""
    lines = []
    lines.append("# EVO Selection Decisions Analysis")
    lines.append("")
    lines.append("Analysis of what EVO chose vs what led to the best solutions.")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Task | Gold Step | Gold Metric | Gold Architectures | Missed Opportunities |")
    lines.append("|------|-----------|-------------|-------------------|---------------------|")

    for a in analyses:
        if "error" in a:
            lines.append(f"| {a['task']} | Error | - | - | - |")
            continue
        gold = a["gold"]
        archs = ", ".join(gold["architectures"][:3]) or "N/A"
        lines.append(f"| {a['task'][:25]} | {gold['step']} | {gold['metric']:.4f} | {archs} | {len(a['missed_opportunities'])} |")

    lines.append("")

    # Detailed analysis per task
    for a in analyses:
        if "error" in a:
            continue

        lines.append(f"## {a['task']}")
        lines.append("")

        gold = a["gold"]
        lines.append(f"**Gold Solution**: Step {gold['step']} with metric **{gold['metric']:.4f}**")
        lines.append(f"- Architectures: {', '.join(gold['architectures']) or 'N/A'}")
        lines.append(f"- Lineage: {' → '.join(map(str, gold['lineage']))}")
        lines.append("")

        lines.append("**Plan Preview**:")
        lines.append(f"> {gold['plan_preview'][:400]}...")
        lines.append("")

        # Architecture diversity
        lines.append("**Architecture Diversity** (how many valid nodes used each):")
        arch_div = a["architecture_diversity"]
        if arch_div:
            sorted_archs = sorted(arch_div.items(), key=lambda x: -x[1])
            for arch, count in sorted_archs[:5]:
                lines.append(f"- {arch}: {count} nodes")
        lines.append("")

        # Missed opportunities
        if a["missed_opportunities"]:
            lines.append("**Key Missed Opportunities** (chose different node when gold-path node was available):")
            lines.append("")
            for opp in a["missed_opportunities"][:3]:
                lines.append(f"- **Step {opp['step']}**: Chose {opp['chose_parents']} ({', '.join(opp['chose_archs']) or 'N/A'})")
                lines.append(f"  - Could have selected: Step {opp['missed_node']} ({opp['missed_metric']:.4f}, {', '.join(opp['missed_archs']) or 'N/A'})")
                if opp['actual_metric']:
                    lines.append(f"  - Actual result: {opp['actual_metric']:.4f}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Overall insights
    lines.append("## Key Insights")
    lines.append("")

    # Count architecture preferences
    all_gold_archs = []
    all_archs = {}
    for a in analyses:
        if "error" not in a:
            all_gold_archs.extend(a["gold"]["architectures"])
            for arch, count in a["architecture_diversity"].items():
                all_archs[arch] = all_archs.get(arch, 0) + count

    lines.append("**Gold Solution Architectures**:")
    from collections import Counter
    gold_counts = Counter(all_gold_archs)
    for arch, count in gold_counts.most_common(5):
        lines.append(f"- {arch}: {count} gold solutions")
    lines.append("")

    lines.append("**Most Explored Architectures**:")
    sorted_all = sorted(all_archs.items(), key=lambda x: -x[1])
    for arch, count in sorted_all[:5]:
        lines.append(f"- {arch}: {count} total valid nodes")
    lines.append("")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    journals = list(args.logs_dir.rglob("checkpoint/journal.jsonl"))
    log.info(f"Found {len(journals)} journals")

    # Analyze each, keeping one per task (most nodes)
    task_analyses = {}
    for jp in journals:
        analysis = analyze_journal(jp)
        task = analysis.get("task", "unknown")
        if task not in task_analyses or analysis.get("total_nodes", 0) > task_analyses[task].get("total_nodes", 0):
            task_analyses[task] = analysis

    analyses = list(task_analyses.values())
    log.info(f"Analyzed {len(analyses)} unique tasks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(analyses, args.output)
    log.info(f"Saved to {args.output}")
    print(report)


if __name__ == "__main__":
    main()
