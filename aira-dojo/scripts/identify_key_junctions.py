#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Identify key decision junctions in EVO search runs.

Scans JOURNAL.jsonl files from EVO runs and identifies candidate junctions
where different selection choices could have led to better outcomes.

Junction types:
- best_abandonment: Step where selected parent != best scoring node
- debug_chain: 3+ consecutive debug operations on same parent
- score_plateau: No improvement in last N steps
- crossover_candidate: 2+ leaves with different architectures
- repeated_failure: Same error pattern in 3+ consecutive buggy nodes

Usage:
    python -m scripts.identify_key_junctions \
        --logs_dir logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm \
        --output experiments/rlm_replay_results/junctions.json
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dojo.core.solvers.selection.replay import load_journal_from_jsonl
from dojo.core.solvers.utils.journal import Journal, Node

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# Architecture keywords for diversity analysis
ARCHITECTURE_KEYWORDS = {
    # Deep learning
    "transformer": ["transformer", "bert", "gpt", "attention", "distilbert", "roberta", "deberta", "albert"],
    "cnn": ["convnext", "resnet", "efficientnet", "vit", "cnn", "convolutional", "mobilenet", "densenet"],
    "rnn": ["lstm", "gru", "rnn", "recurrent", "bigru", "bilstm"],
    "mlp": ["mlp", "feedforward", "dense layer", "neural network"],
    # Tree-based
    "lightgbm": ["lightgbm", "lgbm"],
    "xgboost": ["xgboost", "xgb"],
    "catboost": ["catboost"],
    "random_forest": ["random forest", "randomforest"],
    "extratrees": ["extratrees", "extra trees"],
    "gradient_boosting": ["gradient boosting", "gbdt"],
    # Other
    "linear": ["linear", "logistic", "ridge", "lasso", "elasticnet"],
    "svm": ["svm", "support vector"],
    "knn": ["knn", "k-nearest", "nearest neighbor"],
    "ensemble": ["ensemble", "stacking", "blending", "voting"],
    "gnn": ["gnn", "graph neural", "graph network"],
}


@dataclass
class Junction:
    """A key decision point in the search."""
    task: str
    step: int
    junction_type: str
    description: str
    context: Dict[str, Any]
    journal_path: str
    lower_is_better: bool = False
    priority: float = 0.0  # Higher = more important


@dataclass
class JunctionResult:
    """Result of junction identification for a task."""
    task: str
    journal_path: str
    total_nodes: int
    best_metric: Optional[float]
    junctions: List[Junction]


def extract_architecture(text: str) -> List[str]:
    """Extract architecture keywords from plan/code text."""
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for arch, keywords in ARCHITECTURE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(arch)
                break
    return list(set(found))


def get_node_architecture(node: Node) -> List[str]:
    """Get architecture keywords for a node."""
    text = (node.plan or "") + " " + (node.code or "")[:5000]
    return extract_architecture(text)


def find_best_abandonment(journal: Journal, lower_is_better: bool) -> List[Tuple[int, Dict]]:
    """
    Find steps where the selected parent was not the best scoring node.

    Returns list of (step, context) tuples.
    """
    results = []
    nodes_by_step = {n.step: n for n in journal.nodes}

    for node in journal.nodes:
        if node.step == 0 or not node.parents:
            continue

        # Get valid nodes at time of selection (before this node was created)
        valid_nodes = [n for n in journal.nodes
                       if n.step < node.step
                       and not n.is_buggy
                       and n.metric is not None
                       and hasattr(n.metric, 'value')
                       and n.metric.value is not None]

        if not valid_nodes:
            continue

        # Find best node at that time
        if lower_is_better:
            best_node = min(valid_nodes, key=lambda n: n.metric.value)
        else:
            best_node = max(valid_nodes, key=lambda n: n.metric.value)

        # Check if any parent was NOT the best node
        parent_steps = [p.step for p in node.parents if p]
        if best_node.step not in parent_steps:
            parent_metrics = []
            for p in node.parents:
                if p and hasattr(p.metric, 'value') and p.metric.value is not None:
                    parent_metrics.append(p.metric.value)

            context = {
                "best_node_step": best_node.step,
                "best_node_metric": best_node.metric.value,
                "selected_parent_steps": parent_steps,
                "selected_parent_metrics": parent_metrics,
                "best_architecture": get_node_architecture(best_node),
                "parent_architectures": [get_node_architecture(p) for p in node.parents if p],
            }
            results.append((node.step, context))

    return results


def find_debug_chains(journal: Journal) -> List[Tuple[int, Dict]]:
    """
    Find sequences of 3+ consecutive debug operations on same parent.

    Returns list of (step, context) tuples where step is the 3rd+ debug.
    """
    results = []

    # Build parent -> children mapping
    for node in journal.nodes:
        if not node.parents or len(node.parents) != 1:
            continue

        parent = node.parents[0]
        if not parent or not parent.is_buggy:
            continue

        # Count debug depth
        depth = 0
        current = node
        while current.parents and len(current.parents) == 1:
            p = current.parents[0]
            if p and p.is_buggy:
                depth += 1
                current = p
            else:
                break

        if depth >= 3:
            context = {
                "debug_depth": depth,
                "parent_step": parent.step,
                "error_snippet": (parent.term_out or "")[:500] if parent.term_out else "",
            }
            results.append((node.step, context))

    return results


def find_score_plateaus(journal: Journal, lookback: int = 5, lower_is_better: bool = False) -> List[Tuple[int, Dict]]:
    """
    Find steps where the score has plateaued for N consecutive steps.

    Returns list of (step, context) tuples.
    """
    results = []

    valid_nodes = sorted(
        [n for n in journal.nodes
         if not n.is_buggy
         and n.metric is not None
         and hasattr(n.metric, 'value')
         and n.metric.value is not None],
        key=lambda n: n.step
    )

    if len(valid_nodes) < lookback + 1:
        return results

    # Track best metric over time
    best_metric = valid_nodes[0].metric.value
    plateau_start = 0

    for i, node in enumerate(valid_nodes):
        current_metric = node.metric.value

        # Check if this is an improvement
        is_better = (current_metric < best_metric) if lower_is_better else (current_metric > best_metric)

        if is_better:
            best_metric = current_metric
            plateau_start = i
        elif i - plateau_start >= lookback:
            # We've been plateaued for at least `lookback` steps
            context = {
                "plateau_length": i - plateau_start,
                "best_metric": best_metric,
                "current_metric": current_metric,
                "plateau_start_step": valid_nodes[plateau_start].step,
            }
            results.append((node.step, context))

    return results


def find_crossover_candidates(journal: Journal, lower_is_better: bool = False) -> List[Tuple[int, Dict]]:
    """
    Find steps with 2+ leaf nodes having different architectures.

    These are opportunities for crossover that may not have been taken.
    Returns list of (step, context) tuples.
    """
    results = []
    nodes_by_step = {n.step: n for n in journal.nodes}

    # Process each step
    for target_step in range(1, len(journal.nodes)):
        # Get leaves at this step
        leaves_at_step = []
        for node in journal.nodes:
            if node.step > target_step:
                continue
            # Check if it's a leaf at this step (no children with step <= target)
            is_leaf = True
            for child in node.children:
                if child.step <= target_step:
                    is_leaf = False
                    break
            if is_leaf and not node.is_buggy and node.metric is not None:
                leaves_at_step.append(node)

        if len(leaves_at_step) < 2:
            continue

        # Get architectures for each leaf
        arch_to_nodes = {}
        for leaf in leaves_at_step:
            archs = get_node_architecture(leaf)
            arch_key = tuple(sorted(archs)) if archs else ("unknown",)
            if arch_key not in arch_to_nodes:
                arch_to_nodes[arch_key] = []
            arch_to_nodes[arch_key].append(leaf)

        # If we have 2+ different architectures, it's a crossover opportunity
        if len(arch_to_nodes) >= 2:
            # Find best pair for crossover
            pairs = []
            arch_keys = list(arch_to_nodes.keys())
            for i, ak1 in enumerate(arch_keys):
                for ak2 in arch_keys[i+1:]:
                    # Get best node from each architecture
                    if lower_is_better:
                        n1 = min(arch_to_nodes[ak1], key=lambda n: n.metric.value)
                        n2 = min(arch_to_nodes[ak2], key=lambda n: n.metric.value)
                    else:
                        n1 = max(arch_to_nodes[ak1], key=lambda n: n.metric.value)
                        n2 = max(arch_to_nodes[ak2], key=lambda n: n.metric.value)
                    pairs.append({
                        "node1": {"step": n1.step, "metric": n1.metric.value, "archs": list(ak1)},
                        "node2": {"step": n2.step, "metric": n2.metric.value, "archs": list(ak2)},
                    })

            context = {
                "num_architectures": len(arch_to_nodes),
                "architecture_counts": {str(k): len(v) for k, v in arch_to_nodes.items()},
                "crossover_pairs": pairs[:5],  # Top 5 pairs
            }
            results.append((target_step, context))

    return results


def find_repeated_failures(journal: Journal) -> List[Tuple[int, Dict]]:
    """
    Find steps with 3+ consecutive buggy nodes with similar error patterns.

    Returns list of (step, context) tuples.
    """
    results = []

    # Group buggy nodes by error type
    def extract_error_type(term_out: str) -> str:
        if not term_out:
            return "unknown"
        term_lower = term_out.lower()

        # Common error patterns
        patterns = [
            (r"importerror|modulenotfounderror|no module named", "import_error"),
            (r"attributeerror", "attribute_error"),
            (r"typeerror", "type_error"),
            (r"valueerror", "value_error"),
            (r"keyerror", "key_error"),
            (r"indexerror", "index_error"),
            (r"filenotfounderror|no such file", "file_error"),
            (r"memoryerror|out of memory|oom", "memory_error"),
            (r"timeout|timed out", "timeout_error"),
            (r"cuda|gpu|device", "cuda_error"),
            (r"cv2|opencv", "cv2_error"),
        ]

        for pattern, error_type in patterns:
            if re.search(pattern, term_lower):
                return error_type
        return "other"

    # Find consecutive buggy sequences
    buggy_nodes = [n for n in journal.nodes if n.is_buggy and n.step > 0]

    i = 0
    while i < len(buggy_nodes):
        current_error = extract_error_type(buggy_nodes[i].term_out)
        sequence = [buggy_nodes[i]]

        j = i + 1
        while j < len(buggy_nodes):
            if extract_error_type(buggy_nodes[j].term_out) == current_error:
                sequence.append(buggy_nodes[j])
                j += 1
            else:
                break

        if len(sequence) >= 3:
            context = {
                "error_type": current_error,
                "sequence_length": len(sequence),
                "first_step": sequence[0].step,
                "last_step": sequence[-1].step,
                "error_snippet": (sequence[0].term_out or "")[:500],
            }
            results.append((sequence[-1].step, context))

        i = j

    return results


def identify_junctions_for_task(
    journal_path: Path,
    task_name: str,
    lower_is_better: bool = False,
) -> JunctionResult:
    """Identify all key junctions for a single task."""
    log.info(f"Processing {task_name} from {journal_path}")

    journal = load_journal_from_jsonl(journal_path)
    log.info(f"Loaded {len(journal.nodes)} nodes")

    # Get best metric
    valid_nodes = [n for n in journal.nodes
                   if not n.is_buggy
                   and n.metric is not None
                   and hasattr(n.metric, 'value')
                   and n.metric.value is not None]

    if valid_nodes:
        if lower_is_better:
            best_node = min(valid_nodes, key=lambda n: n.metric.value)
        else:
            best_node = max(valid_nodes, key=lambda n: n.metric.value)
        best_metric = best_node.metric.value
    else:
        best_metric = None

    junctions = []

    # Find all junction types
    for step, ctx in find_best_abandonment(journal, lower_is_better):
        j = Junction(
            task=task_name,
            step=step,
            junction_type="best_abandonment",
            description=f"Selected parent {ctx['selected_parent_steps']} instead of best node {ctx['best_node_step']} (metric {ctx['best_node_metric']:.4f})",
            context=ctx,
            journal_path=str(journal_path),
            lower_is_better=lower_is_better,
            priority=abs(ctx['best_node_metric'] - ctx['selected_parent_metrics'][0]) if ctx['selected_parent_metrics'] else 0,
        )
        junctions.append(j)

    for step, ctx in find_debug_chains(journal):
        j = Junction(
            task=task_name,
            step=step,
            junction_type="debug_chain",
            description=f"Debug chain of depth {ctx['debug_depth']} - same bug being debugged repeatedly",
            context=ctx,
            journal_path=str(journal_path),
            lower_is_better=lower_is_better,
            priority=ctx['debug_depth'],
        )
        junctions.append(j)

    for step, ctx in find_score_plateaus(journal, lookback=5, lower_is_better=lower_is_better):
        j = Junction(
            task=task_name,
            step=step,
            junction_type="score_plateau",
            description=f"Score plateaued for {ctx['plateau_length']} steps at {ctx['best_metric']:.4f}",
            context=ctx,
            journal_path=str(journal_path),
            lower_is_better=lower_is_better,
            priority=ctx['plateau_length'],
        )
        junctions.append(j)

    crossover_results = find_crossover_candidates(journal, lower_is_better)
    # Only keep crossover opportunities where we have good candidates
    for step, ctx in crossover_results:
        if ctx['num_architectures'] >= 2 and ctx['crossover_pairs']:
            j = Junction(
                task=task_name,
                step=step,
                junction_type="crossover_candidate",
                description=f"{ctx['num_architectures']} different architectures available for crossover",
                context=ctx,
                journal_path=str(journal_path),
                lower_is_better=lower_is_better,
                priority=ctx['num_architectures'],
            )
            junctions.append(j)

    for step, ctx in find_repeated_failures(journal):
        j = Junction(
            task=task_name,
            step=step,
            junction_type="repeated_failure",
            description=f"{ctx['sequence_length']}x consecutive {ctx['error_type']} errors",
            context=ctx,
            journal_path=str(journal_path),
            lower_is_better=lower_is_better,
            priority=ctx['sequence_length'],
        )
        junctions.append(j)

    log.info(f"Found {len(junctions)} junctions for {task_name}")

    return JunctionResult(
        task=task_name,
        journal_path=str(journal_path),
        total_nodes=len(journal.nodes),
        best_metric=best_metric,
        junctions=junctions,
    )


def select_best_junction(junctions: List[Junction], task_name: str) -> Optional[Junction]:
    """
    Select the single most impactful junction for a task.

    Priority order:
    1. best_abandonment with highest metric gap
    2. repeated_failure (especially cv2, import errors)
    3. score_plateau with longest duration
    4. crossover_candidate with most architectures
    5. debug_chain with highest depth
    """
    if not junctions:
        return None

    # Score each junction
    def junction_score(j: Junction) -> float:
        base_scores = {
            "best_abandonment": 100,  # Highest priority - clear wrong choice
            "repeated_failure": 80,   # High priority - stuck in error loop
            "score_plateau": 60,      # Medium - needs fresh direction
            "crossover_candidate": 40, # Lower - opportunity not taken
            "debug_chain": 30,        # Lowest - but still a signal
        }
        base = base_scores.get(j.junction_type, 0)
        return base + j.priority

    return max(junctions, key=junction_score)


def find_journals(logs_dir: Path) -> List[Tuple[Path, str]]:
    """Find all journal.jsonl files and extract task names."""
    results = []

    for journal_path in logs_dir.rglob("checkpoint/journal.jsonl"):
        run_dir = journal_path.parent.parent
        run_dir_name = run_dir.name

        # First try: get task name from dojo_config.json
        task_name = None
        config_path = run_dir / "dojo_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                task_name = config.get("task", {}).get("name")
            except Exception as e:
                log.warning(f"Could not read config {config_path}: {e}")

        # Second try: get task name from journal content
        if not task_name:
            try:
                with open(journal_path) as f:
                    first_lines = [f.readline() for _ in range(10)]
                content = " ".join(first_lines)

                # Look for task name patterns
                task_patterns = [
                    r'"(tabular-playground[^"]+)"',
                    r'"(spooky-author[^"]+)"',
                    r'"(dog-breed[^"]+)"',
                    r'"(learning-agency[^"]+)"',
                    r'"(stanford-covid[^"]+)"',
                    r'"(mlsp-[^"]+)"',
                    r'"(jigsaw[^"]+)"',
                    r'"(denoising[^"]+)"',
                    r'"(icecube[^"]+)"',
                    r'"(nyc-taxi[^"]+)"',
                ]

                for pattern in task_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        task_name = match.group(1)
                        break
            except Exception as e:
                log.warning(f"Could not read {journal_path}: {e}")

        # Fall back to directory name
        if not task_name:
            task_name = run_dir_name.split("_id_")[0].replace("user_antonis_issue_QD_STUDY_evo_gdm_", "")

        results.append((journal_path, task_name))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Identify key decision junctions in EVO search runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Output path for junctions JSON",
    )
    parser.add_argument(
        "--single-per-task",
        action="store_true",
        default=True,
        help="Select only the best junction per task (default: True)",
    )
    parser.add_argument(
        "--all-junctions",
        action="store_true",
        help="Output all junctions, not just best per task",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Lower metric values are better (e.g., loss)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to top N junctions by priority",
    )

    args = parser.parse_args()

    if not args.logs_dir.exists():
        log.error(f"Logs directory not found: {args.logs_dir}")
        sys.exit(1)

    # Find all journals
    journals = find_journals(args.logs_dir)
    log.info(f"Found {len(journals)} journal files")

    if not journals:
        log.error("No journal files found")
        sys.exit(1)

    # Process each journal
    all_results = []
    best_junctions = []

    for journal_path, task_name in journals:
        try:
            result = identify_junctions_for_task(
                journal_path,
                task_name,
                lower_is_better=args.lower_is_better,
            )
            all_results.append(result)

            if result.junctions:
                best = select_best_junction(result.junctions, task_name)
                if best:
                    best_junctions.append(best)
        except Exception as e:
            log.error(f"Failed to process {journal_path}: {e}")
            import traceback
            traceback.print_exc()

    # Prepare output
    output_data = {
        "summary": {
            "total_tasks": len(all_results),
            "tasks_with_junctions": len([r for r in all_results if r.junctions]),
            "total_junctions": sum(len(r.junctions) for r in all_results),
            "selected_junctions": len(best_junctions),
        },
        "junctions": [],
    }

    if args.all_junctions:
        # Output all junctions grouped by task
        for result in all_results:
            for j in result.junctions:
                output_data["junctions"].append(asdict(j))
    else:
        # Output only best junction per task
        for j in best_junctions:
            output_data["junctions"].append(asdict(j))

    # Apply limit if specified
    if args.limit and len(output_data["junctions"]) > args.limit:
        # Sort by priority descending and take top N
        output_data["junctions"] = sorted(
            output_data["junctions"],
            key=lambda x: x.get("priority", 0),
            reverse=True
        )[:args.limit]
        output_data["summary"]["selected_junctions"] = len(output_data["junctions"])

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    log.info(f"Saved {len(output_data['junctions'])} junctions to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("Junction Identification Summary")
    print("=" * 60)
    print(f"Total tasks processed: {len(all_results)}")
    print(f"Tasks with junctions: {len([r for r in all_results if r.junctions])}")
    print(f"Total junctions found: {sum(len(r.junctions) for r in all_results)}")
    print(f"Selected junctions: {len(best_junctions)}")
    print()

    for j in best_junctions:
        print(f"  {j.task}:")
        print(f"    Step {j.step}: {j.junction_type}")
        print(f"    {j.description}")
        print()


if __name__ == "__main__":
    main()
