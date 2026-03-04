# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tree serialization for RLM-based node selection.

Simple serialization that passes journal data to the RLM,
letting the LLM do its own analysis of plans, code, and errors.
"""

from typing import Any, Dict, List

from dojo.core.solvers.utils.journal import Journal


def serialize_for_rlm(
    journal: Journal,
    full: bool = True,
    max_nodes: int = 100,
    lower_is_better: bool = False,
) -> Dict[str, Any]:
    """
    Serialize journal for RLM context.

    Plan summarization is now done via llm_query_batched() at runtime,
    so we include fuller plan text for top candidates rather than truncated previews.

    Args:
        journal: The search journal to serialize.
        full: If True, include all node data (plan, code, term_out, metric_info).
              If False, just essentials (step, metric, is_buggy, parents, children).
        max_nodes: Maximum number of nodes to include.
        lower_is_better: If True, lower metric values are better (e.g., loss, error).

    Returns:
        Dict with summary, leaf_steps, nodes, and metric_maximize.
    """
    # Max plan chars per node - enough for full approach description
    # The RLM will use llm_query_batched() to summarize these
    MAX_PLAN_CHARS = 3000

    export_data = journal.export_data()
    nodes_list = export_data.get("nodes", [])

    # Calculate summary
    good_nodes = [n for n in nodes_list if not n.get("is_buggy", True)]
    buggy_nodes = [n for n in nodes_list if n.get("is_buggy", True)]
    leaf_nodes = [n for n in nodes_list if not n.get("children")]
    metrics = [n.get("metric") for n in good_nodes if n.get("metric") is not None]

    summary = {
        "total_nodes": len(nodes_list),
        "good_nodes": len(good_nodes),
        "buggy_nodes": len(buggy_nodes),
        "best_metric": max(metrics) if metrics else None,
        "worst_metric": min(metrics) if metrics else None,
        "leaf_count": len(leaf_nodes),
    }

    leaf_steps = [n["step"] for n in leaf_nodes]
    metric_maximize = not lower_is_better

    # Limit nodes if too many
    if len(nodes_list) > max_nodes:
        # Keep root + best nodes + recent nodes
        sorted_by_metric = sorted(
            [n for n in good_nodes if n.get("metric")],
            key=lambda x: x.get("metric", 0),
            reverse=metric_maximize,
        )
        top_nodes = sorted_by_metric[:max_nodes // 2]
        recent_nodes = sorted(nodes_list, key=lambda x: x.get("step", 0), reverse=True)[:max_nodes // 2]
        keep_steps = set([0]) | {n["step"] for n in top_nodes} | {n["step"] for n in recent_nodes}
        nodes_list = [n for n in nodes_list if n["step"] in keep_steps]

    # Serialize nodes
    # Only include fields relevant for selection (exclude huge fields like code, operators_metrics)
    SELECTION_FIELDS = {"step", "metric", "is_buggy", "parents", "children", "plan", "analysis"}

    if full:
        # Include plan text (truncated to MAX_PLAN_CHARS) for semantic analysis
        nodes = []
        for n in nodes_list:
            node_copy = {k: v for k, v in n.items() if k in SELECTION_FIELDS}
            plan = node_copy.get("plan", "") or ""
            if len(plan) > MAX_PLAN_CHARS:
                node_copy["plan"] = plan[:MAX_PLAN_CHARS] + "..."
            nodes.append(node_copy)
    else:
        # Minimal mode - just structure
        nodes = [
            {
                "step": n["step"],
                "metric": n.get("metric"),
                "is_buggy": n.get("is_buggy", False),
                "parents": n.get("parents", []),
                "children": n.get("children", []),
            }
            for n in nodes_list
        ]

    return {
        "summary": summary,
        "leaf_steps": leaf_steps,
        "nodes": nodes,
        "metric_maximize": metric_maximize,
    }


def generate_ascii_tree(journal: Journal, max_depth: int = 5) -> str:
    """Generate ASCII art representation of tree structure."""
    export_data = journal.export_data()
    nodes_map = {n["step"]: n for n in export_data.get("nodes", [])}
    lines = []

    def format_node(step: int) -> str:
        node = nodes_map.get(step)
        if not node:
            return f"[{step}:?]"
        metric = node.get("metric")
        buggy = "!" if node.get("is_buggy") else ""
        metric_str = f"{metric:.3f}" if metric else "null"
        return f"[{step}:{metric_str}{buggy}]"

    def print_tree(step: int, prefix: str = "", is_last: bool = True, depth: int = 0):
        if depth > max_depth:
            return
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + format_node(step))
        node = nodes_map.get(step)
        if node:
            children = node.get("children", [])
            for i, child_step in enumerate(children):
                is_child_last = i == len(children) - 1
                child_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(child_step, child_prefix, is_child_last, depth + 1)

    roots = [n["step"] for n in export_data.get("nodes", []) if not n.get("parents")]
    for i, root in enumerate(roots):
        print_tree(root, "", i == len(roots) - 1)

    return "\n".join(lines)
