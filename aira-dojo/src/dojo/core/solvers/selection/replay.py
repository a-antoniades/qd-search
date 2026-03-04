# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Replay system for RLM-based node selection.

Enables prototyping and debugging RLM selection decisions using saved
journal data from previous EVO runs. Provides time-travel functionality
to reconstruct the tree state at any historical step.

Usage:
    from dojo.core.solvers.selection.replay import (
        snapshot_at_step,
        build_replay_context,
        replay_selection,
    )

    # Load journal and time-travel to step 15
    snapshot = snapshot_at_step(journal, step=15)

    # Build exact RLM input from that snapshot
    context = build_replay_context(snapshot)

    # Re-run selection with current RLM config
    result = replay_selection(context, cfg)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dojo.config_dataclasses.selector.rlm import RLMSelectorConfig
from dojo.core.solvers.selection.base import SelectionResult
from dojo.core.solvers.selection.rlm_selector import RLMNodeSelector
from dojo.core.solvers.selection.tree_serializer import serialize_for_rlm
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue

log = logging.getLogger(__name__)


@dataclass
class JournalSnapshot:
    """Journal state at a specific step (for time-travel)."""

    journal: Journal
    step: int
    leaf_steps: List[int]

    @property
    def num_nodes(self) -> int:
        return len(self.journal.nodes)


@dataclass
class ReplayContext:
    """Exact RLM input at a selection point."""

    step: int
    tree_context: Dict[str, Any]
    setup_code: str
    root_prompt: str
    leaf_steps: List[int]
    num_samples: int = 1


@dataclass
class ReplayResult:
    """Result of a replay execution."""

    step: int
    model: str
    leaf_steps: List[int]
    selected_steps: List[int]
    operator: str
    reasoning: str
    elapsed_s: float
    raw_response: str = ""
    error: Optional[str] = None


def load_journal_from_jsonl(path: Path) -> Journal:
    """
    Load a Journal from a JSONL checkpoint file.

    Each line is a node dict. Reconstructs the Journal with
    proper parent/child relationships.
    """
    path = Path(path)
    nodes_data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                nodes_data.append(json.loads(line))

    # Sort by step to ensure proper ordering
    nodes_data.sort(key=lambda x: x.get("step", 0))

    # Use Journal's from_export_data method
    export_data = {"nodes": nodes_data, "solution": ""}
    return Journal.from_export_data(export_data)


def snapshot_at_step(full_journal: Journal, target_step: int) -> JournalSnapshot:
    """
    Reconstruct journal state as it was at step N.

    Given a complete journal, creates a snapshot showing only the nodes
    that existed at the target step, with children lists filtered to
    only include nodes that existed at that point.

    Args:
        full_journal: The complete journal with all nodes.
        target_step: The step number to snapshot at.

    Returns:
        JournalSnapshot with filtered journal and identified leaf steps.
    """
    # Get all nodes up to target step
    nodes_data = []
    for node in full_journal.nodes:
        if node.step <= target_step:
            node_data = _node_to_dict(node)
            nodes_data.append(node_data)

    if not nodes_data:
        raise ValueError(f"No nodes found at or before step {target_step}")

    valid_steps = {d["step"] for d in nodes_data}

    # Filter children to only include valid steps
    for node_data in nodes_data:
        node_data["children"] = [
            c for c in node_data.get("children", []) if c in valid_steps
        ]

    # Identify leaves (nodes with no children at this point)
    leaf_steps = [
        d["step"] for d in nodes_data if not d.get("children")
    ]

    # Reconstruct journal
    export_data = {"nodes": nodes_data, "solution": ""}
    filtered_journal = Journal.from_export_data(export_data)

    return JournalSnapshot(
        journal=filtered_journal,
        step=target_step,
        leaf_steps=leaf_steps,
    )


def _node_to_dict(node: Node) -> Dict[str, Any]:
    """Convert a Node to a dict suitable for export."""
    if isinstance(node.metric, MetricValue):
        metric_value = node.metric.value
        metric_info = node.metric.info
        metric_maximize = node.metric.maximize
    elif isinstance(node.metric, WorstMetricValue):
        metric_value = None
        metric_info = None
        metric_maximize = True
    else:
        metric_value = None
        metric_info = None
        metric_maximize = True

    children = [c.step for c in node.children] if node.children else []
    parents = [p.step for p in node.parents] if node.parents else []

    return {
        "step": node.step,
        "id": node.id,
        "plan": node.plan,
        "code": node.code,
        "metric": metric_value,
        "metric_info": metric_info,
        "metric_maximize": metric_maximize,
        "is_buggy": node.is_buggy,
        "analysis": node.analysis,
        "operators_metrics": getattr(node, "operators_metrics", []),
        "children": children,
        "parents": parents,
        "creation_time": node.ctime,
        "term_out": node.term_out,
        "operators_used": getattr(node, "operators_used", []),
        "exec_time": node.exec_time,
        "exit_code": node.exit_code,
        "_term_out": node._term_out,
    }


def build_replay_context(
    snapshot: JournalSnapshot,
    num_samples: int = 1,
    full_context: bool = False,
    max_nodes: int = 100,
    lower_is_better: bool = False,
    step_limit: int = 100,
) -> ReplayContext:
    """
    Build the exact RLM input from a snapshot.

    Creates all the data that would be passed to the RLM at this
    selection point, enabling exact replay of decisions.

    Args:
        snapshot: The journal snapshot to build context from.
        num_samples: Number of nodes to select.
        full_context: If True, include full node data.
        max_nodes: Maximum nodes to include.
        lower_is_better: If True, lower metric values are better (e.g., loss, error).
        step_limit: Total number of steps in the search (for step-awareness).

    Returns:
        ReplayContext with all RLM inputs.
    """
    from dojo.core.solvers.selection.rlm_selector import _create_setup_code

    # Serialize tree for RLM
    tree_context = serialize_for_rlm(
        snapshot.journal,
        full=full_context,
        max_nodes=max_nodes,
        lower_is_better=lower_is_better,
    )

    leaf_steps = tree_context.get("leaf_steps", [])

    # Build search state for step-awareness
    current_step = snapshot.step
    search_state = {
        "current_step": current_step,
        "step_limit": step_limit,
        "steps_remaining": step_limit - current_step,
        "steps_used_pct": round(current_step / step_limit * 100, 1) if step_limit > 0 else 100,
    }

    # Create setup code with helpers
    setup_code = _create_setup_code(tree_context, search_state)

    # Build root prompt
    root_prompt = (
        f"Select {num_samples} node(s) to expand.\n\n"
        f"VALID CHOICES (leaf_steps): {leaf_steps}\n\n"
        f"You MUST select step values from this list only."
    )

    return ReplayContext(
        step=snapshot.step,
        tree_context=tree_context,
        setup_code=setup_code,
        root_prompt=root_prompt,
        leaf_steps=leaf_steps,
        num_samples=num_samples,
    )


def replay_selection(
    context: ReplayContext,
    cfg: RLMSelectorConfig,
    lower_is_better: bool = False,
) -> ReplayResult:
    """
    Execute RLM with reconstructed context.

    Args:
        context: The replay context with RLM inputs.
        cfg: RLM selector configuration.
        lower_is_better: If True, lower metric values are better.

    Returns:
        ReplayResult with selection outcome and timing.
    """
    start_time = time.time()

    selector = RLMNodeSelector(cfg=cfg, lower_is_better=lower_is_better)

    try:
        rlm = selector._get_rlm()

        # Store and modify environment kwargs
        original_env_kwargs = rlm.environment_kwargs.copy()
        rlm.environment_kwargs["setup_code"] = context.setup_code

        try:
            result = rlm.completion(
                prompt=context.tree_context,
                root_prompt=context.root_prompt,
            )

            elapsed = time.time() - start_time

            # Parse the response directly without needing full journal
            selection = _parse_replay_response(
                result.response,
                context.leaf_steps,
            )

            return ReplayResult(
                step=context.step,
                model=cfg.model_name,
                leaf_steps=context.leaf_steps,
                selected_steps=selection["selected_steps"],
                operator=selection["operator"],
                reasoning=selection["reasoning"],
                elapsed_s=elapsed,
                raw_response=result.response[:2000] if result.response else "",
            )

        finally:
            rlm.environment_kwargs = original_env_kwargs

    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"Replay failed: {e}")
        return ReplayResult(
            step=context.step,
            model=cfg.model_name,
            leaf_steps=context.leaf_steps,
            selected_steps=[],
            operator="",
            reasoning="",
            elapsed_s=elapsed,
            error=str(e),
        )


def _parse_replay_response(
    response: str,
    leaf_steps: List[int],
) -> Dict[str, Any]:
    """
    Parse RLM response for replay purposes (no full journal needed).

    Returns dict with selected_steps, operator, reasoning.
    """
    import ast
    import re

    if response is None:
        raise ValueError("RLM returned None response")

    def safe_literal_eval(s: str) -> Optional[Dict]:
        """Parse Python dict repr, handling embedded newlines in strings."""
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            # Try escaping literal newlines inside strings
            escaped = s.replace('\n', r'\n')
            try:
                return ast.literal_eval(escaped)
            except (ValueError, SyntaxError):
                return None

    selection = None

    # Try to extract FINAL_VAR from Python code
    if "FINAL_VAR" in response:
        match = re.search(r'FINAL_VAR\s*=\s*(\{.*)', response, re.DOTALL)
        if match:
            dict_str = match.group(1)
            # Find matching brace
            brace_count = 0
            end_idx = 0
            for i, c in enumerate(dict_str):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            if end_idx > 0:
                dict_str = dict_str[:end_idx]
            # Fix common JSON issues
            dict_str = re.sub(r',\s*]', ']', dict_str)
            dict_str = re.sub(r',\s*}', '}', dict_str)
            selection = safe_literal_eval(dict_str)
            if selection is None:
                try:
                    selection = json.loads(dict_str)
                except json.JSONDecodeError:
                    pass

    # Try direct Python dict parse (with newline handling)
    if selection is None:
        selection = safe_literal_eval(response.strip())

    # Try direct JSON parse
    if selection is None:
        try:
            clean = re.sub(r',\s*]', ']', response)
            clean = re.sub(r',\s*}', '}', clean)
            selection = json.loads(clean)
        except json.JSONDecodeError:
            pass

    # Try code block extraction
    if selection is None:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
            try:
                selection = json.loads(json_str)
            except json.JSONDecodeError:
                pass
        elif "```python" in response or "```repl" in response:
            for marker in ["```python", "```repl"]:
                if marker in response:
                    code = response.split(marker)[1].split("```")[0]
                    if "FINAL_VAR" in code:
                        # Recursive call with just the code block
                        return _parse_replay_response(code, leaf_steps)

    if selection is None or not isinstance(selection, dict):
        raise ValueError(f"Could not parse RLM response: {response[:500]}")

    # Extract selected steps
    selected_steps = []
    for entry in selection.get("selected_nodes", []):
        step = entry.get("step")
        if step is None:
            continue
        if step not in leaf_steps:
            log.warning(f"RLM selected non-leaf step {step}, skipping")
            continue
        selected_steps.append(step)

    operator = selection.get("operator", "improve")
    if operator not in ["improve", "crossover", "draft"]:
        operator = "improve"

    # For DRAFT, empty selected_nodes is valid (no parent needed)
    if not selected_steps and operator != "draft":
        raise ValueError("RLM selected no valid nodes")
    if operator not in ["improve", "crossover", "draft"]:
        operator = "improve"

    return {
        "selected_steps": selected_steps,
        "operator": operator,
        "reasoning": selection.get("reasoning", ""),
    }


def load_selection_history(path: Path) -> List[Dict[str, Any]]:
    """Load selection history from a JSONL file."""
    path = Path(path)
    history = []
    if not path.exists():
        return history
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def compare_selections(
    replay_result: ReplayResult,
    original_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare a replay result against the original selection.

    Args:
        replay_result: The replayed selection result.
        original_record: The original selection record from history.

    Returns:
        Comparison dict with match info and differences.
    """
    original_selected = [
        n["step"] for n in original_record.get("selected_nodes", [])
    ]
    original_operator = original_record.get("operator", "")

    steps_match = set(replay_result.selected_steps) == set(original_selected)
    operator_match = replay_result.operator == original_operator

    return {
        "step": replay_result.step,
        "steps_match": steps_match,
        "operator_match": operator_match,
        "original_selected": original_selected,
        "replay_selected": replay_result.selected_steps,
        "original_operator": original_operator,
        "replay_operator": replay_result.operator,
        "original_reasoning": original_record.get("reasoning", "")[:200],
        "replay_reasoning": replay_result.reasoning[:200],
    }
