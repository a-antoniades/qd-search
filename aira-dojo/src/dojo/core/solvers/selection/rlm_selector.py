# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
RLM-based intelligent node selection.

Uses a Recursive Language Model to analyze the search tree and select
promising nodes for expansion based on plans, architectures, metrics,
and error patterns.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dojo.core.solvers.selection.base import NodeSelector, SelectedNode, SelectionResult
from dojo.core.solvers.selection.tree_serializer import serialize_for_rlm
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.config_dataclasses.selector.rlm import RLMSelectorConfig

log = logging.getLogger(__name__)


# System prompt for RLM tree selection - Deliberative operator selection
RLM_SYSTEM_PROMPT = '''
You are selecting an operator for evolutionary ML search.

## Goal
Choose the BEST operator by deliberating on trade-offs.
No operator has a default advantage - evaluate each fairly.

## Operators
- DRAFT: Generate fresh solution (0 parents) - explore new directions
- IMPROVE: Refine existing solution (1 parent) - exploit promising approach
- CROSSOVER: Combine two solutions (2 parents) - merge complementary ideas

## Available Tools
- `get_search_state()` → {current_step, step_limit, steps_used_pct}
- `get_best_leaves(n)` → top n leaf nodes by metric
- `get_node(step)` → full node data including plan
- `compare_nodes(s1, s2)` → metric gap between nodes
- `get_context()` → full tree context dict
- `detect_plateau(lookback)` → {plateau, best, recent_best}
- `get_approach_summary(step)` → cached 8-word summary or ''
- `get_cached_approaches()` → all cached summaries dict
- `llm_query(prompt)` → ask sub-LM for reasoning
- `llm_query_batched(prompts)` → parallel plan summarization

## Pipeline: Gather Evidence → Deliberate → Select Winner

**IMPORTANT**: All code MUST be in ```repl``` blocks (not ```python```).

### Step 1: Gather Evidence
```repl
state = get_search_state()
candidates = get_best_leaves(5)
ctx = get_context()
valid_leaves = set(ctx.get('leaf_steps', []))

# Get approach summaries (check cache first, then batch-query uncached)
plans = {c['step']: get_node(c['step']).get('plan', '')[:1500] for c in candidates}
approaches = {}
uncached_steps = []
for step in plans.keys():
    cached = get_approach_summary(step)
    if cached:
        approaches[step] = cached
    else:
        uncached_steps.append(step)

if uncached_steps:
    prompts = [f"In 8 words, what ML approach? (model type, key technique)\\n{plans[s]}" for s in uncached_steps]
    new_sums = llm_query_batched(prompts)
    for step, summary in zip(uncached_steps, new_sums):
        approaches[step] = summary

print(f"Approach summaries: {len(approaches)-len(uncached_steps)} cached, {len(uncached_steps)} new")

# Compute evidence
plateau_info = detect_plateau(10)
unique_approaches = len(set(approaches.values()))
converged = unique_approaches <= 2

# Find ALL crossover pairs (different approaches - no metric filtering)
# Let the RLM reason about which pairs have complementary strengths
crossover_pairs = []
steps = [c['step'] for c in candidates if c['step'] in valid_leaves]
for i, s1 in enumerate(steps):
    for s2 in steps[i+1:]:
        a1, a2 = approaches.get(s1, ''), approaches.get(s2, '')
        if a1 != a2:  # Only requirement: different approaches
            m1 = next((c['metric'] for c in candidates if c['step'] == s1), None)
            m2 = next((c['metric'] for c in candidates if c['step'] == s2), None)
            crossover_pairs.append({
                'steps': [s1, s2],
                'metrics': [round(m1, 4) if m1 else None, round(m2, 4) if m2 else None],
                'approaches': [a1[:80], a2[:80]]
            })

# Print evidence summary
print(f"Search: {state['steps_used_pct']}% used, {state['steps_remaining']} left")
print(f"Plateau: {plateau_info['plateau']}")
print(f"Diversity: {unique_approaches} unique approaches, converged={converged}")
print(f"Top candidates: {[(c['step'], round(c['metric'],4)) for c in candidates[:3]]}")
print(f"Crossover candidates: {len(crossover_pairs)} pairs with different approaches")
for p in crossover_pairs[:5]:
    print(f"  {p['steps']}: {p['metrics']} - {p['approaches']}")
```

### Step 2: Deliberate on Each Operator
```repl
deliberation_prompt = f"""
Given this evidence, choose the best operator by weighing trade-offs.

EVIDENCE:
- Search progress: {state['steps_used_pct']}% budget used, {state['steps_remaining']} steps remaining
- Plateau: {plateau_info['plateau']} (best={plateau_info['best']}, recent={plateau_info['recent_best']})
- Diversity: {unique_approaches} unique approaches out of {len(candidates)} candidates
- Converged: {converged} (all top nodes use similar approach)
- Top nodes: {[(c['step'], round(c['metric'],4), approaches.get(c['step'],'')[:40]) for c in candidates[:3]]}

CROSSOVER CANDIDATES ({len(crossover_pairs)} pairs with different approaches):
{crossover_pairs[:5]}

For each operator, state ONE pro and ONE con given THIS evidence:

DRAFT:
+ [pro]
- [con]

IMPROVE:
+ [pro]
- [con]

CROSSOVER:
+ [pro] Consider: Are any pairs COMPLEMENTARY? Could combining different paradigms yield synergy?
        A weaker approach might contribute valuable techniques (features, architecture, training) to a stronger one.
- [con]

BEST: [DRAFT or IMPROVE or CROSSOVER]
WHY: [one sentence explaining why this operator wins]
"""
deliberation = llm_query(deliberation_prompt)
print(deliberation)
```

### Step 3: Parse Decision
```repl
# Parse the chosen operator from deliberation
deliberation_upper = deliberation.upper()

# Look for "BEST: OPERATOR" pattern
winner = None
if 'BEST:' in deliberation_upper:
    after_best = deliberation_upper.split('BEST:')[1]
    for op in ['CROSSOVER', 'IMPROVE', 'DRAFT']:  # Check in QD-priority order
        if op in after_best.split('\\n')[0]:  # Only check the line with BEST:
            winner = op.lower()
            break

# Fallback: look for operator mentioned near the end
if not winner:
    last_100 = deliberation_upper[-100:]
    for op in ['CROSSOVER', 'IMPROVE', 'DRAFT']:
        if op in last_100:
            winner = op.lower()
            break

# Final fallback
if not winner:
    winner = 'improve'

print(f"Winner: {winner}")
```

### Step 4: Build Selection Based on Winner
```repl
if winner == 'draft':
    selected = []
    reasoning = "Fresh exploration based on deliberation"

elif winner == 'crossover' and crossover_pairs:
    pair = crossover_pairs[0]
    selected = [
        {"step": pair['steps'][0], "priority": 1, "reason": pair['approaches'][0]},
        {"step": pair['steps'][1], "priority": 2, "reason": pair['approaches'][1]}
    ]
    reasoning = f"Crossover: combining {pair['approaches'][0][:40]} + {pair['approaches'][1][:40]}"

else:  # improve (or crossover without valid pairs)
    if winner == 'crossover':
        winner = 'improve'  # Fallback if no valid crossover pairs
    best = candidates[0]
    selected = []
    if best['step'] in valid_leaves:
        selected = [{"step": best['step'], "priority": 1, "reason": approaches.get(best['step'], '')[:80]}]
    else:
        for c in candidates:
            if c['step'] in valid_leaves:
                selected = [{"step": c['step'], "priority": 1, "reason": approaches.get(c['step'], '')[:80]}]
                break
    reasoning = "Improve best valid candidate"

result = {
    "selected_nodes": selected,
    "operator": winner,
    "deliberation": deliberation[:500],
    "reasoning": reasoning,
    "approach_summaries": approaches  # For cache update
}
print(f"Decision: {winner} on {[s['step'] for s in selected]}")
```

### Step 5: Return Result
```repl
FINAL_VAR("result")
```

## Key Principles

1. EQUAL PRIORS - No operator starts with an advantage
2. EVIDENCE-BASED - Compute crossover pairs, plateau, diversity upfront
3. DELIBERATION - LLM weighs pros/cons, then decides
4. DIRECT DECISION - No arbitrary scores, just the best operator

## Critical Rules
- Use ```repl``` code blocks for all code execution (NOT ```python```)
- DO NOT call FINAL_VAR() until Step 5 - it terminates execution immediately
- ALWAYS use llm_query_batched() to summarize plans (not string matching)
- Every step value MUST be from context["leaf_steps"]
- The argument to FINAL_VAR() is the VARIABLE NAME as a string, not the value
- For crossover: select EXACTLY 2 nodes with DIFFERENT approaches
- For draft: selected_nodes should be empty list []
'''


def _create_setup_code(tree: Dict, search_state: Dict, approach_cache: Dict[int, str] = None) -> str:
    """Generate setup code with tree helpers and search state.

    Note: llm_query() and llm_query_batched() are provided by the RLM environment,
    not defined here. They allow the RLM to make sub-LM calls for semantic analysis.
    """
    tree_json = json.dumps(tree, default=str)
    search_state_json = json.dumps(search_state, default=str)

    return f'''
import json

# Tree data and helpers
_TREE = json.loads({repr(tree_json)})
_NODES_MAP = {{n["step"]: n for n in _TREE["nodes"]}}
_METRIC_MAXIMIZE = _TREE.get("metric_maximize", True)
_LEAF_STEPS = set(_TREE.get("leaf_steps", []))
_SEARCH_STATE = json.loads({repr(search_state_json)})

# Approach summary cache (step -> 8-word summary)
_APPROACH_CACHE = {repr(approach_cache or {})}

def get_approach_summary(step: int) -> str:
    """Get cached approach summary for a node, or empty string if not cached."""
    return _APPROACH_CACHE.get(step, '')

def get_cached_approaches() -> dict:
    """Get all cached approach summaries."""
    return _APPROACH_CACHE.copy()

def get_node(step: int) -> dict:
    """Get full node data including complete plan text."""
    return _NODES_MAP.get(step, {{"error": f"Node {{step}} not found"}})

def get_best_leaves(n: int = 5) -> list:
    """Get top n leaf nodes by metric, respecting metric direction."""
    leaves = [x for x in _TREE["nodes"]
              if not x.get("children")
              and x.get("metric") is not None
              and not x.get("is_buggy")]
    return sorted(leaves, key=lambda x: x["metric"], reverse=_METRIC_MAXIMIZE)[:n]

def compare_nodes(step1: int, step2: int) -> dict:
    """Compare two nodes: metrics, gap percentage."""
    n1, n2 = _NODES_MAP.get(step1, {{}}), _NODES_MAP.get(step2, {{}})
    m1, m2 = n1.get("metric"), n2.get("metric")

    if m1 is not None and m2 is not None and max(abs(m1), abs(m2)) > 0:
        gap_pct = round(abs(m1 - m2) / max(abs(m1), abs(m2)) * 100, 2)
    else:
        gap_pct = None

    return {{
        "step1": step1, "metric1": m1,
        "step2": step2, "metric2": m2,
        "metric_gap_pct": gap_pct
    }}

def get_context() -> dict:
    """Get the full tree context dict (summary, leaf_steps, nodes, metric_maximize)."""
    return _TREE

def get_search_state() -> dict:
    """Get current search progress: current_step, step_limit, steps_remaining, steps_used_pct."""
    return _SEARCH_STATE

def get_underexplored_leaves(max_children: int = 1) -> list:
    """Find leaf nodes with few children (underexplored)."""
    leaves = [n for n in _TREE["nodes"]
              if not n.get("children")
              and not n.get("is_buggy")
              and n.get("metric") is not None]
    # Filter by number of siblings (how much this parent was explored)
    underexplored = []
    for leaf in leaves:
        parents = leaf.get("parents", [])
        if parents:
            parent = _NODES_MAP.get(parents[0], {{}})
            num_siblings = len(parent.get("children", []))
        else:
            num_siblings = 0
        if num_siblings <= max_children:
            underexplored.append({{
                "step": leaf["step"],
                "metric": leaf["metric"],
                "siblings": num_siblings
            }})
    return sorted(underexplored, key=lambda x: x["metric"], reverse=_METRIC_MAXIMIZE)

def detect_plateau(lookback: int = 10) -> dict:
    """Check if best metric has improved in recent steps."""
    current_step = _SEARCH_STATE.get("current_step", 0)
    all_nodes = sorted(_TREE["nodes"], key=lambda x: x["step"])

    # Best overall (non-buggy with valid metric)
    valid = [n for n in all_nodes if n.get("metric") is not None and not n.get("is_buggy")]
    if not valid:
        return {{"plateau": False, "best": None, "recent_best": None}}

    if _METRIC_MAXIMIZE:
        best = max(valid, key=lambda x: x["metric"])
    else:
        best = min(valid, key=lambda x: x["metric"])

    # Recent best (within lookback steps)
    recent = [n for n in valid if n["step"] > current_step - lookback]
    if recent:
        if _METRIC_MAXIMIZE:
            recent_best = max(recent, key=lambda x: x["metric"])
        else:
            recent_best = min(recent, key=lambda x: x["metric"])
    else:
        recent_best = None

    # Plateau if recent best is not better than overall best (within 0.1%)
    if recent_best is not None:
        if _METRIC_MAXIMIZE:
            plateau = recent_best["metric"] <= best["metric"] * 1.001
        else:
            plateau = recent_best["metric"] >= best["metric"] * 0.999
    else:
        plateau = True  # No recent nodes = stalled

    return {{
        "plateau": plateau,
        "best": best["metric"],
        "recent_best": recent_best["metric"] if recent_best else None
    }}

# Note: llm_query() and llm_query_batched() are provided by the RLM environment
print("Tree helpers ready: get_node, get_best_leaves, compare_nodes, get_context")
print("Search helpers ready: get_search_state, get_underexplored_leaves, detect_plateau")
print("Cache helpers ready: get_approach_summary, get_cached_approaches")
print("LLM tools available: llm_query(prompt), llm_query_batched(prompts)")
print(f"Valid leaf_steps: {{sorted(_LEAF_STEPS)}}")
print(f"Search state: {{_SEARCH_STATE}}")
'''


class RLMNodeSelector(NodeSelector):
    """
    RLM-based intelligent node selector.

    Uses a language model to analyze the search tree and select
    promising nodes based on rich context including plans,
    architectures, and error patterns.
    """

    def __init__(
        self,
        cfg: RLMSelectorConfig,
        lower_is_better: bool,
    ):
        """
        Initialize the RLM selector.

        Args:
            cfg: RLM selector configuration.
            lower_is_better: If True, lower metric values are better.
        """
        self.cfg = cfg
        self.lower_is_better = lower_is_better
        self._rlm = None

        # Lazy import fitness selector for fallback
        self._fitness_selector = None

        # Selection history for analysis
        self.selection_history: List[Dict[str, Any]] = []

        # Approach summary cache: step -> 8-word summary
        # Persists across select() calls to avoid regenerating summaries
        self.approach_cache: Dict[int, str] = {}

    @property
    def selector_type(self) -> str:
        return "rlm"

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        api_key = os.environ.get(self.cfg.api_key_env, "")
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{self.cfg.api_key_env}'"
            )
        return api_key

    def _get_rlm(self):
        """Lazy initialization of RLM."""
        if self._rlm is None:
            # Add RLM to path if needed
            rlm_path = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "external" / "rlm"
            if str(rlm_path) not in sys.path and rlm_path.exists():
                sys.path.insert(0, str(rlm_path))

            try:
                from rlm import RLM
            except ImportError as e:
                raise ImportError(
                    f"RLM package not found. Install it or ensure it's in the path. Error: {e}"
                )

            self._rlm = RLM(
                backend=self.cfg.backend,
                backend_kwargs={
                    "model_name": self.cfg.model_name,
                    "api_key": self._get_api_key(),
                },
                environment="local",
                max_iterations=self.cfg.max_iterations,
                custom_system_prompt=RLM_SYSTEM_PROMPT,
                verbose=self.cfg.verbose,
            )

        return self._rlm

    def _get_fitness_selector(self):
        """Get fitness selector for fallback."""
        if self._fitness_selector is None:
            from dojo.core.solvers.selection.fitness_selector import FitnessNodeSelector
            self._fitness_selector = FitnessNodeSelector(
                lower_is_better=self.lower_is_better,
                verbose=self.cfg.verbose,
            )
        return self._fitness_selector

    def _fix_json(self, text: str) -> str:
        """Fix common LLM JSON mistakes like trailing commas."""
        import re
        # Remove trailing commas before ] or }
        text = re.sub(r',\s*]', ']', text)
        text = re.sub(r',\s*}', '}', text)
        return text

    def _extract_final_var(self, text: str) -> Optional[Dict]:
        """Extract FINAL_VAR dict from Python code."""
        import re
        import ast

        # Try to find FINAL_VAR = {...} pattern
        match = re.search(r'FINAL_VAR\s*=\s*(\{.*)', text, re.DOTALL)
        if not match:
            return None

        # Find the dict - balance braces
        dict_str = match.group(1)
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

        if end_idx == 0:
            # Unbalanced - try to close it
            dict_str = dict_str.rstrip() + '}'
        else:
            dict_str = dict_str[:end_idx]

        dict_str = self._fix_json(dict_str)

        try:
            return ast.literal_eval(dict_str)
        except (ValueError, SyntaxError):
            try:
                return json.loads(dict_str)
            except json.JSONDecodeError:
                return None

    def _parse_response(
        self,
        response: str,
        journal: Journal,
        leaf_steps: List[int],
    ) -> SelectionResult:
        """Parse RLM response into SelectionResult.

        The RLM returns the stringified value of the FINAL_VAR variable.
        This may be a dict repr, JSON, or wrapped in code blocks.
        """
        import ast
        import re

        selection = None

        # Clean the response - strip whitespace and common wrapper text
        cleaned = response.strip()

        # Helper to safely parse Python dict repr with embedded newlines
        def safe_literal_eval(s: str) -> Optional[Dict]:
            """Parse Python dict repr, handling embedded newlines in strings."""
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError) as e1:
                log.debug(f"First literal_eval failed: {e1}")
                # Try escaping literal newlines inside strings
                # Replace actual newlines with escaped \n sequences
                # Use raw string r'\n' to produce the literal backslash-n
                escaped = s.replace('\n', r'\n')
                log.debug(f"Escaped string (first 200 chars): {escaped[:200]}")
                try:
                    return ast.literal_eval(escaped)
                except (ValueError, SyntaxError) as e2:
                    log.debug(f"Second literal_eval failed: {e2}")
                    return None

        # Try direct JSON parse first (RLM may return stringified dict)
        try:
            selection = json.loads(self._fix_json(cleaned))
        except json.JSONDecodeError:
            pass

        # Try Python literal eval (for dict repr format with integer keys)
        if selection is None:
            selection = safe_literal_eval(cleaned)

        # Try to extract from FINAL_VAR assignment (legacy format)
        if selection is None and "FINAL_VAR" in response:
            selection = self._extract_final_var(response)

        # Try extracting from code blocks
        if selection is None:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
                try:
                    selection = json.loads(self._fix_json(json_str))
                except json.JSONDecodeError:
                    pass
            elif "```python" in response or "```repl" in response:
                # Try both python and repl code blocks
                for marker in ["```python", "```repl"]:
                    if marker in response:
                        code = response.split(marker)[1].split("```")[0]
                        selection = self._extract_final_var(code)
                        if selection:
                            break
            elif "```" in response:
                code_block = response.split("```")[1].split("```")[0]
                try:
                    selection = json.loads(self._fix_json(code_block))
                except json.JSONDecodeError:
                    selection = safe_literal_eval(code_block)
                    if selection is None:
                        selection = self._extract_final_var(code_block)

        if selection is None or not isinstance(selection, dict):
            log.warning(f"Could not parse RLM response: {response[:500]}")
            raise ValueError("Failed to parse RLM response")

        # Build SelectedNode objects
        selected_nodes = []
        nodes_map = {n.step: n for n in journal.nodes}

        for entry in selection.get("selected_nodes", []):
            step = entry.get("step")
            if step is None:
                continue

            # Validate step is a leaf
            if step not in leaf_steps:
                log.warning(f"RLM selected non-leaf step {step}, skipping")
                continue

            node = nodes_map.get(step)
            if node is None:
                log.warning(f"Node {step} not found in journal, skipping")
                continue

            selected_nodes.append(
                SelectedNode(
                    step=step,
                    priority=entry.get("priority", len(selected_nodes) + 1),
                    reason=entry.get("reason", "RLM selection"),
                    node=node,
                )
            )

        operator = selection.get("operator", "improve")
        if operator not in ["improve", "crossover", "draft"]:
            operator = "improve"

        # For DRAFT, empty selected_nodes is valid (no parent needed)
        if not selected_nodes and operator != "draft":
            raise ValueError("RLM selected no valid nodes")

        return SelectionResult(
            selected_nodes=selected_nodes,
            operator=operator,
            reasoning=selection.get("reasoning", ""),
            tree_insights=selection.get("tree_insights", ""),
            metadata={
                "rlm_response": response[:1000],
                "approach_summaries": selection.get("approach_summaries", {}),
            },
        )

    def select(
        self,
        journal: Journal,
        context: Dict[str, Any],
    ) -> SelectionResult:
        """
        Select nodes using RLM-based analysis.

        Args:
            journal: The search journal containing all nodes.
            context: Additional context (may include num_samples, islands,
                     step_limit, current_step for step-awareness).

        Returns:
            SelectionResult with selected nodes and operator.
        """
        try:
            # Serialize tree for RLM
            tree_context = serialize_for_rlm(
                journal,
                full=self.cfg.full_context,
                max_nodes=self.cfg.max_nodes,
                lower_is_better=self.lower_is_better,
            )

            leaf_steps = tree_context.get("leaf_steps", [])
            if not leaf_steps:
                log.warning("No leaf nodes available for selection")
                return SelectionResult(
                    selected_nodes=[],
                    operator="draft",
                    reasoning="No leaf nodes available",
                )

            # Extract search state for step-awareness
            current_step = context.get("current_step", len(journal.nodes))
            step_limit = context.get("step_limit", 100)
            steps_remaining = context.get("steps_remaining", step_limit - current_step)
            search_state = {
                "current_step": current_step,
                "step_limit": step_limit,
                "steps_remaining": steps_remaining,
                "steps_used_pct": round(current_step / step_limit * 100, 1) if step_limit > 0 else 100,
            }

            # Create setup code with helpers, search state, and approach cache
            setup_code = _create_setup_code(tree_context, search_state, self.approach_cache)

            # Initialize RLM with setup code
            rlm = self._get_rlm()

            # Store original environment kwargs
            original_env_kwargs = rlm.environment_kwargs.copy()
            rlm.environment_kwargs["setup_code"] = setup_code

            try:
                # Run RLM completion
                num_to_select = context.get("num_samples", {}).get("improve", 1)
                root_prompt = (
                    f"Select {num_to_select} node(s) to expand.\n\n"
                    f"VALID CHOICES (leaf_steps): {leaf_steps}\n\n"
                    f"You MUST select step values from this list only."
                )

                result = rlm.completion(
                    prompt=tree_context,
                    root_prompt=root_prompt,
                )

                # Parse response
                selection = self._parse_response(
                    result.response,
                    journal,
                    leaf_steps,
                )

                # Update approach cache from RLM result
                approach_summaries = selection.metadata.get("approach_summaries", {})
                for step, summary in approach_summaries.items():
                    self.approach_cache[int(step)] = summary

                log.info(
                    f"RLM selected {len(selection.selected_nodes)} nodes: "
                    f"{[sn.step for sn in selection.selected_nodes]}"
                )

                # Record selection for analysis
                self._record_selection(
                    selection=selection,
                    tree_context=tree_context,
                    rlm_response=result.response,
                    context=context,
                )

                return selection

            finally:
                # Restore original environment kwargs
                rlm.environment_kwargs = original_env_kwargs

        except Exception as e:
            log.error(f"RLM selection failed: {e}")

            if self.cfg.fallback_to_fitness:
                log.info("Falling back to fitness selector")
                return self._get_fitness_selector().select(journal, context)

            raise

    def _record_selection(
        self,
        selection: SelectionResult,
        tree_context: Dict[str, Any],
        rlm_response: str,
        context: Dict[str, Any],
    ) -> None:
        """Record a selection decision for later analysis."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "selection_index": len(self.selection_history),
            # Tree state at decision time
            "tree_summary": tree_context.get("summary", {}),
            "leaf_steps": tree_context.get("leaf_steps", []),
            # RLM decision
            "rlm_response": rlm_response,
            "selected_nodes": [
                {"step": sn.step, "priority": sn.priority, "reason": sn.reason}
                for sn in selection.selected_nodes
            ],
            "operator": selection.operator,
            "reasoning": selection.reasoning,
            "tree_insights": selection.tree_insights,
            # Context passed to selector
            "num_samples": context.get("num_samples", {}),
            "temperature": context.get("temperature"),
            "crossover_prob": context.get("crossover_prob"),
        }
        self.selection_history.append(record)

    def record_outcome(
        self,
        child_step: int,
        child_metric: Optional[float],
        child_is_buggy: bool,
    ) -> None:
        """
        Record the outcome of the most recent selection.

        Call this after the child node has been evaluated to link
        the selection decision to its result.
        """
        if not self.selection_history:
            return
        self.selection_history[-1]["outcome"] = {
            "child_step": child_step,
            "child_metric": child_metric,
            "child_is_buggy": child_is_buggy,
        }

    def save_history(self, path: Path) -> None:
        """Save selection history and approach cache."""
        path = Path(path)
        with open(path, "w") as f:
            for record in self.selection_history:
                f.write(json.dumps(record, default=str) + "\n")
        log.info(f"Saved {len(self.selection_history)} selection records to {path}")

        # Save approach cache
        cache_path = path.parent / "approach_cache.json"
        with open(cache_path, "w") as f:
            json.dump(self.approach_cache, f, indent=2)
        log.info(f"Saved {len(self.approach_cache)} approach summaries to {cache_path}")

    def load_history(self, path: Path) -> None:
        """Load selection history and approach cache."""
        path = Path(path)
        if not path.exists():
            log.debug(f"No selection history found at {path}")
            return

        # Load selection history
        self.selection_history = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.selection_history.append(json.loads(line))
        log.info(f"Loaded {len(self.selection_history)} selection records from {path}")

        # Load approach cache
        cache_path = path.parent / "approach_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                self.approach_cache = {int(k): v for k, v in json.load(f).items()}
            log.info(f"Loaded {len(self.approach_cache)} approach summaries from {cache_path}")
