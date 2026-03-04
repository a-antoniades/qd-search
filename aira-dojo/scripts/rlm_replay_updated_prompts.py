#!/usr/bin/env python
"""
RLM Replay Test with Updated QD-Aware Prompts

Tests the updated RLM selector prompts on critical decision points from
the 5 EVO trajectories identified in rlm_replay_comprehensive.md.

Key improvements being tested:
1. metric_maximize flag - fixes metric direction confusion
2. New QD helper functions - get_architecture_distribution(), get_neglected_leaves(), find_crossover_candidates()
3. Structured reasoning process in prompt
4. Explicit guidance on crossover operator
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add aira-dojo to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dojo.config_dataclasses.selector.rlm import RLMSelectorConfig
from dojo.core.solvers.selection.replay import (
    load_journal_from_jsonl,
    snapshot_at_step,
    build_replay_context,
    replay_selection,
    ReplayResult,
)
from dojo.core.solvers.selection.rlm_selector import _create_setup_code
from dojo.core.solvers.selection.tree_serializer import serialize_for_rlm

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Journal paths (largest/most complete runs)
JOURNAL_PATHS = {
    "spooky-author": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_7ab4eef463e0f2e202eba8d86a34923d99f13f62eea4fd14eb28c1a3/checkpoint/journal.jsonl",
    "stanford-covid": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_d6c58b6f6a87be1583935437f450f7d6632f0c9576e0310c3074daca/checkpoint/journal.jsonl",
    "tabular-playground": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_913a0638c077fd805a83f1054fa02c0f9f0b717334f04799d5affd3b/checkpoint/journal.jsonl",
    "essay-scoring": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_983ba5560499f1cd9d463354a0b801973c0d8c1ffe4db6eab1c8d0a4/checkpoint/journal.jsonl",
    "dog-breed": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_cdb7ea2f8ba4b8c2063bbb0d770d322c8ba233ca5ffa636108144d48/checkpoint/journal.jsonl",
}

# Critical decision points from the report
# Format: task -> list of (step, description, problem_being_tested)
CRITICAL_STEPS = {
    "spooky-author": [
        (43, "Best node step 10 (metric 0.748) available but not selected", "metric_direction_confusion"),
        (50, "Step 10 vs step 37 - log loss lower is better", "metric_direction_confusion"),
    ],
    "stanford-covid": [
        (5, "GNN node (step 3) first available", "diversity_underexplored"),
        (10, "GNN still rare (7% of nodes)", "diversity_underexplored"),
        (50, "GNN at 2.1%, step 49 best GNN leaf", "diversity_underexplored"),
    ],
    "tabular-playground": [
        (25, "XGBoost (step 18) vs LightGBM (step 22) both at 0.963", "crossover_opportunity"),
        (35, "XGBoost dominance, LightGBM diversity needed", "crossover_opportunity"),
    ],
    "essay-scoring": [
        (10, "Early stage, GBDT dominance beginning", "crossover_needed"),
        (75, "GBDT 0.817 vs DeBERTa emerging", "crossover_needed"),
        (83, "GBDT 0.817 vs DeBERTa 0.815 - optimal crossover", "crossover_needed"),
    ],
    "dog-breed": [
        (15, "ConvNeXt early (13%), EVA-02 competitive", "convergence_risk"),
        (50, "ConvNeXt 69%, should diversify", "convergence_risk"),
        (70, "ConvNeXt 74% dominance", "convergence_risk"),
    ],
}

# Metric directions (True = higher is better, False = lower is better)
METRIC_DIRECTIONS = {
    "spooky-author": False,  # Log loss - lower is better
    "stanford-covid": True,   # MCRMSE inverted or accuracy
    "tabular-playground": True,  # AUC - higher is better
    "essay-scoring": True,    # QWK - higher is better
    "dog-breed": True,        # Accuracy - higher is better (but metric shown as log loss?)
}


@dataclass
class TestCase:
    """A single replay test case."""
    task: str
    step: int
    description: str
    problem_type: str
    lower_is_better: bool


@dataclass
class TestResult:
    """Result of a replay test."""
    test_case: TestCase
    replay_result: Optional[ReplayResult]
    tree_context: Dict[str, Any]
    error: Optional[str] = None


def analyze_tree_state(tree_context: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze tree state for reporting."""
    nodes = tree_context.get("nodes", [])
    leaf_steps = set(tree_context.get("leaf_steps", []))

    # Count architectures in leaves
    arch_counts = {}
    for n in nodes:
        if n["step"] in leaf_steps:
            for arch in n.get("architectures", []):
                arch_counts[arch] = arch_counts.get(arch, 0) + 1

    total_archs = sum(arch_counts.values()) or 1
    dominant_arch = max(arch_counts.items(), key=lambda x: x[1])[0] if arch_counts else None
    dominant_pct = max(arch_counts.values()) / total_archs * 100 if arch_counts else 0

    # Get metric stats
    metric_max = tree_context.get("metric_maximize", True)
    valid_nodes = [n for n in nodes if n.get("metric") is not None and not n.get("is_buggy")]
    if valid_nodes:
        best_fn = max if metric_max else min
        best_node = best_fn(valid_nodes, key=lambda x: x["metric"])
        best_metric = best_node["metric"]
        best_step = best_node["step"]
    else:
        best_metric = None
        best_step = None

    return {
        "num_nodes": len(nodes),
        "num_leaves": len(leaf_steps),
        "metric_maximize": metric_max,
        "best_metric": best_metric,
        "best_step": best_step,
        "dominant_arch": dominant_arch,
        "dominant_pct": round(dominant_pct, 1),
        "arch_distribution": dict(sorted(arch_counts.items(), key=lambda x: -x[1])[:5]),
    }


def run_test_case(test_case: TestCase, cfg: RLMSelectorConfig) -> TestResult:
    """Run a single replay test case."""
    log.info(f"\n{'='*60}")
    log.info(f"Testing: {test_case.task} @ step {test_case.step}")
    log.info(f"Problem: {test_case.description}")
    log.info(f"lower_is_better: {test_case.lower_is_better}")
    log.info(f"{'='*60}")

    journal_path = JOURNAL_PATHS[test_case.task]

    try:
        # Load journal
        journal = load_journal_from_jsonl(Path(journal_path))
        log.info(f"Loaded journal with {len(journal.nodes)} nodes")

        # Check if step exists
        max_step = max(n.step for n in journal.nodes)
        if test_case.step > max_step:
            return TestResult(
                test_case=test_case,
                replay_result=None,
                tree_context={},
                error=f"Step {test_case.step} > max step {max_step} in journal"
            )

        # Snapshot at step
        snapshot = snapshot_at_step(journal, test_case.step)
        log.info(f"Snapshot at step {test_case.step}: {snapshot.num_nodes} nodes, {len(snapshot.leaf_steps)} leaves")

        # Build replay context with lower_is_better and full_context for plan analysis
        context = build_replay_context(
            snapshot,
            num_samples=1,
            full_context=True,  # Include plans for approach diversity analysis
            max_nodes=100,
            lower_is_better=test_case.lower_is_better,
        )

        # Verify metric_maximize is set correctly
        assert context.tree_context.get("metric_maximize") == (not test_case.lower_is_better), \
            f"metric_maximize mismatch: expected {not test_case.lower_is_better}, got {context.tree_context.get('metric_maximize')}"

        log.info(f"Tree context: metric_maximize={context.tree_context.get('metric_maximize')}")

        # Run replay
        result = replay_selection(
            context,
            cfg,
            lower_is_better=test_case.lower_is_better,
        )

        if result.error:
            log.warning(f"Replay error: {result.error}")
        else:
            log.info(f"Selected: {result.selected_steps}")
            log.info(f"Operator: {result.operator}")
            log.info(f"Reasoning: {result.reasoning[:200]}...")

        return TestResult(
            test_case=test_case,
            replay_result=result,
            tree_context=context.tree_context,
        )

    except Exception as e:
        log.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return TestResult(
            test_case=test_case,
            replay_result=None,
            tree_context={},
            error=str(e)
        )


def generate_report(results: List[TestResult]) -> str:
    """Generate markdown report from test results."""
    lines = []
    lines.append("# RLM Replay Test Results: Updated QD-Aware Prompts")
    lines.append("")
    lines.append("Testing the updated RLM selector with:")
    lines.append("- `metric_maximize` flag in context")
    lines.append("- QD helper functions: `get_architecture_distribution()`, `get_neglected_leaves()`, `find_crossover_candidates()`")
    lines.append("- Structured reasoning process in prompt")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Task | Step | Problem | Selected | Operator | Fix Status | Key Insight |")
    lines.append("|------|------|---------|----------|----------|------------|-------------|")

    for r in results:
        tc = r.test_case
        if r.error:
            lines.append(f"| {tc.task} | {tc.step} | {tc.problem_type} | ERROR | - | - | {r.error[:50]} |")
        elif r.replay_result:
            rr = r.replay_result
            selected = ",".join(map(str, rr.selected_steps)) if rr.selected_steps else "None"
            # Determine fix status
            fix_status = assess_fix(r)
            insight = extract_key_insight(rr.reasoning)
            lines.append(f"| {tc.task} | {tc.step} | {tc.problem_type} | {selected} | {rr.operator} | {fix_status} | {insight} |")

    lines.append("")

    # Detailed results by task
    current_task = None
    for r in results:
        if r.test_case.task != current_task:
            current_task = r.test_case.task
            lines.append(f"## {current_task.title()}")
            lines.append("")

            # Show metric direction
            lib = METRIC_DIRECTIONS.get(current_task, True)
            direction = "lower is better" if not lib else "higher is better"
            lines.append(f"**Metric direction:** {direction}")
            lines.append("")

        tc = r.test_case
        lines.append(f"### Step {tc.step}: {tc.description}")
        lines.append("")

        if r.error:
            lines.append(f"**Error:** {r.error}")
            lines.append("")
            continue

        if not r.replay_result:
            lines.append("**No result**")
            lines.append("")
            continue

        rr = r.replay_result
        tree_analysis = analyze_tree_state(r.tree_context)

        lines.append("**Tree State:**")
        lines.append(f"- Nodes: {tree_analysis['num_nodes']}, Leaves: {tree_analysis['num_leaves']}")
        lines.append(f"- metric_maximize: {tree_analysis['metric_maximize']}")
        lines.append(f"- Best: step {tree_analysis['best_step']} ({tree_analysis['best_metric']:.4f})" if tree_analysis['best_metric'] else "- Best: N/A")
        lines.append(f"- Dominant arch: {tree_analysis['dominant_arch']} ({tree_analysis['dominant_pct']}%)")
        lines.append(f"- Distribution: {tree_analysis['arch_distribution']}")
        lines.append("")

        lines.append("**RLM Decision:**")
        lines.append(f"- Selected: {rr.selected_steps}")
        lines.append(f"- Operator: `{rr.operator}`")
        lines.append(f"- Elapsed: {rr.elapsed_s:.1f}s")
        lines.append("")

        lines.append("**Reasoning:**")
        lines.append("```")
        lines.append(rr.reasoning[:500] if rr.reasoning else "(no reasoning)")
        lines.append("```")
        lines.append("")

        if rr.raw_response:
            lines.append("<details>")
            lines.append("<summary>Raw Response (click to expand)</summary>")
            lines.append("")
            lines.append("```python")
            lines.append(rr.raw_response[:1500])
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    # Conclusions
    lines.append("## Conclusions")
    lines.append("")
    lines.append("### What the Updated Prompts Fix")
    lines.append("")

    # Analyze results
    metric_fix_count = sum(1 for r in results
                          if r.test_case.problem_type == "metric_direction_confusion"
                          and not r.error
                          and assess_fix(r) in ["FIXED", "PARTIAL"])
    metric_total = sum(1 for r in results if r.test_case.problem_type == "metric_direction_confusion")

    diversity_fix_count = sum(1 for r in results
                             if r.test_case.problem_type == "diversity_underexplored"
                             and not r.error
                             and assess_fix(r) in ["FIXED", "PARTIAL"])
    diversity_total = sum(1 for r in results if r.test_case.problem_type == "diversity_underexplored")

    crossover_fix_count = sum(1 for r in results
                             if r.test_case.problem_type in ["crossover_opportunity", "crossover_needed"]
                             and not r.error
                             and r.replay_result and r.replay_result.operator == "crossover")
    crossover_total = sum(1 for r in results if r.test_case.problem_type in ["crossover_opportunity", "crossover_needed"])

    lines.append(f"| Issue | Fixed/Total | Status |")
    lines.append(f"|-------|-------------|--------|")
    lines.append(f"| Metric direction confusion | {metric_fix_count}/{metric_total} | {'FIXED' if metric_fix_count == metric_total else 'PARTIAL'} |")
    lines.append(f"| Diversity/underexplored | {diversity_fix_count}/{diversity_total} | {'FIXED' if diversity_fix_count == diversity_total else 'PARTIAL'} |")
    lines.append(f"| Crossover suggestions | {crossover_fix_count}/{crossover_total} | {'FIXED' if crossover_fix_count > 0 else 'NOT FIXED'} |")
    lines.append("")

    return "\n".join(lines)


def assess_fix(r: TestResult) -> str:
    """Assess if the test case problem was fixed."""
    if r.error or not r.replay_result:
        return "ERROR"

    tc = r.test_case
    rr = r.replay_result
    tree_analysis = analyze_tree_state(r.tree_context)

    if tc.problem_type == "metric_direction_confusion":
        # Check if selected node is actually the best according to correct metric direction
        if rr.selected_steps and tree_analysis["best_step"] in rr.selected_steps:
            return "FIXED"
        return "NOT FIXED"

    if tc.problem_type == "diversity_underexplored":
        # Check if selected a minority architecture
        selected_step = rr.selected_steps[0] if rr.selected_steps else None
        if selected_step:
            for n in r.tree_context.get("nodes", []):
                if n["step"] == selected_step:
                    archs = n.get("architectures", [])
                    # Check if any of the architectures are minority (<30% of distribution)
                    arch_dist = tree_analysis["arch_distribution"]
                    total = sum(arch_dist.values()) or 1
                    for arch in archs:
                        if arch in arch_dist and arch_dist[arch] / total < 0.3:
                            return "FIXED"
        return "PARTIAL"

    if tc.problem_type in ["crossover_opportunity", "crossover_needed"]:
        if rr.operator == "crossover":
            return "FIXED"
        # Partial if it at least selected diversity
        return "PARTIAL"

    if tc.problem_type == "convergence_risk":
        # Check if avoided the dominant architecture
        dominant = tree_analysis["dominant_arch"]
        selected_step = rr.selected_steps[0] if rr.selected_steps else None
        if selected_step:
            for n in r.tree_context.get("nodes", []):
                if n["step"] == selected_step:
                    if dominant not in n.get("architectures", []):
                        return "FIXED"
        return "NOT FIXED"

    return "UNKNOWN"


def extract_key_insight(reasoning: str) -> str:
    """Extract key insight from reasoning for summary table."""
    if not reasoning:
        return "(no reasoning)"

    # Look for key phrases
    keywords = ["metric_maximize", "diversity", "crossover", "underexplored",
                "architecture", "convergence", "neglected", "unique"]

    for kw in keywords:
        if kw.lower() in reasoning.lower():
            # Extract surrounding context
            idx = reasoning.lower().find(kw.lower())
            start = max(0, idx - 20)
            end = min(len(reasoning), idx + 50)
            snippet = reasoning[start:end].replace("\n", " ").strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(reasoning):
                snippet = snippet + "..."
            return snippet[:60]

    return reasoning[:50] + "..."


def main():
    """Run all replay tests."""
    import os

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Try loading from .env
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"')
                        os.environ["GEMINI_API_KEY"] = api_key
                        break

    if not api_key:
        log.error("GEMINI_API_KEY not found")
        return

    # RLM config - full_context=True to include plans for approach analysis
    cfg = RLMSelectorConfig(
        backend="gemini",
        model_name="gemini-2.0-flash",
        api_key_env="GEMINI_API_KEY",
        max_iterations=10,
        verbose=False,
        full_context=True,  # IMPORTANT: Include plans for approach diversity analysis
        max_nodes=100,
        fallback_to_fitness=True,
    )

    # Build test cases
    test_cases = []
    for task, steps in CRITICAL_STEPS.items():
        lower_is_better = not METRIC_DIRECTIONS.get(task, True)
        for step, description, problem_type in steps:
            test_cases.append(TestCase(
                task=task,
                step=step,
                description=description,
                problem_type=problem_type,
                lower_is_better=lower_is_better,
            ))

    log.info(f"Running {len(test_cases)} test cases...")

    # Run tests
    results = []
    for tc in test_cases:
        result = run_test_case(tc, cfg)
        results.append(result)
        time.sleep(1)  # Rate limiting

    # Generate report
    report = generate_report(results)

    # Save report
    report_path = Path(__file__).parent.parent / "experiments" / "rlm_replay_updated_prompts_results.md"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    log.info(f"\nReport saved to: {report_path}")
    print("\n" + "="*60)
    print("REPORT PREVIEW")
    print("="*60)
    print(report[:3000])


if __name__ == "__main__":
    main()
