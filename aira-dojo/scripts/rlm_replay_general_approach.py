#!/usr/bin/env python
"""
RLM Replay Test with General Operator Evaluation Approach

Tests the updated RLM selector with:
1. Explicit operator evaluation framework (evaluate all operators before deciding)
2. Lightweight aggregation (top_candidates, pairwise_gaps)
3. operator_evaluation field in output

This is the "general approach" vs the previous "specific rules" approach.
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Journal paths
JOURNAL_PATHS = {
    "spooky-author": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_7ab4eef463e0f2e202eba8d86a34923d99f13f62eea4fd14eb28c1a3/checkpoint/journal.jsonl",
    "stanford-covid": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_d6c58b6f6a87be1583935437f450f7d6632f0c9576e0310c3074daca/checkpoint/journal.jsonl",
    "tabular-playground": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_913a0638c077fd805a83f1054fa02c0f9f0b717334f04799d5affd3b/checkpoint/journal.jsonl",
    "essay-scoring": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_983ba5560499f1cd9d463354a0b801973c0d8c1ffe4db6eab1c8d0a4/checkpoint/journal.jsonl",
    "dog-breed": "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_cdb7ea2f8ba4b8c2063bbb0d770d322c8ba233ca5ffa636108144d48/checkpoint/journal.jsonl",
}

# Critical decision points
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

METRIC_DIRECTIONS = {
    "spooky-author": False,
    "stanford-covid": True,
    "tabular-playground": True,
    "essay-scoring": True,
    "dog-breed": True,
}


@dataclass
class TestCase:
    task: str
    step: int
    description: str
    problem_type: str
    lower_is_better: bool


@dataclass
class TestResult:
    test_case: TestCase
    replay_result: Optional[ReplayResult]
    tree_context: Dict[str, Any]
    error: Optional[str] = None


def analyze_tree_state(tree_context: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze tree state for reporting."""
    nodes = tree_context.get("nodes", [])
    leaf_steps = set(tree_context.get("leaf_steps", []))
    metric_max = tree_context.get("metric_maximize", True)

    # Get metric stats
    valid_nodes = [n for n in nodes if n.get("metric") is not None and not n.get("is_buggy")]
    if valid_nodes:
        best_fn = max if metric_max else min
        best_node = best_fn(valid_nodes, key=lambda x: x["metric"])
        best_metric = best_node["metric"]
        best_step = best_node["step"]
    else:
        best_metric = None
        best_step = None

    # Get top_candidates and pairwise_gaps from context
    top_candidates = tree_context.get("top_candidates", [])
    pairwise_gaps = tree_context.get("pairwise_gaps", [])

    return {
        "num_nodes": len(nodes),
        "num_leaves": len(leaf_steps),
        "metric_maximize": metric_max,
        "best_metric": best_metric,
        "best_step": best_step,
        "top_candidates_count": len(top_candidates),
        "pairwise_gaps_count": len(pairwise_gaps),
        "smallest_gap": pairwise_gaps[0]["gap_pct"] if pairwise_gaps else None,
    }


def check_has_operator_evaluation(raw_response: str) -> bool:
    """Check if response includes operator_evaluation field."""
    return "operator_evaluation" in raw_response


def run_test_case(test_case: TestCase, cfg: RLMSelectorConfig) -> TestResult:
    """Run a single replay test case."""
    log.info(f"\n{'='*60}")
    log.info(f"Testing: {test_case.task} @ step {test_case.step}")
    log.info(f"Problem: {test_case.description}")
    log.info(f"{'='*60}")

    journal_path = JOURNAL_PATHS[test_case.task]

    try:
        journal = load_journal_from_jsonl(Path(journal_path))
        max_step = max(n.step for n in journal.nodes)

        if test_case.step > max_step:
            return TestResult(
                test_case=test_case,
                replay_result=None,
                tree_context={},
                error=f"Step {test_case.step} > max step {max_step}"
            )

        snapshot = snapshot_at_step(journal, test_case.step)
        context = build_replay_context(
            snapshot,
            num_samples=1,
            full_context=True,
            max_nodes=100,
            lower_is_better=test_case.lower_is_better,
        )

        # Log aggregation info
        top_candidates = context.tree_context.get("top_candidates", [])
        pairwise_gaps = context.tree_context.get("pairwise_gaps", [])
        log.info(f"Aggregation: {len(top_candidates)} top_candidates, {len(pairwise_gaps)} pairwise_gaps")
        if pairwise_gaps:
            log.info(f"Smallest gap: {pairwise_gaps[0]}")

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
            has_eval = check_has_operator_evaluation(result.raw_response or "")
            log.info(f"Has operator_evaluation: {has_eval}")

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


def generate_report(results: List[TestResult], exp_name: str) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# RLM Replay: General Operator Evaluation Approach")
    lines.append("")
    lines.append("Testing the updated RLM selector with:")
    lines.append("- **Explicit operator evaluation** - RLM must evaluate all operators before deciding")
    lines.append("- **Lightweight aggregation** - `top_candidates` and `pairwise_gaps` in context")
    lines.append("- **`operator_evaluation` field** - Self-documenting decision process")
    lines.append("")
    lines.append(f"**Experiment:** `{exp_name}`")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary statistics
    total = len(results)
    errors = sum(1 for r in results if r.error)
    crossover_count = sum(1 for r in results if r.replay_result and r.replay_result.operator == "crossover")
    has_eval_count = sum(1 for r in results if r.replay_result and check_has_operator_evaluation(r.replay_result.raw_response or ""))

    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total test cases | {total} |")
    lines.append(f"| Errors | {errors} |")
    lines.append(f"| **Crossover triggered** | **{crossover_count}/{total-errors}** |")
    lines.append(f"| Has operator_evaluation | {has_eval_count}/{total-errors} |")
    lines.append("")

    # Detailed results table
    lines.append("## Results by Test Case")
    lines.append("")
    lines.append("| Task | Step | Problem | Selected | Operator | Gap% | Has Eval |")
    lines.append("|------|------|---------|----------|----------|------|----------|")

    for r in results:
        tc = r.test_case
        if r.error:
            lines.append(f"| {tc.task} | {tc.step} | {tc.problem_type} | ERROR | - | - | - |")
        elif r.replay_result:
            rr = r.replay_result
            selected = ",".join(map(str, rr.selected_steps)) if rr.selected_steps else "None"
            tree_analysis = analyze_tree_state(r.tree_context)
            gap = tree_analysis["smallest_gap"]
            gap_str = f"{gap}%" if gap is not None else "-"
            has_eval = "✓" if check_has_operator_evaluation(rr.raw_response or "") else "✗"
            op_marker = f"**{rr.operator}**" if rr.operator == "crossover" else rr.operator
            lines.append(f"| {tc.task} | {tc.step} | {tc.problem_type} | {selected} | {op_marker} | {gap_str} | {has_eval} |")

    lines.append("")

    # Crossover opportunities analysis
    crossover_cases = [r for r in results if r.test_case.problem_type in ["crossover_opportunity", "crossover_needed"]]
    if crossover_cases:
        lines.append("## Crossover Opportunity Analysis")
        lines.append("")
        lines.append("| Task | Step | Ideal Pair | Gap% | Triggered? | Reasoning |")
        lines.append("|------|------|------------|------|------------|-----------|")

        for r in crossover_cases:
            tc = r.test_case
            if r.error or not r.replay_result:
                continue
            rr = r.replay_result
            tree_analysis = analyze_tree_state(r.tree_context)
            pairwise_gaps = r.tree_context.get("pairwise_gaps", [])

            if pairwise_gaps:
                best_pair = pairwise_gaps[0]
                pair_str = f"{best_pair['pair'][0]}/{best_pair['pair'][1]}"
                gap = best_pair["gap_pct"]
            else:
                pair_str = "-"
                gap = "-"

            triggered = "✓" if rr.operator == "crossover" else "✗"
            reason = rr.reasoning[:50] + "..." if rr.reasoning else "-"
            lines.append(f"| {tc.task} | {tc.step} | {pair_str} | {gap}% | {triggered} | {reason} |")

        lines.append("")

    # Detailed responses
    lines.append("## Detailed Responses")
    lines.append("")

    for r in results:
        tc = r.test_case
        lines.append(f"### {tc.task} @ step {tc.step}")
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

        lines.append(f"**Tree State:** {tree_analysis['num_nodes']} nodes, {tree_analysis['num_leaves']} leaves")
        lines.append(f"**Best:** step {tree_analysis['best_step']} ({tree_analysis['best_metric']:.4f})" if tree_analysis['best_metric'] else "**Best:** N/A")
        lines.append(f"**Aggregation:** {tree_analysis['top_candidates_count']} top_candidates, smallest gap={tree_analysis['smallest_gap']}%")
        lines.append("")

        lines.append(f"**Decision:** Selected {rr.selected_steps}, operator=`{rr.operator}`")
        lines.append("")

        lines.append("**Reasoning:**")
        lines.append("```")
        lines.append(rr.reasoning[:500] if rr.reasoning else "(no reasoning)")
        lines.append("```")
        lines.append("")

        if rr.raw_response:
            lines.append("<details>")
            lines.append("<summary>Raw Response</summary>")
            lines.append("")
            lines.append("```python")
            lines.append(rr.raw_response[:2000])
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)


def main():
    """Run all replay tests."""
    import os

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
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

    cfg = RLMSelectorConfig(
        backend="gemini",
        model_name="gemini-2.0-flash",
        api_key_env="GEMINI_API_KEY",
        max_iterations=10,
        verbose=False,
        full_context=True,
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

    results = []
    for tc in test_cases:
        result = run_test_case(tc, cfg)
        results.append(result)
        time.sleep(1)

    # Generate report
    exp_name = f"general_approach_{time.strftime('%Y%m%d_%H%M%S')}"
    report = generate_report(results, exp_name)

    # Save to new experiment folder
    exp_dir = Path(__file__).parent.parent / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    report_path = exp_dir / "results.md"
    with open(report_path, "w") as f:
        f.write(report)

    # Also save raw results as JSON
    raw_results = []
    for r in results:
        raw_results.append({
            "task": r.test_case.task,
            "step": r.test_case.step,
            "problem_type": r.test_case.problem_type,
            "selected": r.replay_result.selected_steps if r.replay_result else None,
            "operator": r.replay_result.operator if r.replay_result else None,
            "reasoning": r.replay_result.reasoning if r.replay_result else None,
            "raw_response": r.replay_result.raw_response if r.replay_result else None,
            "error": r.error,
            "pairwise_gaps": r.tree_context.get("pairwise_gaps", []),
            "top_candidates": r.tree_context.get("top_candidates", []),
        })

    json_path = exp_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(raw_results, f, indent=2, default=str)

    log.info(f"\nResults saved to: {exp_dir}")
    log.info(f"Report: {report_path}")
    log.info(f"Raw JSON: {json_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total = len(results)
    errors = sum(1 for r in results if r.error)
    crossover_count = sum(1 for r in results if r.replay_result and r.replay_result.operator == "crossover")
    print(f"Total: {total}, Errors: {errors}, Crossover: {crossover_count}/{total-errors}")
    print("="*60)


if __name__ == "__main__":
    main()
