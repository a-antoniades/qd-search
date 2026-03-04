#!/usr/bin/env python
"""Quick test of RLM with 3 key crossover cases."""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dojo.config_dataclasses.selector.rlm import RLMSelectorConfig
from dojo.core.solvers.selection.replay import (
    load_journal_from_jsonl,
    snapshot_at_step,
    build_replay_context,
    replay_selection,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Key test cases
TEST_CASES = [
    ("tabular-playground", 25, True, "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_913a0638c077fd805a83f1054fa02c0f9f0b717334f04799d5affd3b/checkpoint/journal.jsonl"),
    ("essay-scoring", 83, True, "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_983ba5560499f1cd9d463354a0b801973c0d8c1ffe4db6eab1c8d0a4/checkpoint/journal.jsonl"),
    ("spooky-author", 50, False, "/share/edc/home/antonis/qd-search/aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm/user_antonis_issue_QD_STUDY_evo_gdm_seed_1_id_7ab4eef463e0f2e202eba8d86a34923d99f13f62eea4fd14eb28c1a3/checkpoint/journal.jsonl"),
]

def main():
    import os
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
    
    results = []
    for task, step, metric_max, path in TEST_CASES:
        log.info(f"\n{'='*60}")
        log.info(f"Testing: {task} @ step {step}")
        log.info(f"{'='*60}")
        
        journal = load_journal_from_jsonl(Path(path))
        snapshot = snapshot_at_step(journal, step)
        context = build_replay_context(
            snapshot,
            num_samples=1,
            full_context=True,
            max_nodes=100,
            lower_is_better=not metric_max,
        )
        
        # Log aggregation
        top_cands = context.tree_context.get("top_candidates", [])
        gaps = context.tree_context.get("pairwise_gaps", [])
        log.info(f"Top candidates: {len(top_cands)}")
        log.info(f"Pairwise gaps: {gaps[:3]}")
        
        result = replay_selection(context, cfg, lower_is_better=not metric_max)
        
        if result.error:
            log.warning(f"Error: {result.error}")
            results.append((task, step, "ERROR", result.error))
        else:
            log.info(f"Selected: {result.selected_steps}")
            log.info(f"Operator: {result.operator}")
            log.info(f"Reasoning: {result.reasoning[:200]}...")
            results.append((task, step, result.operator, result.selected_steps))
        
        time.sleep(5)  # More conservative rate limiting
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"| Task | Step | Operator | Selected |")
    print(f"|------|------|----------|----------|")
    for task, step, op, sel in results:
        print(f"| {task} | {step} | {op} | {sel} |")
    
    crossover_count = sum(1 for _, _, op, _ in results if op == "crossover")
    print(f"\nCrossover triggered: {crossover_count}/{len(results)}")

if __name__ == "__main__":
    main()
