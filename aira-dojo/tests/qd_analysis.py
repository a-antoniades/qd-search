"""Comprehensive QD analysis: Baseline EVO vs MAP-Elites (no diversity prompts)."""
import json, sys, os
from pathlib import Path
from collections import defaultdict

ROOT = Path("/share/edc/home/antonis/qd-search")
sys.path.insert(0, str(ROOT / "aira-dojo" / "src"))
sys.path.insert(0, str(ROOT))
from qd.features import extract_features, feature_names

BASELINE_DIR = ROOT / "aira-dojo/logs/aira-dojo/user_antonis_issue_QD_STUDY_evo_gdm"
MAPELITES_DIR = ROOT / "experiments/20260225_152603_evo_mapelites_vs_baseline/outputs/aira-dojo/user_antonis_issue_MAPELITES_evo_gdm"

def analyze_run(run_dir):
    journal_path = run_dir / "json" / "JOURNAL.jsonl"
    config_path = run_dir / "dojo_config.json"
    if not journal_path.exists() or not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    task = cfg.get("task", {}).get("name", "?")
    seed = cfg.get("metadata", {}).get("seed", "?")
    raw_nodes = []
    with open(journal_path) as f:
        for line in f:
            if line.strip():
                raw_nodes.append(json.loads(line))
    if not raw_nodes:
        return None

    # Unwrap data envelope if present
    nodes = []
    for rn in raw_nodes:
        if "data" in rn and isinstance(rn["data"], dict):
            nodes.append(rn["data"])
        else:
            nodes.append(rn)

    # Determine metric direction from metric_maximize field in journal nodes
    lower_is_better = False
    for n in nodes:
        mm = n.get("metric_maximize")
        if mm is not None:
            lower_is_better = not mm
            break

    cells = defaultdict(list)
    all_scores = []
    total_valid = 0
    cell_counts = defaultdict(int)
    for n in nodes:
        if n.get("is_buggy", True):
            continue
        metric = n.get("metric")
        if isinstance(metric, dict):
            score = metric.get("value")
        elif isinstance(metric, (int, float)):
            score = metric
        else:
            continue
        if score is None:
            continue
        plan = n.get("plan", "")
        code = n.get("code", "")
        feats = extract_features(plan, code)
        fn = feature_names(feats)
        cell = (fn["model_family"], fn["data_strategy"])
        cells[cell].append(score)
        cell_counts[cell] += 1
        all_scores.append(score)
        total_valid += 1

    if total_valid == 0:
        return None

    n_occupied = len(cells)
    coverage = n_occupied / 30
    if lower_is_better:
        best_per_cell = {c: min(s) for c, s in cells.items()}
        best_overall = min(all_scores)
        top5 = sorted(all_scores)[:5]
    else:
        best_per_cell = {c: max(s) for c, s in cells.items()}
        best_overall = max(all_scores)
        top5 = sorted(all_scores, reverse=True)[:5]
    qd_score = sum(best_per_cell.values())
    dominant_cell = max(cell_counts, key=cell_counts.get)
    dominance = cell_counts[dominant_cell] / total_valid
    top5_mean = sum(top5) / len(top5)
    return {
        "task": task, "seed": seed, "total_nodes": len(nodes), "valid_nodes": total_valid,
        "n_occupied": n_occupied, "coverage": coverage, "qd_score": qd_score,
        "best_score": best_overall, "top5_mean": top5_mean, "dominance": dominance,
        "dominant_cell": dominant_cell, "lower_is_better": lower_is_better,
        "cell_counts": dict(cell_counts),
    }


def main():
    baseline_results = {}
    for d in sorted(BASELINE_DIR.iterdir()):
        if not d.is_dir():
            continue
        r = analyze_run(d)
        if r is None:
            continue
        key = (r["task"], r["seed"])
        if key not in baseline_results or r["valid_nodes"] > baseline_results[key]["valid_nodes"]:
            baseline_results[key] = r

    mapelites_results = {}
    for d in sorted(MAPELITES_DIR.iterdir()):
        if not d.is_dir():
            continue
        r = analyze_run(d)
        if r is None:
            continue
        key = (r["task"], r["seed"])
        if key not in mapelites_results or r["valid_nodes"] > mapelites_results[key]["valid_nodes"]:
            mapelites_results[key] = r

    all_tasks = sorted(set(r["task"] for r in list(baseline_results.values()) + list(mapelites_results.values())))
    all_seeds = sorted(set(r["seed"] for r in list(baseline_results.values()) + list(mapelites_results.values())), key=lambda x: int(x))

    W = 135
    print("=" * W)
    print("COMPREHENSIVE COMPARISON: Baseline EVO vs MAP-Elites (no diversity prompts)")
    print("=" * W)
    print(f"Baseline: {len(baseline_results)} runs, tasks: {sorted(set(r['task'] for r in baseline_results.values()))}")
    print(f"MAP-Elites: {len(mapelites_results)} runs, tasks: {sorted(set(r['task'] for r in mapelites_results.values()))}")

    overlapping = set(baseline_results.keys()) & set(mapelites_results.keys())
    print(f"Overlapping (task,seed) pairs: {len(overlapping)}")

    fmt = "{:<38} {:>4} {:>2} | {:<5} {:>5} {:>5} {:>6} {:>6} {:>10} {:>10} {:>10} | {}"
    hdr = fmt.format("Task", "Seed", "Dt", "Slvr", "Valid", "Cells", "Cov%", "Dom%", "Best", "Top5mu", "QD", "Dominant")

    print("\n" + "-" * W)
    print(hdr)
    print("-" * W)

    deltas = {"coverage": [], "dominance": [], "best": [], "top5": [], "cells": []}

    for task in all_tasks:
        for seed in all_seeds:
            key = (task, seed)
            b = baseline_results.get(key)
            m = mapelites_results.get(key)
            if not b and not m:
                continue
            dt = "lo" if (b or m)["lower_is_better"] else "hi"

            if b:
                dc = f"{b['dominant_cell'][0][:12]}+{b['dominant_cell'][1][:10]}"
                print(fmt.format(task[:38], str(seed), dt, "BASE", b['valid_nodes'], b['n_occupied'],
                    f"{b['coverage']*100:.1f}", f"{b['dominance']*100:.1f}", f"{b['best_score']:.5f}",
                    f"{b['top5_mean']:.5f}", f"{b['qd_score']:.4f}", dc))
            if m:
                dc = f"{m['dominant_cell'][0][:12]}+{m['dominant_cell'][1][:10]}"
                lbl = "" if b else task[:38]
                print(fmt.format(lbl, "" if b else str(seed), "" if b else dt, "ME", m['valid_nodes'], m['n_occupied'],
                    f"{m['coverage']*100:.1f}", f"{m['dominance']*100:.1f}", f"{m['best_score']:.5f}",
                    f"{m['top5_mean']:.5f}", f"{m['qd_score']:.4f}", dc))

            if b and m:
                cov_d = m['coverage'] - b['coverage']
                dom_d = m['dominance'] - b['dominance']
                cells_d = m['n_occupied'] - b['n_occupied']
                if b['lower_is_better']:
                    bw = "ME" if m['best_score'] < b['best_score'] else "BASE"
                    tw = "ME" if m['top5_mean'] < b['top5_mean'] else "BASE"
                else:
                    bw = "ME" if m['best_score'] > b['best_score'] else "BASE"
                    tw = "ME" if m['top5_mean'] > b['top5_mean'] else "BASE"
                deltas["coverage"].append(cov_d)
                deltas["dominance"].append(dom_d)
                deltas["cells"].append(cells_d)
                deltas["best"].append(bw)
                deltas["top5"].append(tw)
                print(fmt.format("", "", "", "D", "", f"{cells_d:+d}",
                    f"{cov_d*100:+.1f}", f"{dom_d*100:+.1f}", f"<-{bw}",
                    f"<-{tw}", "", ""))
            print()

    # Baseline-only
    bonly = set(baseline_results.keys()) - overlapping
    if bonly:
        print("=" * W)
        print("BASELINE-ONLY (no MAP-Elites counterpart)")
        print("-" * W)
        for task in all_tasks:
            for seed in all_seeds:
                key = (task, seed)
                if key not in bonly:
                    continue
                b = baseline_results[key]
                dt = "lo" if b["lower_is_better"] else "hi"
                dc = f"{b['dominant_cell'][0][:12]}+{b['dominant_cell'][1][:10]}"
                print(fmt.format(task[:38], str(seed), dt, "BASE", b['valid_nodes'], b['n_occupied'],
                    f"{b['coverage']*100:.1f}", f"{b['dominance']*100:.1f}", f"{b['best_score']:.5f}",
                    f"{b['top5_mean']:.5f}", f"{b['qd_score']:.4f}", dc))

    # ME-only
    monly = set(mapelites_results.keys()) - overlapping
    if monly:
        print("\n" + "=" * W)
        print("MAP-ELITES-ONLY (no baseline counterpart)")
        print("-" * W)
        for task in all_tasks:
            for seed in all_seeds:
                key = (task, seed)
                if key not in monly:
                    continue
                m = mapelites_results[key]
                dt = "lo" if m["lower_is_better"] else "hi"
                dc = f"{m['dominant_cell'][0][:12]}+{m['dominant_cell'][1][:10]}"
                print(fmt.format(task[:38], str(seed), dt, "ME", m['valid_nodes'], m['n_occupied'],
                    f"{m['coverage']*100:.1f}", f"{m['dominance']*100:.1f}", f"{m['best_score']:.5f}",
                    f"{m['top5_mean']:.5f}", f"{m['qd_score']:.4f}", dc))

    # Summary
    print("\n" + "=" * W)
    print("SUMMARY (overlapping pairs only)")
    print("=" * W)
    n = len(deltas["coverage"])
    if n == 0:
        print("No overlapping pairs to compare!")
        return
    print(f"Pairs compared:       {n}")
    print(f"Avg coverage delta:   {sum(deltas['coverage'])/n*100:+.1f}%  (ME - BASE)")
    print(f"Avg dominance delta:  {sum(deltas['dominance'])/n*100:+.1f}%  (negative = more diverse)")
    print(f"Avg cells delta:      {sum(deltas['cells'])/n:+.1f}")
    me_b = sum(1 for x in deltas["best"] if x == "ME")
    me_t = sum(1 for x in deltas["top5"] if x == "ME")
    print(f"Best score wins:      ME {me_b}/{n}, BASE {n-me_b}/{n}")
    print(f"Top-5 mean wins:      ME {me_t}/{n}, BASE {n-me_t}/{n}")


if __name__ == "__main__":
    main()
