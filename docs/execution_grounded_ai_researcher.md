# Execution-Grounded Automated AI Researcher

**Paper:** [arXiv:2601.14525](https://arxiv.org/abs/2601.14525) — Si, Yang, Choi, Candes, Yang, Hashimoto (Stanford, Jan 2026)
**Code:** [github.com/NoviScl/Automated-AI-Researcher](https://github.com/NoviScl/Automated-AI-Researcher)

---

## Terminology

| Term | Meaning in this paper |
|------|----------------------|
| **Search epoch** | One full cycle of the evolutionary loop: generate a batch of ideas → implement as code diffs → execute on GPUs → collect results → update database. NOT a training epoch. The entire search runs for 10 search epochs. |
| **Idea** | A natural-language description of a modification to the codebase (e.g. "replace token-level GRPO loss with sequence-level loss"). Paired with a `[Code Changes]` section describing what files/functions to modify. |
| **Batch** | Within one search epoch, ideas are generated in sub-batches of 10 (via separate LLM calls) to stay within context limits. 80 ideas = 8 batches. |
| **Database** | A JSON file accumulating all `(idea_text, scalar_reward)` pairs across all search epochs. This is the only cross-epoch memory. |
| **Dedup cache** | A per-epoch file storing raw idea texts from previous batches within the same search epoch. Appended to the prompt as "avoid similar ideas." Resets each epoch. |
| **Baseline (β)** | The score achieved by running the unmodified codebase (GRPO: 0.49 accuracy, nanoGPT: 3.255 val_loss). Used to filter "positive" ideas for exploitation. |
| **Exploit** | Generate new ideas by showing the LLM only ideas that beat the baseline, asking it to "combine their strengths and refine." |
| **Explore** | Generate new ideas by showing the LLM a random sample of all ideas (including failures), asking it to "generate brand new ideas, avoid known failure patterns." |

---

## System Architecture Diagram

```
 SEARCH EPOCH 0 (Cold Start)                      SEARCH EPOCH t ≥ 1 (Evolutionary)
 ──────────────────────────                       ───────────────────────────────────

 ┌─────────────────────────┐                      ┌──────────────────────────────────┐
 │  IDEATOR (LLM)          │                      │  DATABASE  (database.json)       │
 │                         │                      │                                  │
 │  agent_call_idea        │                      │  Stores per idea:                │
 │  _simple()              │                      │    • NL idea text                │
 │                         │                      │    • scalar reward               │
 │  Input:                 │                      │    • epoch + idea_id             │
 │   • full codebase       │                      │  Does NOT store:                 │
 │     (all .py/.sh        │                      │    • code diffs                  │
 │     with line #s)       │                      │    • training logs/curves        │
 │   • few-shot examples   │                      │    • execution traces            │
 │     in prompt (e.g.     │                      │    • lineage / parentage         │
 │     "token→seq level    │                      │                                  │
 │     loss")              │                      │  ┌────────────────────────────┐  │
 │   • dedup cache: raw NL │                      │  │ idea_7: acc=0.694  ★       │  │
 │     texts from prior    │                      │  │ idea_3: acc=0.651          │  │
 │     batches in THIS     │                      │  │ idea_12: acc=0.583         │  │
 │     search epoch only   │                      │  │ ─── baseline β = 0.49 ─── │  │
 │     (no scores, resets  │                      │  │ idea_0: acc=0.412          │  │
 │     each search epoch)  │                      │  │ idea_9: acc=-999 (failed)  │  │
 │   • NO execution        │                      │  └────────────────────────────┘  │
 │     results — pure      │                      │                                  │
 │     LLM knowledge       │                      │                                  │
 │                         │                      │                                  │
 │  8 batches × 10 ideas   │                      │                                  │
 │  thinking=3000 tok      │                      │                                  │
 │  temp=1.0               │                      │                                  │
 └────────────┬────────────┘                      └──────────┬──────────┬────────────┘
              │                                              │          │
              │ 80 ideas (NL)                     ┌──────────┘          └──────────┐
              ▼                                   ▼                                ▼
                                      ┌──────────────────────┐     ┌──────────────────────┐
                                      │  EXPLOIT (50-80%)    │     │  EXPLORE (20-50%)    │
                                      │                      │     │                      │
                                      │  FILTER: reward > β  │     │  SAMPLE: random 100  │
                                      │  SAMPLE: top-k (10)  │     │  from ALL (incl.     │
                                      │                      │     │  failures)           │
                                      │  ┌────────────────┐  │     │  ┌────────────────┐  │
                                      │  │ LLM PROMPT:    │  │     │  │ LLM PROMPT:    │  │
                                      │  │                │  │     │  │                │  │
                                      │  │ [codebase]     │  │     │  │ [codebase]     │  │
                                      │  │                │  │     │  │                │  │
                                      │  │ Idea: "Use     │  │     │  │ Idea: "Use     │  │
                                      │  │  cosine lr     │  │     │  │  cosine lr..." │  │
                                      │  │  warmup..."    │  │     │  │ Eval Accuracy: │  │
                                      │  │ Eval Accuracy: │  │     │  │  0.651         │  │
                                      │  │  0.651         │  │     │  │                │  │
                                      │  │                │  │     │  │ Idea: "Add     │  │
                                      │  │ Idea: "GRPO    │  │     │  │  KL penalty"   │  │
                                      │  │  + entropy     │  │     │  │ Eval Accuracy: │  │
                                      │  │  bonus..."     │  │     │  │  Failed to     │  │
                                      │  │ Eval Accuracy: │  │     │  │  implement or  │  │
                                      │  │  0.694         │  │     │  │  execute       │  │
                                      │  │                │  │     │  │                │  │
                                      │  │ "combine their │  │     │  │ "generate new  │  │
                                      │  │  strengths,    │  │     │  │  ideas, avoid  │  │
                                      │  │  refine"       │  │     │  │  known failure │  │
                                      │  │                │  │     │  │  patterns"     │  │
                                      │  └────────────────┘  │     │  └────────────────┘  │
                                      │                      │     │                      │
                                      │  temp=1.0            │     │  temp=1.0            │
                                      │  thinking=1500 tok   │     │  thinking=1500 tok   │
                                      └──────────┬───────────┘     └──────────┬───────────┘
                                                 │                            │
                                                 └──────────┬─────────────────┘
                                                            │
              ┌─────────────────────────────────────────────┘
              │ N new ideas (NL text: [Experiment] + [Code Changes])
              │
              │ NOTE: ideas carry NO code from parents — only NL descriptions.
              │ The implementer generates fresh diffs from scratch each time.
              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                         IMPLEMENTER  (CPU)                              │
 │                                                                         │
 │  Input per idea:  NL description only (no parent code/diffs)           │
 │  Context:         full original codebase (not parent's patched version) │
 │                                                                         │
 │  ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌─────────────┐  │
 │  │ Idea (NL) ├───▶│ LLM: gen     ├───▶│ patch -p1 ├───▶│ Patched     │  │
 │  │ + original│    │ unified diff │    │ --dry-run │    │ codebase    │  │
 │  │ codebase  │    │ (10 samples  │    └─────┬─────┘    └─────────────┘  │
 │  └──────────┘    │  in parallel) │     ✗ fail│                          │
 │                  └──────────────┘          │                            │
 │                        ▲                   ▼                            │
 │                        │ retry w/   ┌───────────┐                      │
 │                        │ error msg  │ Self-     │ (up to 2 retries)    │
 │                        └────────────│ revision  │                      │
 │                                     └───────────┘                      │
 │                                                                         │
 │  Success rate: Claude-Opus ~95%, Claude-Sonnet ~84%, GPT-5 ~65%        │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   │ N patched repos (zipped)
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                         SCHEDULER  (CPU)                                │
 │                                                                         │
 │  Upload zips to HuggingFace → Scheduler polls → submits to workers     │
 │  Examines GPU/memory requirements → matches to available hardware      │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   │ jobs dispatched
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                         WORKERS  (GPU cluster)                          │
 │                                                                         │
 │  ┌──────────────────────┐     ┌────────────────────────────┐           │
 │  │ GRPO Environment     │     │ NanoGPT Environment        │           │
 │  │ • Qwen2.5-Math-1.5B  │     │ • GPT-2 (124M params)     │           │
 │  │ • MATH dataset       │     │ • FineWeb 10B tokens       │           │
 │  │ • 1× A100/B200       │     │ • 8× H100/B200            │           │
 │  │ • ~1hr timeout       │     │ • 1hr wall-clock limit     │           │
 │  │ • Metric: accuracy   │     │ • Metric: val_loss         │           │
 │  └──────────────────────┘     └────────────────────────────┘           │
 │                                                                         │
 │  Full logs → Weights & Biases (wandb)                                  │
 │  (loss curves, gradients, eval metrics, stdout, etc.)                  │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   │ full training logs
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │               LOG RETRIEVAL (retrieve_training_logs.py)                 │
 │                                                                         │
 │  • Poll wandb every 20min until ≥30% of jobs complete                  │
 │  • Extract SINGLE SCALAR per idea:                                     │
 │      GRPO   → max(eval/mean_reward)    from wandb run.history()        │
 │      nanoGPT → min(val_loss)           from stdout parsing             │
 │  • DISCARDS: loss curves, intermediate metrics, gradients, logs        │
 │  • Output: ranked_ideas.json = [{"idea_7": 0.694}, ...]               │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   │ (idea_text, scalar) pairs only
                                   ▼
                         ┌───────────────────┐
                         │  update_database  │── deduplicate by (epoch, idea_id)
                         │                   │   rank by scalar metric
                         └─────────┬─────────┘   → database.json
                                   │
                                   ▼
                         NEXT SEARCH EPOCH (loop back to EXPLOIT/EXPLORE)
```

### What Exactly Flows Back to the LLM

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                       FEEDBACK CONTENT                           │
 │                                                                  │
 │  ✅ INCLUDED in LLM prompt          ✗ NOT INCLUDED              │
 │  ──────────────────────────         ───────────────              │
 │  • Idea NL text                     • Code diffs / patches      │
 │  • Single scalar metric             • Training loss curves      │
 │  • "Failed to implement"            • Intermediate checkpoints  │
 │    status for broken ideas          • Execution error traces    │
 │  • Full original codebase           • Lineage (no parentage)    │
 │  • Baseline threshold (β)           • Search epoch # per idea   │
 │  • Dedup cache (NL text only,       • Wandb logs / stdout       │
 │    no scores, current search        • Parent's patched codebase │
 │    epoch's prior batches only)                                  │
 │                                                                  │
 │  Exploit prompt literally says:                                  │
 │    "Idea: <NL text>\nEval Accuracy: 0.694\n"                    │
 │                                                                  │
 │  Explore prompt literally says:                                  │
 │    "Idea: <NL text>\nEval Accuracy: Failed to implement\n"     │
 │    or "Eval Accuracy: 0.412\n"                                  │
 └──────────────────────────────────────────────────────────────────┘

 The LLM must infer WHY an idea worked from the NL description + score alone.
 It never sees the actual code that was generated or the training dynamics.
```

---

## Evolutionary Search Algorithm

```
Algorithm 1: Execution-Guided Evolutionary Search
──────────────────────────────────────────────────
Input:  Batch size N (50-80), search epochs T (10), baseline β, codebase C
Output: Best-performing idea across all search epochs

Note: "search epoch" = one full generate→implement→execute→collect cycle.
      NOT a training epoch. Each idea's GPU job trains for ~1hr independently.

1. SEARCH EPOCH 0 (cold start — no prior execution data):
   ideas ← LLM.generate(N ideas | codebase C)             # agent.py:agent_call_idea_simple()
   rewards ← IMPLEMENT + EXECUTE(ideas)                    # full_pipeline.py loop

2. FOR search epoch t = 1 to T:
   a. Compute exploitation ratio (anneals toward exploitation):
      α(t) = min(0.5 + 0.1 × ⌊t/2⌋, 0.8)                # evolutionary_search.py:532
      │  search epoch 1 → 50% exploit / 50% explore        # balanced
      │  search epoch 3 → 60% exploit / 40% explore        # shifting
      │  search epoch 6+ → 80% exploit / 20% explore       # converging

   b. EXPLOIT (α·N ideas):
      positives ← {i ∈ database | reward(i) > β}           # filter above baseline
      parents ← sample(positives, k=10)                     # random top-k
      new_ideas ← LLM("combine strengths" | parents + C)

   c. EXPLORE ((1-α)·N ideas):
      sample_pool ← sample(database, k=100)                # any ideas (incl. failures)
      new_ideas ← LLM("generate novel ideas" | pool + C)

   d. IMPLEMENT: idea → diff → patch (10 parallel samples, 2 self-revision retries)
   e. EXECUTE: upload → schedule → GPU run (~1hr each) → wandb logs
   f. RETRIEVE: poll wandb until ≥30% complete → rank → update database
   g. GOTO step 2
```

---

## Key Design Decisions (vs AIRA / PiEvolve / CodeEvolve)

| Dimension | This Paper | AIRA | PiEvolve | CodeEvolve |
|-----------|-----------|------|----------|------------|
| **Idea representation** | NL text → unified diff | NL plan → full script | NL + DAG lineage | SEARCH/REPLACE diffs |
| **Search space** | Training algorithm/hyperparams | Full ML pipeline | Full ML pipeline | Algorithm discovery |
| **Variation operator** | LLM recombination (exploit) + LLM novelty (explore) | Draft/Improve/Debug/Crossover operators | Score-weighted merge | MAP-Elites islands + GA |
| **Selection** | Threshold filter (reward > β) | Best-of-island | Priority scoring | Pareto front |
| **Execution env** | Real GPU training (GRPO/nanoGPT) | Containerized Kaggle | 24hr budget | Sandboxed subprocess |
| **Failure handling** | Shown in explore prompts, hidden in exploit | Filtered from context | Preserved in DAG | Kept in archive |
| **Exploit/explore** | Annealing ratio (50→80% exploit) | Fixed crossover prob (50%) | Adaptive priority | Island migration |
| **Feedback signal** | Scalar (accuracy / val_loss) | Scalar (Kaggle score) | Scalar (Kaggle score) | Test score |
| **LLM role** | Both ideator AND implementer | Plan + code in one shot | Plan + code in one shot | Code-only evolution |
| **Scale** | 50-80 ideas/search epoch × 10 search epochs | 5-10 steps × 3 seeds | Budget-bounded | Population-based |

---

## Results Summary

| Environment | Baseline | Best Found | Human SOTA | Model | Search Epoch |
|-------------|----------|------------|------------|-------|-------|
| **GRPO** (post-training accuracy) | 48.0% | **69.4%** | 68.8% (CS336) | Claude-Sonnet-4.5 | 2 |
| **NanoGPT** (time to target loss) | 35.9 min | **19.7 min** | 2.1 min | Claude-Opus-4.5 | 9 |

### Critical Findings

1. **Evo search >> Best-of-N**: With identical budget (80 ideas/search epoch), evo search outperforms pure random sampling from search epoch 1 onward. The LLM effectively leverages trajectory history.

2. **RL fails at discovery**: RL from execution reward improves *average* idea quality but NOT the *upper bound*. Mode collapse: 119/128 ideas converge to 2 trivial strategies by epoch 68. Thinking traces shrink 70% (model learns short = higher execution rate).

3. **Models saturate early**: Claude-Opus shows sustained scaling across 10 search epochs. Sonnet and GPT-5 plateau after search epoch 2-3. Only Opus occasionally exhibits true scaling trends.

4. **Implementation rate is the bottleneck**: Claude-Opus achieves ~95% successful patching. GPT-5 only ~65%. Failed implementations = wasted compute.

5. **Rediscovered recent research**: Models independently generated ideas matching published papers within 3 months — response diversity rewards (Li et al.), causal context compression (Allen-Zhu).

---

## Relevance to QD-Search / AIRA

| This paper's advantage | Gap / limitation | Implication for AIRA |
|------------------------|------------------|---------------------|
| Real execution feedback closes the loop — ideas proven by GPU runs | Fixed to 2 envs (GRPO, nanoGPT) — not general ML tasks | AIRA already has execution loop via grader; could adopt exploit/explore annealing |
| Exploit/explore split with annealing prevents premature convergence | No explicit diversity preservation (MAP-Elites style) | AIRA EVO already shows diversity collapse (3/5 tasks) — annealing alone insufficient |
| Shows RL from execution reward causes mode collapse | Only scalar feedback, no structured analysis | Confirms AIRA's analyze operator approach (structured feedback) is potentially better |
| 95% implementation rate with Claude Opus | Requires separate "implementer" LLM call for diffs | AIRA generates full scripts directly — simpler but lower diversity |
| Database of all trajectories (incl. failures) informs explore | Exploit hides failures | AIRA currently hides failures too (`good_nodes` filter) — paper suggests showing them |
