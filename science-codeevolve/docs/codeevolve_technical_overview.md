# CodeEvolve: Technical Overview

## What It Does

CodeEvolve evolves code using LLMs as mutation operators.
An LLM reads a parent solution, generates a SEARCH/REPLACE diff, and the child is evaluated.
Repeat. The best solutions survive. The system runs multiple populations ("islands") in parallel.

---

## The Loop (per epoch)

```
 ┌─ 1. EXPLORE or EXPLOIT? ─── coin flip against exploration_rate
 │       explore → random parent, no chat history, cheap LLM
 │       exploit → fitness-selected parent, full chat history, strong LLM
 │
 ├─ 2. SELECT PARENT ────────── roulette (by rank), tournament, random, or best
 │       also selects "inspiration" programs shown to LLM as context
 │
 ├─ 3. META-PROMPT (optional)── evolve the system prompt itself via LLM diff
 │
 ├─ 4. GENERATE ─────────────── LLM produces SEARCH/REPLACE diff on parent code
 │       only modifies code between # EVOLVE-BLOCK-START/END markers
 │
 ├─ 5. EVALUATE ─────────────── run child in sandbox (timeout + memory limits)
 │       extract fitness from JSON metrics
 │       add to population database
 │
 └─ 6. MIGRATE (periodic) ──── top solutions sent to neighboring islands
```

Entry point: `cli.py:main()` → spawns N island processes → each runs `evolution.py:codeevolve_loop()`.

---

## Quality-Diversity: MAP-Elites

Standard EA keeps a fixed-size population, replacing the worst when full.
MAP-Elites instead maintains a **feature map** where each cell holds the best solution for that region of feature space.

**Why it matters**: Two solutions with the same fitness but different characteristics (e.g. fast-but-approximate vs slow-but-precise) both survive. This diversity feeds better exploration.

### Two Variants

| Variant | How it partitions | Config |
|---------|-------------------|--------|
| **Grid** | Fixed bins per feature dimension | `elite_map_type: grid`, requires `num_bins` |
| **CVT** | Centroidal Voronoi Tessellation | `elite_map_type: cvt`, requires `num_centroids` |

### Where it hooks in

1. **Selection** (`database.py:382`): candidates sampled from elite archive instead of alive pool
2. **Archiving** (`database.py:417`): new solution placed in its feature cell, replaces occupant only if fitter

### Ablation result (CirclePackingSquare n=32, Qwen)

```
CVT MAP-Elites  → ~2.8-3.0  ✅ beats AlphaEvolve benchmark
Grid MAP-Elites → ~2.6-2.7  ~  approaches it
No MAP-Elites   → ~2.3-2.4  ❌ never reaches it
```

CVT outperforms Grid likely because continuous Voronoi partitioning captures feature space structure better than fixed bins.

---

## Exploration vs Exploitation

Each epoch flips a coin: `random() <= exploration_rate` → explore, else exploit.

| | Exploration | Exploitation |
|-|-------------|--------------|
| **Parent** | Random from pool | Roulette/tournament selection |
| **LLM** | `exploration_ensemble` (cheap) | `exploitation_ensemble` (strong) |
| **Chat depth** | 0 (single turn) | `max_chat_depth` (multi-turn lineage) |
| **Inspirations** | None | Sampled from pool |
| **Meta-prompt** | Yes (if enabled) | No |

### Exploration Rate Schedulers (`scheduler.py`)

| Scheduler | Behavior |
|-----------|----------|
| **ExponentialDecay** | `rate(t) = rate₀ × decay^t` — monotonic decrease |
| **Plateau** | Decrease on improvement, increase after N stagnant epochs |
| **Cosine** | Oscillates between min/max over a fixed period |

Default in experiments: **PlateauScheduler** (min=0.2, max=0.5, plateau_threshold=5).

---

## Island Model

Multiple populations evolve independently. Periodically, top solutions migrate between neighbors.

```
Island 0 ──→ Island 1 ──→ Island 2 ──→ ... ──→ Island N ──→ Island 0
```

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `num_islands` | 5-10 | Number of parallel populations |
| `migration_topology` | `ring` | Who sends to whom |
| `migration_interval` | 20-30 | Epochs between migrations |
| `migration_rate` | 0.1 | Fraction of population that migrates |

Topologies: `directed_ring`, `ring`, `complete`, `star`, `empty`.

Islands sync at barriers for migration and checkpointing. Early stopping is global: if no island improves for N consecutive epochs, all stop.

---

## LLM Ensembles

Each ensemble holds one or more models with weights. On each call, one is randomly selected (weighted).

```yaml
EXPLOITATION_ENSEMBLE:
  - {model_name: 'GEMINI-2.5-FLASH', weight: 0.60}  # 60% of calls
  - {model_name: 'GEMINI-2.5-PRO',   weight: 0.40}  # 40% of calls
```

Separate ensembles for exploration (cheap/fast) and exploitation (strong/expensive).

---

## Selection Policies (`database.py`)

| Policy | How it picks | Use case |
|--------|--------------|----------|
| `random` | Uniform random | Exploration mode |
| `roulette` | P ∝ fitness (or P ∝ 1/(1+rank) if `roulette_by_rank`) | Default exploitation |
| `tournament` | Sample k, return best | Alternative exploitation |
| `best` | Always pick highest fitness | Greedy |

Default in experiments: **roulette by rank**.

---

## Meta-Prompting

Optional co-evolution of the system prompt. An auxiliary LLM generates a SEARCH/REPLACE diff to modify the prompt between `# PROMPT-BLOCK-START/END` markers. Only active during exploration. The prompt population is maintained separately from the solution population.

---

## Default Config (from paper experiments)

```yaml
selection_policy: roulette          # rank-based
exploration_rate: 0.2               # 20% explore, 80% exploit
scheduler: PlateauScheduler         # adaptive rate
num_islands: 5-10
migration_topology: ring
migration_interval: 20-30
num_inspirations: 2
max_chat_depth: 5
meta_prompting: true
use_map_elites: true                # CVT, 30 centroids
```

---

## File Map

| File | Role |
|------|------|
| `cli.py` | Entry point, spawns islands |
| `evolution.py` | Main loop, parent selection, generation, evaluation |
| `database.py` | Population management, selection policies, MAP-Elites |
| `islands.py` | Migration, topology, early stopping |
| `lm.py` | LLM API wrapper, ensemble weighted selection |
| `evaluator.py` | Sandboxed execution with resource limits |
| `scheduler.py` | Exploration rate schedulers |
| `prompt/sampler.py` | Chat construction, meta-prompting |
| `utils/parsing_utils.py` | SEARCH/REPLACE diff parsing and application |
| `utils/ckpt_utils.py` | Checkpoint save/load |
