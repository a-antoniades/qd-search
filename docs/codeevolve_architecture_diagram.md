# CodeEvolve Architecture & Flow Diagram

## Overview

CodeEvolve is a distributed evolutionary algorithm that uses LLMs to evolve code solutions.
It runs multiple "islands" in parallel, each maintaining a population of solutions that
periodically migrate between islands.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLI ENTRY POINT                                    │
│                                  cli.py:main() L370                                  │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  1. Parse args & validate environment                                                │
│  2. Load config YAML                                                                 │
│  3. Setup shared memory (GlobalData)                        cli.py:create_global_data│
│  4. Create migration topology (ring/complete/star/etc)   cli.py:setup_island_topology│
│  5. Spawn island processes                              cli.py:spawn_island_processes│
│                                                                                      │
└─────────────────────────────────────────┬────────────────────────────────────────────┘
                                          │
                                          │ multiprocessing.Process() for each island
                                          ▼
┌────────────────────────────────────────────────────`──────────────────────────────────┐
│                             ISLAND PROCESSES (parallel)                              │
│                                                                                      │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          ┌─────────────┐     │
│   │  Island 0   │    │  Island 1   │    │  Island 2   │          │  Island N   │     │
│   │             │◄───│             │◄───│             │◄── ... ──│             │     │
│   │  codeevolve │───►│  codeevolve │───►│  codeevolve │─── ... ─►│  codeevolve │     │
│   └─────────────┘    └─────────────┘    └─────────────┘          └─────────────┘     │ 
│          │                  │                  │                        │            │
│          └──────────────────┴──────────────────┴────────────────────────┘            │
│                            Migration (periodic, via pipes)                           │
│                            islands.py:sync_migrate() L71                             │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Per-Island Evolution Flow

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              ISLAND INITIALIZATION                                   │
│                            evolution.py:codeevolve() L811                            │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐  │
│  │      LM Ensembles      │  │    Program Database    │  │       Evaluator        │  │
│  │    lm.py:LMEnsemble    │  │    database.py L282    │  │    evaluator.py L47    │  │
│  │                        │  │                        │  │                        │  │
│  │ • exploration_ensemble │  │ • programs dict        │  │ • Sandboxed exec       │  │
│  │ • exploitation_ensemble│  │ • elite_map (optional) │  │ • Timeout/memory       │  │
│  │                        │  │ • selection methods    │  │ • Metrics extraction   │  │
│  └────────────────────────┘  └───────────┬────────────┘  └────────────────────────┘  │
│                                          │                                           │
│                                          │ if use_map_elites:                        │
│                                          ▼                                           │
│                              ┌────────────────────────┐                              │
│                              │       Elite Map        │                              │
│                              │      database.py       │                              │
│                              │                        │                              │
│                              │  • GridEliteMap  L146  │                              │
│                              │  • CVTEliteMap   L218  │                              │
│                              └────────────────────────┘                              │
│                                                                                      │
└─────────────────────────────────────────┬────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                               MAIN EVOLUTION LOOP                                    │
│                          evolution.py:codeevolve_loop() L547                         │
│                                                                                      │
│                      for epoch in range(start_epoch, num_epochs):                    │
│                                                                                      │
└─────────────────────────────────────────┬────────────────────────────────────────────┘
                                          │
                                          ▼
```

---

## Epoch Detail: The 6-Step Evolution Cycle

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: MODE DECISION                                             evolution.py L636 │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│      exploration = random() <= exploration_rate                                      │
│      exploitation = not exploration                                                  │
│                                                                                      │
│      ┌───────────────────────────────┐      ┌───────────────────────────────┐       │
│      │      EXPLORATION MODE         │      │      EXPLOITATION MODE        │       │
│      │                               │      │                               │       │
│      │  • Random parent selection    │      │  • Fitness-based selection    │       │
│      │  • No chat history (depth=0)  │      │  • Full chat history          │       │
│      │  • exploration_ensemble LLM   │      │  • exploitation_ensemble LLM  │       │
│      │  • Encourages diversity       │      │  • Refines best solutions     │       │
│      └───────────────────────────────┘      └───────────────────────────────┘       │
│                                                                                     │
│      Exploration rate controlled by scheduler:                      scheduler.py    │
│        • ExponentialDecayScheduler L86   - rate decays over time                    │
│        • PlateauScheduler L160           - adapts based on fitness improvement      │
│        • CosineScheduler L266            - oscillates periodically                  │
│                                                                                     │
└─────────────────────────────────────────┬───────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: PARENT SELECTION                                          evolution.py L646  │
│                                        select_parents() L39                          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        ProgramDatabase.sample()                                │  │
│  │                             database.py L525                                   │  │
│  │                                                                                │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   update_caches()  database.py L376                      │  │  │
│  │  │                                                                          │  │  │
│  │  │     if elite_map is not None:           ◄── MAP-ELITES INTEGRATION       │  │  │
│  │  │         _pids_pool_cache = elite_map.get_elite_ids()                     │  │  │
│  │  │     else:                                                                │  │  │
│  │  │         _pids_pool_cache = [pid for pid if is_alive[pid]]                │  │  │
│  │  │                                                                          │  │  │
│  │  └──────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                      │                                         │  │
│  │                                      ▼                                         │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   SELECTION POLICIES  database.py                        │  │  │
│  │  │                                                                          │  │  │
│  │  │     random_selection()     L433   - Uniform random                       │  │  │
│  │  │     roulette_selection()   L449   - Weighted by fitness/rank             │  │  │
│  │  │     tournament_selection() L483   - Sample k, return best                │  │  │
│  │  │     best_selection()       L513   - Always pick highest fitness          │  │  │
│  │  │                                                                          │  │  │
│  │  └──────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  Returns: (parent_sol, parent_prompt, inspirations[])                                │
│                                                                                      │
└─────────────────────────────────────────┬────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: META-PROMPTING (Optional)                                 evolution.py L662 │
│                                        run_meta_prompting() L127                     │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  if meta_prompting and not exploitation:                                             │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      PromptSampler.meta_prompt()              prompt/sampler.py                │  │
│  │                                                                                │  │
│  │      1. Query aux_lm to generate prompt modification                           │  │
│  │      2. LLM outputs SEARCH/REPLACE diff for prompt                             │  │
│  │      3. apply_diff() to create child_prompt          utils/parsing_utils.py    │  │
│  │      4. Add child_prompt to prompt_db                                          │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  Evolves the system prompt itself for better guidance                                │
│                                                                                      │
└─────────────────────────────────────────┬────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: SOLUTION GENERATION                                       evolution.py L683 │
│                                        generate_solution() L255                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      1. BUILD CHAT MESSAGES             prompt/sampler.py:build()              │  │
│  │                                                                                │  │
│  │         • System prompt (possibly evolved)                                     │  │
│  │         • Parent solution code                                                 │  │
│  │         • Inspiration programs (if exploitation)                               │  │
│  │         • Conversation history (if max_chat_depth > 0)                         │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                           │
│                                          ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      2. QUERY LLM ENSEMBLE              lm.py:LMEnsemble.generate()            │  │
│  │                                                                                │  │
│  │         • Weighted random model selection from ensemble                        │  │
│  │         • OpenAI-compatible API call                                           │  │
│  │         • Returns: SEARCH/REPLACE diff                                         │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                           │
│                                          ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      3. APPLY DIFF                      utils/parsing_utils.py:apply_diff()    │  │
│  │                                                                                │  │
│  │         • Parse SEARCH/REPLACE blocks                                          │  │
│  │         • Apply only within # EVOLVE-BLOCK-START/END markers                   │  │
│  │         • Create child_sol Program object                                      │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└─────────────────────────────────────────┬────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: EVALUATION & STORAGE                                      evolution.py L703 │
│                                        evaluate_and_store() L387                     │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      1. EXECUTE PROGRAM                 evaluator.py:Evaluator.execute()       │  │
│  │                                                                                │  │
│  │         • Write child code to temp file                                        │  │
│  │         • Spawn subprocess with timeout & memory limits                        │  │
│  │         • Run: python evaluate.py <code_path> <results_path>                   │  │
│  │         • Parse JSON results for metrics                                       │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                           │
│                                          ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      2. EXTRACT FITNESS                 evolution.py L436                      │  │
│  │                                                                                │  │
│  │         child_sol.fitness = eval_metrics[fitness_key]                          │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                           │
│                                          ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      3. ADD TO DATABASE                 database.py:ProgramDatabase.add() L399 │  │
│  │                                                                                │  │
│  │      ┌──────────────────────────────────────────────────────────────────────┐  │  │
│  │      │   if elite_map is not None:           ◄── MAP-ELITES ARCHIVING       │  │  │
│  │      │       elite_map.add_elite(prog)                                      │  │  │
│  │      │                                                                      │  │  │
│  │      │       GridEliteMap.add_elite() L194:                                 │  │  │
│  │      │         cell_idx = discretize(prog.features)                         │  │  │
│  │      │         if prog.fitness > map[cell_idx].fitness:                     │  │  │
│  │      │             map[cell_idx] = (prog.id, prog.fitness)                  │  │  │
│  │      │                                                                      │  │  │
│  │      │       CVTEliteMap.add_elite() L252:                                  │  │  │
│  │      │         centroid_idx = closest_centroid(prog.features)               │  │  │
│  │      │         if prog.fitness > map[centroid_idx].fitness:                 │  │  │
│  │      │             map[centroid_idx] = (prog.id, prog.fitness)              │  │  │
│  │      │                                                                      │  │  │
│  │      └──────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                                │  │
│  │      else:  # Standard EA                                                      │  │
│  │          if num_alive < max_alive:                                             │  │
│  │              add to population                                                 │  │
│  │          elif prog.fitness >= worst.fitness:                                   │  │
│  │              replace worst                                                     │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└─────────────────────────────────────────┬────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: MIGRATION & HOUSEKEEPING                                  evolution.py L717 │
│                                        handle_migration() L489                       │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  if epoch % migration_interval == 0:                                                 │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      MIGRATION                          islands.py:sync_migrate() L71          │  │
│  │                                                                                │  │
│  │      1. Select top programs as migrants       database.py:get_migrants() L563  │  │
│  │      2. Wait at barrier for all islands                                        │  │
│  │      3. Send migrants to outgoing neighbor (pipe)                              │  │
│  │      4. Receive migrants from incoming neighbor                                │  │
│  │      5. Add received programs to local database                                │  │
│  │                                                                                │  │
│  │      Topologies (islands.py:get_edge_list):                                    │  │
│  │        • directed_ring  :  0→1→2→...→N→0                                       │  │
│  │        • ring           :  bidirectional ring                                  │  │
│  │        • complete       :  all-to-all                                          │  │
│  │        • star           :  hub-and-spoke                                       │  │
│  │        • empty          :  no migration                                        │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      CHECKPOINTING                      utils/ckpt_utils.py:save_ckpt()        │  │
│  │                                                                                │  │
│  │      if epoch % ckpt == 0:                                                     │  │
│  │          Save: prompt_db, sol_db, evolve_state, scheduler                      │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │      EARLY STOPPING                     islands.py:early_stopping_check() L99  │  │
│  │                                                                                │  │
│  │      Track consecutive epochs without global improvement                       │  │
│  │      Stop if early_stop_counter >= early_stopping_rounds                       │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## MAP-Elites Integration Summary

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              MAP-ELITES INTEGRATION                                  │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  MAP-Elites maintains a grid/CVT of "elites" - the best solution found for each     │
│  region of the feature space. This encourages diversity while preserving quality.   │
│                                                                                      │
│  CONFIGURATION (from YAML):                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │    use_map_elites: true                                                        │  │
│  │    MAP_ELITES:                                                                 │  │
│  │      elite_map_type: 'cvt'  # or 'grid'                                        │  │
│  │      features:                                                                 │  │
│  │        - {name: 'sum_radii', min_val: 0, max_val: 3}                           │  │
│  │        - {name: 'benchmark_ratio', min_val: 0, max_val: 1.5}                   │  │
│  │        - {name: 'eval_time', min_val: 0, max_val: 60}                          │  │
│  │      num_centroids: 30  # for CVT                                              │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  TWO INTEGRATION POINTS:                                                             │
│                                                                                      │
│  1. SELECTION (database.py L382-385)                                                 │
│     ┌──────────────────────────────────────────────────────────────────────────┐     │
│     │   Standard EA  :  select from all alive programs                         │     │
│     │   MAP-Elites   :  select from elite archive only                         │     │
│     │                   (diverse high-performers across feature space)         │     │
│     └──────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
│  2. ARCHIVING (database.py L415-417)                                                 │
│     ┌──────────────────────────────────────────────────────────────────────────┐     │
│     │   Standard EA  :  replace worst if better                                │     │
│     │   MAP-Elites   :  place in feature cell, replace only if better          │     │
│     │                   than current cell occupant                             │     │
│     └──────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
│  ABLATION RESULT (CirclePackingSquare n=32):                                         │
│     ┌──────────────────────────────────────────────────────────────────────────┐     │
│     │   CVT MAP-Elites   ≈ 2.8-3.0   ✅ Beats AlphaEvolve                      │     │
│     │   Grid MAP-Elites  ≈ 2.6-2.7   ~  Approaches AlphaEvolve                 │     │
│     │   Naive (no ME)    ≈ 2.3-2.4   ❌ Below AlphaEvolve                      │     │
│     └──────────────────────────────────────────────────────────────────────────┘     │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Reference

| File                                                        | Key Functions                              | Purpose                              |
| ----------------------------------------------------------- | ------------------------------------------ | ------------------------------------ |
| [cli.py](src/codeevolve/cli.py)                             | `main()` L370, `spawn_island_processes()`  | Entry point, process spawning        |
| [evolution.py](src/codeevolve/evolution.py)                 | `codeevolve()` L811, `codeevolve_loop()`   | Main evolution loop                  |
| [evolution.py](src/codeevolve/evolution.py)                 | `select_parents()` L39                     | Parent selection dispatch            |
| [evolution.py](src/codeevolve/evolution.py)                 | `generate_solution()` L255                 | LLM code generation                  |
| [evolution.py](src/codeevolve/evolution.py)                 | `evaluate_and_store()` L387                | Evaluation and DB update             |
| [database.py](src/codeevolve/database.py)                   | `ProgramDatabase` L282, `sample()` L525    | Population management, selection     |
| [database.py](src/codeevolve/database.py)                   | `GridEliteMap` L146, `CVTEliteMap` L218    | MAP-Elites implementations           |
| [islands.py](src/codeevolve/islands.py)                     | `sync_migrate()` L71                       | Island migration                     |
| [islands.py](src/codeevolve/islands.py)                     | `early_stopping_check()` L99               | Early stopping logic                 |
| [lm.py](src/codeevolve/lm.py)                               | `LMEnsemble`, `OpenAILM`                   | LLM API interface                    |
| [evaluator.py](src/codeevolve/evaluator.py)                 | `Evaluator.execute()`                      | Sandboxed code execution             |
| [scheduler.py](src/codeevolve/scheduler.py)                 | `ExponentialDecayScheduler` L86            | Rate decay over time                 |
| [scheduler.py](src/codeevolve/scheduler.py)                 | `PlateauScheduler` L160                    | Adaptive rate based on fitness       |
| [scheduler.py](src/codeevolve/scheduler.py)                 | `CosineScheduler` L266                     | Oscillating rate                     |
| [prompt/sampler.py](src/codeevolve/prompt/sampler.py)       | `PromptSampler.build()`, `meta_prompt()`   | Chat construction, prompt evolution  |
| [utils/parsing_utils.py](src/codeevolve/utils/parsing_utils.py) | `apply_diff()`                         | SEARCH/REPLACE diff application      |
| [utils/ckpt_utils.py](src/codeevolve/utils/ckpt_utils.py)   | `save_ckpt()`, `load_ckpt()`               | Checkpoint serialization             |
