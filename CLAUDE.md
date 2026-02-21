# qd-search — Quality-Diversity Search Research Projects

Monorepo containing four research projects on LLM-driven evolutionary algorithms, AI agents, and open-ended search.

## Projects

| Project | Purpose | Paper |
|---------|---------|-------|
| **qd/** | Standalone MAP-Elites library (archives, features, selection, metrics) | — |
| **aira-dojo/** | AI Research Agent framework for MLE-bench (Meta FAIR) | arXiv:2507.02554 |
| **science-codeevolve/** | LLM-driven evolutionary algorithm discovery (CodeEvolve) | arXiv:2510.14150 |
| **aideml/** | ML Engineering Agent with tree search | arXiv:2502.13138 |
| **omni-epic/** | Open-ended environment generation with RL + LLM | arXiv:2405.15568 |

## Quick Reference

### qd/ (MAP-Elites library)
- **Pure Python** library, no special env needed (requires numpy)
- **Archives:** `GridArchive` (fixed grid), `CVTArchive` (Voronoi tessellation) — both in `map_elites.py`
- **Features:** Keyword-based extraction (`features.py`) — 2 dimensions: model_family (6 bins: Classical ML, GBDT, CNN, RNN, Transformer, Ensemble) × data_strategy (5 bins: Simple, K-Fold CV, Augmentation, Transfer Learning, Feature Eng.) = 30 cells
- **Selection:** `Selector` base class (`selection.py`) for parent selection strategies
- **Metrics:** `coverage()`, `qd_score()`, `best_fitness()` in `metrics.py`
- **Key API:** `archive.add(id, fitness, features)`, `archive.elites()`, `archive.occupied_cells()`, `extract_features(plan, code)`, `feature_names(features)`
- **Tests:** `qd/tests/`
- **Used by:** `experiments/20260220_map_elites_ideation/` (ideation loop, baseline, comparison)

### aira-dojo
- **Env:** `conda activate aira-dojo` (Python 3.12)
- **Config:** Hydra YAML in `src/dojo/configs/`, experiments in `configs/_exp/`
- **Run single:** `python -m dojo.main_run +_exp=run_example logger.use_wandb=False`
- **Run parallel (SLURM):** `python -m dojo.main_runner_job_array +_exp=runner_example`
- **Key paths (from .env):** logs → `aira-dojo/logs/`, data → `aira-dojo/data/mlebench/`, containers → `aira-dojo/sif/`
- **Solvers:** AIRA_GREEDY, AIDE_GREEDY, AIRA_MCTS, AIRA_EVO

### science-codeevolve
- **Env:** `conda activate codeevolve` (Python 3.13.5+)
- **Run:** `codeevolve --inpt_dir=input --cfg_path=config.yaml --out_dir=results --terminal_logging`
- **Resume from checkpoint:** `--load_ckpt=-1`
- **Architecture:** Islands-based GA + MAP-Elites, SEARCH/REPLACE diffs, sandboxed evaluation

### aideml
- **Install:** `pip install -U aideml`
- **Run:** `aide data_dir="example_tasks/house_prices" goal="Predict price" eval="RMSE"`
- **Web UI:** `streamlit run aide/webui/app.py`

### omni-epic
- **Container:** `apptainer build apptainer/container.sif apptainer/container.def`
- **Run:** `python main_omni_epic.py`
- **Game UI:** `python -m game.backend.app` + `cd game/frontend && npm run dev`

## Experiment Structure

Experiments are organized in `experiments/` directories with symlinks to run data (no duplication).

**Top-level experiments** (cross-project analysis):
```
qd-search/experiments/
└── 20260205_163244_qd-diversity-study/
    ├── experiment.py      # Analysis script
    ├── data/, figures/    # Outputs
    ├── runs/              # Symlinks to aira-dojo/logs/aira-dojo/...
    │   ├── evo_gdm → QD_STUDY_evo_gdm (12GB)
    │   └── greedy_gdm → QD_STUDY_greedy_gdm (1.8GB)
    └── console_logs/      # Symlinks to *.log files
```

**Project-specific experiments** (aira-dojo):
```
aira-dojo/experiments/
└── 20260215_evo_rlm_gemini3/
    ├── ANALYSIS_REPORT.md
    ├── runs/ → symlinks to logs/aira-dojo/...
    └── data/, figures/, logs/
```

**Key paths:**
- Run artifacts (JOURNAL.jsonl, code, metrics): `aira-dojo/logs/aira-dojo/user_<user>_issue_<id>/`
- Console logs: `aira-dojo/logs/*.log`
- Use `/experiment` skill to create new experiments (auto-configures `LOGGING_DIR` for aira-dojo)

## Environment Variables

```bash
# LLM API Keys (project-dependent)
OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY

# aira-dojo specific
LOGGING_DIR, MLE_BENCH_DATA_DIR, SUPERIMAGE_DIR

# codeevolve specific
API_KEY, API_BASE
```

## Shared Tech Stack
- **LLM backends:** anthropic, openai, google-genai, litellm
- **Data:** numpy, pandas, scikit-learn, scipy
- **Config:** Hydra/OmegaConf (aira-dojo), YAML (codeevolve)
- **Tracking:** wandb, tensorboard
- **Execution:** sandboxed subprocess, Apptainer containers, SLURM

## Current Focus
- Active development on **aira-dojo**, **qd/**, and **experiments/20260220_map_elites_ideation/**

### RLM Node Selection (Feb 2026)
Intelligent node selection for EVO solver using RLM (Recursive Language Models).

**Module:** `dojo.core.solvers.selection`
- `FitnessNodeSelector`: Softmax-weighted selection (original EVO behavior)
- `RLMNodeSelector`: LLM-based intelligent selection using plan/architecture analysis

**Usage:**
```bash
# Default (fitness-based selection - backward compatible)
python -m dojo.main_run +_exp=run_example solver=evo

# RLM-based intelligent selection
python -m dojo.main_run +_exp=run_example solver=evo solver.selector=rlm
```

**Config files:**
- `configs/selector/fitness.yaml` — Fitness-based (default)
- `configs/selector/rlm.yaml` — RLM-based (requires GEMINI_API_KEY)

**Key features:**
- Extracts architecture keywords from plans/code (ConvNeXt, XGBoost, Transformer, etc.)
- Analyzes error patterns for buggy nodes
- Considers medal status (GOLD/SILVER/BRONZE)
- Falls back to fitness selector if RLM fails

## Active Experiments

### QD Diversity Study (Feb 2026)
- **Goal:** Measure quality-diversity in aira-dojo — do agents generate diverse approaches or converge?
- **Comparison:** AIRA_GREEDY (5 drafts + improve best) vs AIRA_EVO (island-based evo with crossover)
- **LLM:** `gemini-3-flash-preview` via `litellm` client (config: `litellm_gemini`, 10 retries with backoff)
- **Tasks (5):** tabular-playground-series-dec-2021, spooky-author-identification, dog-breed-identification, learning-agency-lab-automated-essay-scoring-2, stanford-covid-vaccine
- **Seeds:** 3 per task × solver → 30 total runs
- **Configs:** `configs/_exp/mlebench/qd_greedy_gdm.yaml`, `qd_evo_gdm.yaml`, `benchmark/mlebench/qd_study.yaml`
- **Run:** `cd aira-dojo && conda run -n aira-dojo bash scripts/run_qd_study.sh {greedy|evo|all} [--dry|--local|--single <task>|--parallel [gpu_ids]]`
- **Run parallel:** `conda run -n aira-dojo bash scripts/run_qd_study.sh greedy --parallel 6,7` (2 workers on GPUs 6,7)
- **Execution timeout:** 1800s (30 min) per step (reduced from 14400s to prevent hangs)
- **Monitor logs:** `tail -f aira-dojo/logs/qd_study_greedy_*.log`
- **Analysis:** `python experiments/20260205_163244_qd-diversity-study/experiment.py --exp_dir $LOGGING_DIR/aira-dojo`
- **Machine:** 96 CPUs, 8x A100 40GB GPUs (GPUs 0-4 occupied by other users, 5-7 free)
- **Note:** SLURM not configured (DEFAULT_SLURM_ACCOUNT commented in .env). Use local runs or `run_qd_study.sh`.
- **Note:** Single runs require `task=mlebench/_default task.name=<name>` (not `+_exp=...` which is for the runner)
- **Note:** `google-research-identify-contrails` skipped (35GB dataset). Seed 1 greedy/tabular-playground completed previously (gold medal, 0.96071).
- **Critical fixes applied:**
  1. `PythonInterpreter` `__name__` guard fix (code with `if __name__ == "__main__":` was silently skipping)
  2. HF expired token cleanup (env vars + token files at `~/.cache/huggingface/` and `$HF_HOME/`)
  3. `use_test_score=true` + `is_buggy` override (Gemini can't do function calling → analyze returns plain text → use grader score instead)
  4. `litellm.UnsupportedParamsError` catch for Gemini `function_call` param
- **Status (Feb 7):** All 15 greedy runs launched with fixes on GPUs 4-7. Verified working: dog-breed metric=0.44843, spooky-author metric=0.38682. Evo runs pending.

### EVO Runs Status (Feb 11)

**Additional fixes applied for EVO solver crashes:**
1. `main_run.py:116-131` — Added fallback to `metric.value` when "score" key missing from `metric.info`
2. `evo.py:1088-1091` — Added try-except around `export_search_results()` to match greedy solver pattern
3. `gdm.py:71,174-204` — Added `httpx.TransportError` to retry logic for network failures (DNS, connection errors)

**EVO Results & Diversity Assessment (seed 1 only):**

| Task | Status | Best Score | Valid Solutions | Architectures Explored | Diverse? |
|------|--------|------------|-----------------|------------------------|----------|
| tabular-playground | ✅ Done | 0.963 | 39 | XGBoost, LightGBM, CatBoost | ❌ No - GBDT convergence |
| spooky-author | 🔄 Running | 0.265 | 55 | 97% Transformers only | ❌ No - NN dominated |
| dog-breed | 🔄 Running | 0.393 | 19 | 68% ConvNeXt, rest scattered | ❌ No - premature convergence |
| stanford-covid | 🔄 Running | 0.221 🥇 | 74 | GRU, Transformer, BiGRU, CNN, LSTM, GNN (7 types) | ✅ Yes |
| learning-agency-lab | 🔄 Running | 0.817 | 8 | TF-IDF+LightGBM, TF-IDF+Ridge, DeBERTa | ✅ Yes |

**Key findings:**
- Only 2/5 runs show genuine architectural diversity
- Most runs converge to a single dominant architecture early (ConvNeXt for images, Transformers for text)
- Stanford-covid shows best QD behavior: 7 architectures, 19 unique combinations, no dominant approach
- Simplicity often wins: single tuned LightGBM (0.817) beat complex ensembles on essay scoring
