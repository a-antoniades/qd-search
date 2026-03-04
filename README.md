# qd-search

Research monorepo for LLM-driven evolutionary search, quality-diversity optimization, and AI research agents.

## Projects

| Directory | Description | Upstream | Paper |
|-----------|-------------|----------|-------|
| `qd/` | MAP-Elites library — grid/CVT archives, feature extraction, selection strategies | — | — |
| `aira-dojo/` | AI Research Agent framework for MLE-bench | [facebookresearch/aira-dojo](https://github.com/facebookresearch/aira-dojo) | [2507.02554](https://arxiv.org/abs/2507.02554) |
| `aideml/` | ML engineering agent with tree search | [WecoAI/aideml](https://github.com/WecoAI/aideml) | [2502.13138](https://arxiv.org/abs/2502.13138) |
| `science-codeevolve/` | LLM-driven evolutionary algorithm discovery | [inter-co/science-codeevolve](https://github.com/inter-co/science-codeevolve) | [2510.14150](https://arxiv.org/abs/2510.14150) |
| `omni-epic/` | Open-ended environment generation with RL + LLM | [maxencefaldor/omni-epic](https://github.com/maxencefaldor/omni-epic) | [2405.15568](https://arxiv.org/abs/2405.15568) |
| `Automated-AI-Researcher/` | Automated research idea generation and evaluation | [NoviScl/Automated-AI-Researcher](https://github.com/NoviScl/Automated-AI-Researcher) | — |
| `evaluator/` | LLM-based solution scoring via Swiss tournaments | — | — |
| `external/` | Vendored references (RLM, PiEvolve, OpenCode) | — | — |

## Quick Start

```bash
# aira-dojo — run a single MLE-bench task
conda activate aira-dojo
python -m dojo.main_run +_exp=run_example logger.use_wandb=False

# aideml — solve a Kaggle-style task
pip install -U aideml
aide data_dir="example_tasks/house_prices" goal="Predict price" eval="RMSE"

# science-codeevolve — evolve algorithms
conda activate codeevolve
codeevolve --inpt_dir=input --cfg_path=config.yaml --out_dir=results

# qd library — use MAP-Elites directly
from qd.map_elites import GridArchive
archive = GridArchive(dims=[6, 5], ranges=[(0, 6), (0, 5)])
archive.add(id="solution_1", fitness=0.95, features=[2, 3])
```

## Repository Structure

```
qd-search/
├── qd/                        # Core QD library (pure Python)
├── evaluator/                 # Solution scoring & benchmarking
├── aira-dojo/                 # AI Research Agent (Hydra configs, solvers, operators)
├── aideml/                    # ML Agent with tree search
├── science-codeevolve/        # Evolutionary algorithm discovery
├── omni-epic/                 # Open-ended RL environments
├── Automated-AI-Researcher/   # Research automation pipeline
├── external/                  # Vendored third-party repos
├── experiments/               # Analysis scripts & results (gitignored)
├── docs/                      # Research notes
└── papers/                    # Reference papers
```

## Tech Stack

**LLM backends:** Anthropic, OpenAI, Google GenAI, LiteLLM
**Config:** Hydra/OmegaConf, YAML
**Execution:** Sandboxed subprocess, Apptainer containers, SLURM
**Tracking:** W&B, TensorBoard
