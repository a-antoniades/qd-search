# Short-Horizon ML Agent Benchmarks for AIRA-dojo

*Deep research report — February 2026*

## Executive Summary

We surveyed 50+ benchmarks across 9 categories to find where AIRA-dojo (LLM-driven ML agent, single A100, 30min-2hr steps, ~8hr total budget) can realistically compete. The key finding: **the strongest opportunities lie at the intersection of algorithm discovery and tabular ML**, where algorithmic creativity matters more than compute scale. A new wave of agent-specific ML benchmarks (MLAgentBench, MLGym, CO-Bench) has emerged in 2024-2025, purpose-built for exactly this use case.

---

## Unified Benchmark Table

### Tier 1: Highest Priority — Best Fit for AIRA-dojo

| # | Benchmark | Domain | Tasks | Train Time | Hardware | SOTA Gap | Diversity | Agent Fit | Priority | Interestingness | Links |
|---|-----------|--------|-------|------------|----------|----------|-----------|-----------|----------|-----------------|-------|
| 1 | **CO-Bench** | Algorithm Discovery | 36 combinatorial opt problems | Minutes (no training) | CPU only | High (best=0.84) | Very High | Perfect — designed for LLM agents | **Critical** | ★★★★★ | [Paper](https://arxiv.org/abs/2504.04310) · [GitHub](https://github.com/sunnweiwei/CO-Bench) |
| 2 | **HeuriGym** | Algorithm Discovery | 9 heuristic design problems | Minutes (iterative) | CPU only | High (best QYI=0.6) | High | Perfect — interactive agent loop | **Critical** | ★★★★★ | [Paper](https://arxiv.org/abs/2506.07972) · [GitHub](https://github.com/cornell-zhang/heurigym) |
| 3 | **MLAgentBench** | Agent-Specific ML | 13 ML research tasks | Min-hours per task | Single GPU | High (37.5% avg success) | High (CV, NLP, tabular, graph) | Perfect — built for ML agents | **Critical** | ★★★★★ | [Paper](https://arxiv.org/abs/2310.03302) · [GitHub](https://github.com/snap-stanford/MLAgentBench) |
| 4 | **TabZilla-Hard** | Tabular ML | 36 hard classification tasks | Minutes per task | CPU/GPU | High (selected for headroom) | High | Excellent — rewards creative approaches | **Critical** | ★★★★★ | [Paper](https://arxiv.org/abs/2305.02997) · [GitHub](https://github.com/naszilla/tabzilla) |
| 5 | **ARC-AGI-2** | Abstract Reasoning | 240 visual puzzles | Min/puzzle (no training) | Single GPU feasible | Massive (24% vs 95% human) | Very High | Excellent — iterative refinement | **Critical** | ★★★★★ | [Website](https://arcprize.org/) · [Paper](https://arxiv.org/abs/2505.11831) |
| 6 | **MLGym** | Agent-Specific ML | 13 open-ended research tasks | Task-dependent | Single GPU | High (models reach L1 only) | High (CV, NLP, RL) | Perfect — Gym env for ML agents | **Critical** | ★★★★★ | [Paper](https://arxiv.org/abs/2502.14499) · [GitHub](https://github.com/facebookresearch/MLGym) |
| 7 | **LLM-FE / CAAFE** | Tabular Feature Eng. | 14-29 datasets | Minutes | CPU/GPU | High (+2-11% over baselines) | High | Perfect — directly tests LLM-driven FE | **Critical** | ★★★★★ | [CAAFE](https://github.com/noahho/CAAFE) · [LLM-FE](https://arxiv.org/abs/2503.14434) |

### Tier 2: High Priority — Excellent Fit

| # | Benchmark | Domain | Tasks | Train Time | Hardware | SOTA Gap | Diversity | Agent Fit | Priority | Interestingness | Links |
|---|-----------|--------|-------|------------|----------|----------|-----------|-----------|----------|-----------------|-------|
| 8 | **D4RL** | Offline RL | 20+ envs | <20min (IQL on A100) | Single GPU | Moderate-High | High | Excellent — fixed data, algo design | **High** | ★★★★ | [Paper](https://arxiv.org/abs/2004.07219) · [GitHub](https://github.com/Farama-Foundation/D4RL) |
| 9 | **Brax** | RL (Continuous) | Standard locomotion | Seconds-minutes | Single GPU (JAX) | Moderate | Moderate | Excellent — ultra-fast iteration | **High** | ★★★★ | [GitHub](https://github.com/google/brax) |
| 10 | **BBOB/COCO** | Black-Box Optimization | 24 functions × 6 dims | Seconds per eval | CPU only | Open (LLM-designed optimizers is new) | Very High | Excellent — instant eval, algo design | **High** | ★★★★ | [GitHub](https://github.com/numbbo/coco) · [Website](https://coco-platform.org/) |
| 11 | **BLADE** | LLM Algorithm Design | Continuous optimization | Fast (surrogate) | CPU + LLM | New benchmark | Moderate | Excellent — LLM algo design focus | **High** | ★★★★ | [Paper](https://arxiv.org/abs/2504.20183) · [GitHub](https://github.com/XAI-liacs/BLADE) |
| 12 | **RE-Bench** | AI R&D Automation | 7 research eng. tasks | 2-8hr budget | Multi-GPU | Compared to 61 experts | Moderate | Excellent — open-ended R&D | **High** | ★★★★★ | [Paper](https://arxiv.org/abs/2411.15114) · [GitHub](https://github.com/METR/RE-Bench) |
| 13 | **TabArena** | Tabular ML (Living) | 51 curated datasets | Min-hours | Single machine | Active leaderboard | Very High | Excellent — living benchmark | **High** | ★★★★ | [Website](https://tabarena.ai) · [Paper](https://arxiv.org/abs/2506.16791) |
| 14 | **DataPerf** | Data-Centric AI | 5 challenge tracks | Min-1hr (fixed model) | CPU/GPU | Large (underexplored) | High | Excellent — pure data strategy | **High** | ★★★★★ | [GitHub](https://github.com/mlcommons/dataperf) |
| 15 | **VTAB-1K** | Transfer Learning | 19 diverse vision tasks | Minutes per task | Single GPU | Moderate | Very High | Excellent — per-task adaptation | **High** | ★★★★ | [Website](https://google-research.github.io/task_adaptation/) |
| 16 | **LiveCodeBench** | Program Synthesis | 1055+ problems | Seconds per problem | CPU only | Significant | High | Good — contamination-free, self-repair | **High** | ★★★★ | [Website](https://livecodebench.github.io/) |
| 17 | **SRBench** | Symbolic Regression | Ground-truth + black-box | Min-hours | CPU/GPU | Significant (no dominant method) | Very High | Good — equation discovery | **High** | ★★★★ | [Website](https://cavalab.org/srbench/) |
| 18 | **CDALBench** | Active Learning | 9-14 datasets (CV/NLP/tab) | Min per AL round | Single GPU | Large (no method dominates) | Very High | Excellent — adaptive strategy design | **High** | ★★★★ | [Paper](https://arxiv.org/abs/2408.00426) |

### Tier 3: Medium Priority — Good Fit with Caveats

| # | Benchmark | Domain | Tasks | Train Time | Hardware | SOTA Gap | Diversity | Agent Fit | Priority | Interestingness | Links |
|---|-----------|--------|-------|------------|----------|----------|-----------|-----------|----------|-----------------|-------|
| 19 | **MLE-bench (Lite)** | ML Engineering | 22 low-complexity | Varies (some >2hr) | Single GPU | High (16.9% bronze) | High | Good — industry standard | **Medium** | ★★★★ | [Paper](https://arxiv.org/abs/2410.07095) · [GitHub](https://github.com/openai/mle-bench) |
| 20 | **ML-Dev-Bench** | ML Development | 30 workflow tasks | Task-dependent | Standard | Moderate | High | Good — full ML workflow | **Medium** | ★★★ | [Paper](https://arxiv.org/abs/2502.00964) · [GitHub](https://github.com/ml-dev-bench/ml-dev-bench) |
| 21 | **dcbench** | Data-Centric AI | 3 task types | Minutes | CPU | Large | Medium | Excellent — data strategy | **Medium** | ★★★★ | [GitHub](https://github.com/data-centric-ai/dcbench) |
| 22 | **DomainBed** | Domain Generalization | 5 datasets | <1hr per run | Single GPU | ERM hard to beat | Medium | Good — rewards innovation | **Medium** | ★★★★ | [Paper](https://arxiv.org/abs/2007.01434) · [GitHub](https://github.com/facebookresearch/DomainBed) |
| 23 | **Isaac Gym** | RL (GPU-accelerated) | Robotics tasks | 2-4min on A100 | Single A100 | Moderate | Moderate | Good — ultra-fast, A100-native | **Medium** | ★★★ | [NVIDIA](https://developer.nvidia.com/isaac-gym) |
| 24 | **APEBench** | PDE Emulation (JAX) | 46 PDEs | Fast (JAX-native) | Single GPU | Wide open (new 2024) | Very High | Good — novel training strategies | **Medium** | ★★★★ | [Paper](https://arxiv.org/abs/2411.00180) · [GitHub](https://github.com/tum-pbs/apebench) |
| 25 | **HPO-B / YAHPO Gym** | HPO Algorithm Design | 176 spaces × 196 datasets | Milliseconds (lookup) | CPU only | Moderate | Very High | Good — instant feedback | **Medium** | ★★★ | [HPO-B](https://github.com/machinelearningnuremberg/HPO-B) · [YAHPO](https://github.com/slds-lmu/yahpo_gym) |
| 26 | **TALENT-tiny** | Tabular Evaluation | 45 core datasets | Minutes | CPU/GPU | Active area | Very High | Good — standardized toolkit | **Medium** | ★★★ | [Paper](https://arxiv.org/abs/2407.00956) · [GitHub](https://github.com/LAMDA-Tabular/TALENT) |
| 27 | **MoleculeNet** | Molecular Properties | 17+ datasets | Min-1hr | Single GPU | Moderate | High | Good — small datasets | **Medium** | ★★★ | [Website](https://moleculenet.org/) |
| 28 | **PMLBmini** | Low-Data Tabular | 44 datasets (≤500 samples) | Minutes | CPU | LogReg often wins | High | Good — tests adaptive strategy | **Medium** | ★★★ | [Paper](https://arxiv.org/abs/2409.01635) · [GitHub](https://github.com/RicardoKnauer/TabMini) |
| 29 | **BigCodeBench** | Practical Programming | 1140 tasks | Seconds | CPU | Large (60% vs 97% human) | High | Good — multi-library tasks | **Medium** | ★★★ | [Website](https://bigcode-bench.github.io/) |
| 30 | **NAS-Bench-101/201/301** | Architecture Search | Lookup tables | Milliseconds | CPU | Limited (small spaces) | Moderate | Moderate — search strategy | **Medium** | ★★ | [NAS-Bench-101](https://arxiv.org/abs/1902.09635) |
| 31 | **DSBench** | Data Science | 540 tasks | Minutes (analysis) | CPU/GPU | High (34% analysis) | High | Moderate — mixed focus | **Medium** | ★★★ | [Paper](https://arxiv.org/abs/2409.07703) · [GitHub](https://github.com/LiqiangJing/DSBench) |
| 32 | **WILDS (fast subset)** | Distribution Shift | 2-3 tasks (~2hr) | 2hr (camelyon17) | Single GPU | Significant | Medium | Moderate — borderline time | **Low-Medium** | ★★★ | [Website](https://wilds.stanford.edu/) · [GitHub](https://github.com/p-lambda/wilds) |
| 33 | **Kaggle Playground** | Tabular Competitions | Monthly | Hours | GPU helpful | Moderate-High | Monthly new | Good — rewards FE creativity | **Medium** | ★★★ | [Kaggle](https://kaggle.com/competitions/playground-series) |

### Not Recommended

| Benchmark | Reason |
|-----------|--------|
| WeatherBench 2 | Weeks on TPU clusters |
| HumanEval / MBPP | Saturated (>95% SOTA) |
| BIG-Bench / MATH / FrontierMath | Tests LLM reasoning, not ML engineering |
| ReX-MLE | Medical imaging, exceeds single A100 |
| MultiAgentBench | Tests coordination, not ML |
| Full MLE-bench (75 tasks) | Many tasks need multi-hour training |

---

## Top 10 Recommendations for AIRA-dojo Evaluation

### Primary Evaluation Suite (6 benchmarks, ~30hr total evaluation time)

| Priority | Benchmark | Why | Est. Eval Time |
|----------|-----------|-----|----------------|
| 1 | **CO-Bench** | Purpose-built for LLM agents, no GPU, 36 problems, perfect match | ~4hr |
| 2 | **MLAgentBench** | Closest to AIRA-dojo's mission, 13 tasks, clear metrics | ~8hr |
| 3 | **TabZilla-Hard** | AIRA's sweet spot (tabular), 36 tasks with real headroom | ~6hr |
| 4 | **ARC-AGI-2** | Iconic benchmark, massive headroom, tests pure reasoning | ~4hr |
| 5 | **LLM-FE / CAAFE** | Directly tests LLM feature engineering, fast, high signal | ~3hr |
| 6 | **D4RL** | Fast offline RL, algorithm design focus, well-established | ~4hr |

### Extended Evaluation (4 more for comprehensive coverage)

| Priority | Benchmark | Why | Est. Eval Time |
|----------|-----------|-----|----------------|
| 7 | **MLGym** | Gym interface, capability levels, RL-trainable, newest | ~6hr |
| 8 | **BBOB/COCO** | Standard optimization benchmark, instant feedback | ~2hr |
| 9 | **DataPerf** | Data-centric AI, pure data strategy, underexplored | ~3hr |
| 10 | **SRBench** | Equation discovery, diverse approaches, fast eval | ~4hr |

---

## Novel Benchmark Directions That Should Exist

### 1. "ML Strategy Benchmark" — Adaptive Method Selection
**Gap**: No benchmark tests whether an agent can diagnose a dataset's properties and select the right approach. MultiTab hints at this (model performance varies by data regime) but doesn't benchmark the selection strategy itself.
**Proposal**: Given a new dataset, the agent must: (a) analyze its properties (size, class imbalance, feature types, noise level), (b) choose an appropriate method, (c) tune it — all within a time budget. Score both accuracy and time-to-solution. This directly tests the meta-reasoning that makes LLM agents valuable.

### 2. "FE-Arena" — Competitive Feature Engineering
**Gap**: CAAFE and LLM-FE test feature engineering but lack a competitive arena format. No benchmark pits different LLM-driven FE strategies against each other in real-time.
**Proposal**: A living leaderboard where agents submit feature engineering programs. New datasets rotate in monthly (like Kaggle Playground). Score by improvement over a fixed XGBoost baseline. This would directly measure the creative FE capability that gives LLM agents an edge.

### 3. "Debug-ML-Bench" — ML Debugging & Error Recovery
**Gap**: Existing benchmarks test "build from scratch" but not "diagnose and fix". Real ML work involves debugging failing training runs, diagnosing data issues, fixing convergence problems. ML-Dev-Bench touches this but doesn't focus on it.
**Proposal**: Provide agents with broken ML pipelines (wrong learning rate, data leakage, label noise, architecture bugs, convergence issues) and measure how quickly and correctly they diagnose and fix them. This tests the iterative debugging loop that LLM agents do naturally.

---

## Gaps in the Benchmark Landscape

1. **No time-constrained ML innovation benchmark**: Existing benchmarks measure accuracy but not the creativity-per-hour trade-off. A benchmark that explicitly rewards novel approaches (not just accuracy) within strict time budgets would better evaluate agents like AIRA-dojo.

2. **No cross-domain transfer benchmark for ML agents**: Can an agent that learned strategies on tabular tasks apply them to scientific computing? No benchmark tests cross-domain transfer of ML engineering skills.

3. **No "explain your approach" evaluation**: Current benchmarks score outputs only. An agent that discovers a novel feature engineering trick but can't explain why it works is less useful than one that can. Interpretability of the agent's ML decisions is untested.

4. **Tabular meta-learning is under-benchmarked**: Despite being the most common data type in industry, there's no comprehensive benchmark for meta-learning strategies across diverse tabular datasets (when to use trees vs. neural nets vs. TabPFN vs. ensembles).

5. **No benchmark for iterative refinement quality**: Agents iterate — but do later iterations actually improve? No benchmark measures the quality trajectory of an agent's iterative refinement process, distinguishing productive iteration from random search.

---

## Key Strategic Insights

1. **The AIDE finding is critical**: On MLE-bench, agents perform significantly better with fewer 2-hour attempts than many 30-minute attempts. AIRA-dojo should default to the 2hr step timeout for complex tasks.

2. **Feature engineering is AIRA's biggest edge**: CAAFE (+2.4% ROC AUC), LLM-FE (rank 1.47/19), and Snowflake FeatEng all show LLMs systematically improve tabular ML through creative feature engineering. This is the single biggest competitive advantage for an LLM agent.

3. **Algorithm discovery benchmarks are the purest test**: CO-Bench, HeuriGym, and BLADE test the core agent loop (propose → evaluate → refine) without confounding factors like data preprocessing or library knowledge.

4. **TabPFN should be a tool, not a competitor**: TabPFN-2.5 matches AutoGluon in a single forward pass on small datasets. AIRA-dojo should include TabPFN in its toolkit and strategically deploy it.

5. **The field is exploding**: 6+ new agent-specific ML benchmarks launched in 2024-2025 (MLAgentBench, MLE-bench, MLGym, ML-Dev-Bench, RE-Bench, ReX-MLE). AIRA-dojo is entering a rapidly maturing evaluation ecosystem.

---

*Report compiled from parallel research across 5 research agents covering: Algorithm Discovery & Program Synthesis, Tabular ML & AutoML, Few-Shot/Meta-Learning/Data-Centric AI, Scientific Computing/Optimization/RL, and Emerging/Robustness/Agent-Specific benchmarks.*
