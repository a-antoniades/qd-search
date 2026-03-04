# Short-Horizon Benchmarks for ML Agents

Comprehensive research on benchmarks suitable for AIRA-dojo evaluation, where solution horizon is 0-1 hour on single GPU.

**Criteria**: Training time <1hr, algorithmic creativity > compute, multiple approaches viable, agent-friendly iteration.

---

## Unified Benchmark Table

| Benchmark | Domain | Time | SOTA Gap | Agent Fit | Priority | Interestingness | Link |
|-----------|--------|------|----------|-----------|----------|-----------------|------|
| **ARC-AGI-2** | Abstract Reasoning | 2-5 min/task | Very High (SOTA 54%, o3→4%) | Excellent - DSL synthesis, few-shot | **High** | Flagship reasoning benchmark, $1M prize | [arcprize.org](https://arcprize.org) |
| **ScienceAgentBench** | Scientific Discovery | 10-30 min | High (best 42.2%) | Excellent - data analysis, iteration | **High** | 102 tasks from peer-reviewed papers | [GitHub](https://github.com/OSU-NLP-Group/ScienceAgentBench) |
| **HeuriGym** | Combinatorial Optimization | Minutes/problem | High (QYI ~0.6) | Excellent - heuristic design | **High** | Purpose-built for LLM agents | [GitHub](https://github.com/cornell-zhang/heurigym) |
| **MLGym** | AI Research | 30-60 min | High (no novel hypotheses) | Excellent - experiment design | **High** | Meta FAIR, tests research capability | [GitHub](https://github.com/facebookresearch/MLGym) |
| **SRBench** | Symbolic Regression | 10 min - 1 hr | Moderate (91.6% noiseless) | Excellent - equation discovery | **High** | Discover physics formulas from data | [cavalab.org/srbench](https://cavalab.org/srbench/) |
| **OpenML-CC18** | Tabular Classification | 10-30 min/task | 3-8% | High - feature engineering | **High** | 72 datasets, semantic column names | [OpenML](https://www.openml.org/s/99) |
| **TabArena** | Tabular ML | <1 hr/config | 2-5% | High - living leaderboard | **High** | 51 datasets, clear baselines | [GitHub](https://github.com/autogluon/tabarena) |
| **Kaggle Playground** | Tabular (monthly) | 15-60 min | 5-15% | Very High - feature creativity | **High** | Fresh tasks monthly, no overfitting | [Kaggle](https://kaggle.com/competitions/playground-series) |
| **PDEBench (1D/2D)** | PDE Simulation | 20-60 min/task | High (~4e-4 error) | Excellent - architecture design | **High** | Physics-informed ML | [GitHub](https://github.com/pdebench/PDEBench) |
| **MoleculeNet (small)** | Drug Discovery | 10-60 min/task | Moderate (82.7% ROC) | Excellent - GNN design | **High** | ESOL, FreeSolv, HIV tasks | [moleculenet.org](https://moleculenet.org/) |
| **WRENCH** | Weak Supervision | 20-60 min | High (many LF combos) | Very High - LLM writes labeling funcs | **High** | 22 datasets, programmatic labels | [GitHub](https://github.com/JieyuZ2/wrench) |
| **DataPerf** | Data Selection | 10-30 min | Moderate | High - strategy iteration | **High** | MLCommons, fixed model | [GitHub](https://github.com/mlcommons/dataperf) |
| **Gymnax/Brax** | RL (JAX) | Seconds-minutes | Low-Medium | Excellent - algorithm prototyping | **High** | 4000x speedup, rapid iteration | [GitHub](https://github.com/google/brax) |
| **Craftax** | RL (exploration) | <1 hr for 1B steps | High | Excellent - deep exploration | **High** | Open-ended survival, procedural | [GitHub](https://github.com/MichaelTMatthews/Craftax) |
| **D4RL (Offline)** | Offline RL | 30-60 min | Medium | Good - pure algorithm design | **High** | No env interaction needed | [GitHub](https://github.com/Farama-Foundation/D4RL) |
| **ML-Dev-Bench** | ML Development | 15-45 min | High (3 agents) | Excellent - debugging, training | **High** | Full ML workflow | [OpenReview](https://openreview.net/forum?id=zrYm2MhGpS) |
| **MiniF2F** | Theorem Proving | Minutes/theorem | Moderate | Excellent - proof synthesis | **High** | Formal verification feedback | [GitHub](https://github.com/openai/miniF2F) |
| **ASAC** | Algorithm Synthesis | Minutes/task | High (ChatGPT 8.8%) | Excellent - formal specs | **High** | 136 tasks, efficiency requirements | [ASAC](https://auqwqua.github.io/ASACBenchmark/) |
| **CodeARC** | Program Synthesis | Minutes/task | New benchmark | Excellent - inductive synthesis | **High** | Differential testing oracle | [OpenReview](https://openreview.net/pdf?id=NImIdZFJXW) |
| **BBOB/COCO** | Black-Box Optimization | Minutes-hours | Active research | High - algorithm design | **High** | 24 functions, industry standard | [GitHub](https://github.com/numbbo/coco) |
| **Meta-Dataset** | Few-Shot (multi-domain) | 4-8 hr full | 15-20% | High - cross-domain transfer | **Medium** | 10 diverse datasets | [GitHub](https://github.com/google-research/meta-dataset) |
| **VTAB-1k** | Few-Shot (vision) | 2-4 hr | 10-15% | High - adaptation strategy | **Medium** | 19 tasks, 1000 examples each | [Google](https://google-research.github.io/task_adaptation/) |
| **TabPFN Benchmarks** | Tabular (inference) | Seconds | 5-10% | Very High - no training | **Medium** | Inference-only, rapid iteration | [GitHub](https://github.com/PriorLabs/TabPFN) |
| **MatBench** | Materials Science | 15-60 min/task | High | Excellent - crystal graphs | **Medium** | 13 tasks, domain knowledge | [matbench.materialsproject.org](https://matbench.materialsproject.org/) |
| **QM9** | Quantum Chemistry | 20-110 min | Moderate (1 kcal/mol) | Excellent - molecular ML | **Medium** | 134k molecules, 12 properties | [quantum-machine.org](https://quantum-machine.org/datasets/) |
| **SciCode** | Scientific Coding | N/A (code gen) | Very High (o1: 7.7%) | Excellent - LLM code gen | **Medium** | 80 problems, 6 domains | [scicode-bench.github.io](https://scicode-bench.github.io/) |
| **dcbench** | Data-Centric | 10-20 min | High (35%→72%) | High - slice discovery | **Medium** | Multiple DCAI tasks | [GitHub](https://github.com/data-centric-ai/dcbench) |
| **MinAtar** | RL (Atari-like) | Minutes on GPU | Medium | Excellent - fast Atari | **Medium** | 5 games, 5M frames | [GitHub](https://github.com/rlai-lab/MinAtar-Faster) |
| **Procgen** | RL (generalization) | Hours | High (~40% gap) | Good - procedural envs | **Medium** | 16 games, generalization test | [GitHub](https://github.com/openai/procgen) |
| **Atari 100k** | RL (sample efficiency) | ~1 hr | Medium | Good - algorithmic creativity | **Medium** | 100k steps constraint | [Overview](https://www.emergentmind.com/topics/atari-100k-benchmark) |
| **SWE-EVO** | Software Evolution | 30-60 min | Very High (GPT-5: 21%) | Excellent - multi-file | **Medium** | Harder than SWE-bench | [arXiv](https://arxiv.org/abs/2512.18470) |
| **LMR-BENCH** | Code Reproduction | 20-40 min | High | Excellent - scientific code | **Medium** | 28 NLP papers | [GitHub](https://github.com/du-nlp-lab/LMR-Bench) |
| **ORBIT** | Few-Shot (video) | 1-2 hr | 20-30% | High - personalization | **Medium** | Real-world, high variation | [GitHub](https://github.com/microsoft/ORBIT-Dataset) |
| **CEC Test Suites** | Numerical Optimization | Hours | Active | High - competition format | **Medium** | Annual competitions | [CEC 2025](https://competition-hub.github.io/GMPB-Competition/) |
| **HPO-B / YAHPO** | HPO | Seconds-minutes | Moderate | Medium - search strategy | **Low** | Surrogate-based, fast | [GitHub](https://github.com/slds-lmu/yahpo_gym) |
| **NAS-Bench-201** | NAS | Seconds (lookup) | Near-closed | Good - no training | **Low** | Pre-computed, 15k architectures | [GitHub](https://github.com/D-X-Y/NAS-Bench-201) |
| **Mini-ImageNet** | Few-Shot (image) | 1-2 hr | 5-10% | Medium - well-studied | **Low** | Standard baseline | [Leaderboard](https://few-shot.yyliu.net/miniimagenet.html) |

---

## Top 10 Recommendations for AIRA

| Rank | Benchmark | Why |
|------|-----------|-----|
| 1 | **ARC-AGI-2** | Flagship reasoning benchmark, massive headroom (o3 drops to 4%), tests compositional reasoning |
| 2 | **ScienceAgentBench** | 102 scientific tasks, containerized eval in 30min, directly tests iterative ML engineering |
| 3 | **HeuriGym** | Purpose-built for LLM agents, 9 optimization problems, clean evaluation |
| 4 | **Kaggle Playground** | Monthly fresh tasks, rewards feature engineering creativity, no data contamination |
| 5 | **SRBench** | Symbolic regression = algorithm discovery, multiple valid solutions, physics insight |
| 6 | **WRENCH** | LLM writes labeling functions programmatically, 22 datasets, huge design space |
| 7 | **Craftax** | Open-ended RL requiring deep exploration, 1B steps in <1hr, tests algorithm innovation |
| 8 | **PDEBench** | Scientific ML with clear metrics, architecture + loss design space, fast training |
| 9 | **ASAC** | Algorithm synthesis from formal specs, 136 tasks, only 8.8% solved by ChatGPT |
| 10 | **OpenML-CC18** | Gold standard tabular benchmark, semantic columns enable LLM feature engineering |

---

## Benchmarks to Avoid

| Benchmark | Reason |
|-----------|--------|
| HumanEval | Saturated (>94%) |
| GPQA Diamond | Saturated (~92%) |
| AIME 2024 | Contamination concerns, >90% |
| GSM8K | Near-ceiling |
| Omniglot | Saturated (~99%) |
| ImageNet training | Requires days of compute |
| Large transformers (DeBERTa, etc.) | Needs >1hr per fold |

---

## Novel Benchmark Directions (Gaps to Fill)

1. **Tabular meta-learning**: Few-shot for tabular data beyond TabPFN
2. **Scientific hypothesis generation**: Beyond MLGym's tuning to actual discovery
3. **Data debugging at scale**: Combining data-centric AI with large datasets
4. **Multi-step algorithm composition**: Chaining multiple optimization steps
5. **Cross-domain transfer for optimization**: Learning to optimize across problem types

---

## Sources

- ARC Prize: [arcprize.org](https://arcprize.org)
- ScienceAgentBench: [ICLR 2025](https://github.com/OSU-NLP-Group/ScienceAgentBench)
- HeuriGym: [arXiv 2506.07972](https://arxiv.org/abs/2506.07972)
- MLGym: [Meta FAIR](https://github.com/facebookresearch/MLGym)
- SRBench: [cavalab.org](https://cavalab.org/srbench/)
- OpenML: [openml.org](https://www.openml.org)
- TabArena: [arXiv 2506.16791](https://arxiv.org/abs/2506.16791)
- PDEBench: [NeurIPS 2022](https://github.com/pdebench/PDEBench)
- MoleculeNet: [moleculenet.org](https://moleculenet.org/)
- WRENCH: [GitHub](https://github.com/JieyuZ2/wrench)
- DataPerf: [MLCommons](https://github.com/mlcommons/dataperf)
- Brax/Gymnax: [Google](https://github.com/google/brax)
- Craftax: [arXiv 2402.16801](https://arxiv.org/abs/2402.16801)
- D4RL: [Farama](https://github.com/Farama-Foundation/D4RL)
- ASAC: [FSE 2024](https://dl.acm.org/doi/10.1145/3663529.3663802)
- BBOB/COCO: [numbbo](https://github.com/numbbo/coco)
