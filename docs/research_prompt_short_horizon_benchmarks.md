# Deep Research Prompt: Short-Horizon ML Agent Benchmarks

## Context

We are developing AIRA-dojo, an LLM-driven ML agent that autonomously solves machine learning tasks. Our current constraint analysis shows:

- **Step timeout**: 30 min (configurable to 2hr)
- **Total budget**: ~8hr effective
- **No checkpoint resumption**: Each step starts fresh
- **Single GPU**: A100 40GB available

On MLE-bench, only **25% of winning solutions** are trainable within 30min, and **38%** within 2hr. The remaining 62% require multi-hour to multi-day training (transformers, large ensembles, medical imaging pipelines).

**We need benchmarks where algorithmic creativity matters more than compute.**

---

## Research Objectives

Find challenges, benchmarks, and competitions where:

1. **Solution horizon is 0-1 hour** on a single GPU
2. **Algorithmic innovation > brute-force compute** (no "just train longer" solutions)
3. **Diversity of approaches is rewarded** (not one dominant paradigm)
4. **Agent capabilities are tested** (reasoning, iteration, debugging, not just hyperparameter tuning)

---

## Categories to Explore

### 1. Algorithm Discovery & Program Synthesis

- **ARC-AGI** (Abstraction and Reasoning Corpus): Few-shot visual reasoning, no training needed
- **Program synthesis benchmarks**: SyGuS, APPS, CodeContests
- **Algorithm design competitions**: CODEFORCES, AtCoder algorithmic problems
- **Symbolic regression**: AI Feynman, SRBench — discover equations from data
- **Neural architecture search micro-benchmarks**: NAS-Bench-101/201/301 (lookup tables, no training)

**Research questions**:
- What program synthesis benchmarks have short evaluation times?
- Are there "algorithm discovery" benchmarks beyond ARC?
- What symbolic AI challenges exist that reward reasoning over compute?

### 2. Tabular ML (AIRA's Sweet Spot)

- **OpenML-CC18**: 72 curated classification tasks, most solvable in minutes
- **TabZilla benchmark**: 36 datasets for tabular methods comparison
- **AutoML benchmarks**: AutoMLBenchmark (OpenML), AMLB
- **Kaggle Playground Series**: Monthly tabular competitions, typically fast
- **UCI repository classics**: Adult, Covertype, HIGGS (well-studied baselines)

**Research questions**:
- Which OpenML tasks have meaningful headroom above AutoML baselines?
- Are there tabular benchmarks that specifically reward feature engineering creativity?
- What's the state of tabular foundation model benchmarks (TabPFN, etc.)?

### 3. Few-Shot & Meta-Learning

- **Meta-Dataset**: Few-shot image classification (inference only, no training)
- **ORBIT**: Few-shot object recognition in videos
- **Mini-ImageNet / Tiered-ImageNet**: Standard few-shot benchmarks
- **VTAB** (Visual Task Adaptation Benchmark): Transfer learning evaluation

**Research questions**:
- Which few-shot benchmarks test adaptation strategy rather than base model size?
- Are there meta-learning benchmarks for tabular data?
- What's the fastest path to evaluating a new few-shot algorithm?

### 4. Scientific Computing & Simulation

- **SciML benchmarks**: PDEBench, NeuralOperator benchmarks
- **Molecular property prediction**: MoleculeNet (small molecules, fast inference)
- **Physics simulations**: Simple physics engines, n-body problems
- **Climate/weather**: WeatherBench (but check compute requirements)

**Research questions**:
- Which scientific ML tasks have fast training times (<1hr)?
- Are there "algorithm design for science" competitions?
- What about optimization benchmarks (BBOB, CEC functions)?

### 5. Optimization & Search

- **Black-box optimization**: BBOB (Black-Box Optimization Benchmarking), CEC test suites
- **Combinatorial optimization**: TSP, scheduling, packing problems
- **Hyperparameter optimization**: HPO-B benchmark, YAHPO Gym
- **Neural network optimization**: Learning to optimize benchmarks

**Research questions**:
- What optimization benchmarks reward novel algorithm design?
- Are there meta-learning benchmarks for optimization?
- What about evolutionary algorithm benchmarks?

### 6. Reinforcement Learning (Short-Horizon)

- **Procgen**: Procedurally generated games, fast episodes
- **MiniGrid**: Simple grid worlds, fast training
- **Brax**: Fast physics simulation in JAX
- **Isaac Gym**: GPU-accelerated simulation (check if single-GPU feasible)

**Research questions**:
- Which RL benchmarks can reach reasonable performance in <1hr?
- Are there RL algorithm design benchmarks (not just environment solving)?
- What about offline RL benchmarks with fixed datasets?

### 7. Data-Centric AI

- **DataPerf**: Benchmarks for data quality, selection, and curation
- **DCAI benchmarks**: Data-centric AI competition tasks
- **Active learning benchmarks**: Which samples to label next?
- **Data augmentation benchmarks**: AutoAugment-style search

**Research questions**:
- What benchmarks test data selection/curation strategies?
- Are there benchmarks for "learning what to learn"?
- What about dataset debugging/cleaning benchmarks?

### 8. Robustness & Distribution Shift

- **WILDS**: Distribution shift benchmark (check per-task compute)
- **ImageNet-C/R/A**: Corruption/rendition/adversarial robustness
- **DomainBed**: Domain generalization (some tasks are fast)

**Research questions**:
- Which robustness benchmarks have fast evaluation?
- Are there benchmarks that test adaptation strategies under shift?

### 9. Emerging/Niche Benchmarks

- **BIG-Bench**: LLM capability evaluation (many short tasks)
- **MATH benchmark**: Mathematical reasoning (compute is in inference)
- **LILA**: Language-informed latent actions
- **Embodied AI challenges**: AI2-THOR, Habitat (check compute)

**Research questions**:
- What new benchmarks have emerged in 2024-2026?
- Are there agent-specific ML benchmarks beyond SWE-bench?
- What about multi-agent or competitive benchmarks?

---

## Evaluation Criteria for Candidate Benchmarks

For each benchmark found, evaluate:

| Criterion | Question |
|-----------|----------|
| **Time to solution** | Can a competitive solution be trained in <1hr on single A100? |
| **Headroom** | Is there meaningful gap between baseline and SOTA? |
| **Diversity reward** | Do multiple distinct approaches achieve good scores? |
| **Agent-friendly** | Does iteration/debugging help, or is it one-shot? |
| **Metric clarity** | Is success clearly defined and measurable? |
| **Community activity** | Is there an active leaderboard or competition? |
| **Reproducibility** | Are baselines and evaluation code available? |

---

## Output Format

For each promising benchmark, provide:

```
### [Benchmark Name]

**Source**: [URL/paper]
**Domain**: [Category from above]
**Task**: [Brief description]

| Property | Value |
|----------|-------|
| Training time | [Estimate for competitive solution] |
| Hardware | [What's needed] |
| SOTA gap | [Baseline vs SOTA] |
| Diversity | [One paradigm vs many approaches] |
| Agent fit | [Why this suits iterative LLM agents] |

**Key insight**: [Why this is interesting for our setting]
```

---

## Anti-Patterns to Avoid

Do NOT recommend benchmarks where:

1. **Winning requires massive compute** (ImageNet training from scratch, large transformers)
2. **One architecture dominates** (BERT variants win everything)
3. **Dataset is the bottleneck** (>10GB downloads, >1M samples requiring full passes)
4. **Evaluation is slow** (>10min per submission)
5. **No headroom exists** (SOTA is 99%+, saturated)

---

## Starting Points for Research

1. **Papers With Code**: Filter by task, sort by recency, check compute requirements
2. **OpenML**: Explore task repository, filter by dataset size
3. **Kaggle**: Playground series, Getting Started competitions, Code competitions
4. **NeurIPS/ICML competition tracks**: 2023-2025 competition papers
5. **AutoML conference proceedings**: Recent benchmark proposals
6. **arXiv cs.LG**: Search "benchmark" + "efficient" / "few-shot" / "meta-learning"
7. **GitHub awesome lists**: awesome-ml-benchmarks, awesome-automl, etc.

---

## Specific Searches to Perform

1. "machine learning benchmark short training time"
2. "algorithm discovery benchmark AI"
3. "meta-learning benchmark fast evaluation"
4. "tabular machine learning benchmark 2024 2025"
5. "program synthesis benchmark evaluation time"
6. "AutoML benchmark small dataset"
7. "few-shot learning benchmark inference only"
8. "black-box optimization benchmark"
9. "neural architecture search without training"
10. "data-centric AI benchmark"

---

## Success Criteria

A successful research output should identify:

- **5-10 high-quality benchmarks** where AIRA can realistically compete
- **2-3 novel benchmark directions** that don't yet exist but should
- **Clear recommendations** for which benchmarks to prioritize for AIRA evaluation
- **Gaps in the benchmark landscape** that our research could fill


Present your results as a unified table, including fit, priority, interestigness, and links